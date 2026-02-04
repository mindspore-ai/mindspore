/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "minddata/dataset/kernels/py_func_op.h"

#include <memory>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/general/transform/transforms_ir.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/command.h"
#include "minddata/dataset/util/sig_handler.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
Status ConvertNumpyToTensor(const py::object &py_obj, TensorRow *output) {
  RETURN_UNEXPECTED_IF_NULL(output);
  std::shared_ptr<Tensor> out;
  // Python object like bool, int, float, list or tuple can also be converted
  // to a NumPy array by the following cast, but the data type will be unknown
  // if it is not a valid NumPy object
  RETURN_IF_NOT_OK(Tensor::CreateFromNpArray(py_obj.cast<py::array>(), &out));
  output->push_back(out);
  return Status::OK();
}

Status ConvertPythonToTensor(const py::object &py_obj, TensorRow *output) {
  RETURN_UNEXPECTED_IF_NULL(output);
  // Python objects such as dictionary are converted to a tensor
  // Note that the tensor will hold a reference to the python object while
  // the python object will be kept alive in Python layer.
  std::shared_ptr<Tensor> out;
  RETURN_IF_NOT_OK(Tensor::CreateFromPythonObject(py_obj, &out));
  output->push_back(out);
  return Status::OK();
}

#if !defined(_WIN32) && !defined(_WIN64)
PyFuncOp::PyFuncOp(const py::function &func)
    : output_type_(DataType::DE_UNKNOWN),
      worker_pid_(-1),
      thread_idx_(-1),
      msg_queue_(nullptr),
      shm_queue_(nullptr),
      monitor_exit_flag_(true) {
  py::gil_scoped_acquire gil_acquire;
  py_func_ptr_ = func;
}

PyFuncOp::PyFuncOp(const py::function &func, DataType::Type output_type)
    : output_type_(output_type),
      worker_pid_(-1),
      thread_idx_(-1),
      msg_queue_(nullptr),
      shm_queue_(nullptr),
      monitor_exit_flag_(true) {
  py::gil_scoped_acquire gil_acquire;
  py_func_ptr_ = func;
}

PyFuncOp::PyFuncOp(std::shared_ptr<PyFuncOp> op)
    : worker_pid_(-1), thread_idx_(-1), msg_queue_(nullptr), shm_queue_(nullptr), monitor_exit_flag_(true) {
  py::gil_scoped_acquire gil_acquire;
  py_func_ptr_ = op->py_func_ptr_;
  output_type_ = op->output_type_;
}
#else
PyFuncOp::PyFuncOp(const py::function &func) : output_type_(DataType::DE_UNKNOWN) {
  py::gil_scoped_acquire gil_acquire;
  py_func_ptr_ = func;
}

PyFuncOp::PyFuncOp(const py::function &func, DataType::Type output_type) : output_type_(output_type) {
  py::gil_scoped_acquire gil_acquire;
  py_func_ptr_ = func;
}

PyFuncOp::PyFuncOp(std::shared_ptr<PyFuncOp> op) {
  py::gil_scoped_acquire gil_acquire;
  py_func_ptr_ = op->py_func_ptr_;
  output_type_ = op->output_type_;
}
#endif

PyFuncOp::~PyFuncOp() {
  {
    py::gil_scoped_acquire gil_acquire;
    py_func_ptr_ = py::object();
  }
#if !defined(_WIN32) && !defined(_WIN64)
  if (shm_queue_) {
    shm_queue_->SetReleaseFlag(true);
  }
  if (msg_queue_) {
    msg_queue_->SetReleaseFlag(true);
  }
#endif
}

#if !defined(_WIN32) && !defined(_WIN64)
Status PyFuncOp::ComputeWithWorker(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  uint64_t start_time = GetSyscnt();
  RETURN_IF_NOT_OK(CollectOpInfo(this->Name(), "ExecutePyFunc", start_time));

  // >> send procedure >>
  // 1. convert TensorRow to shared memory
  RETURN_IF_NOT_OK(shm_queue_->FromTensorRow(input));

  // message queue id maybe -1 when process is started in spawn mode and independent mode
  if (msg_queue_->msg_queue_id_ == -1) {
    RETURN_IF_NOT_OK(msg_queue_->GetOrCreateMessageQueueID());
  }

  std::string current_pid = std::to_string(getpid());
  // register the shm_id & msg_id by MainProcessPID_WorkerPID_"PyFuncOp"
  RegisterShmIDAndMsgID(current_pid + "_" + std::to_string(worker_pid_) + "_PyFuncOp", shm_queue_->GetShmID(),
                        msg_queue_->msg_queue_id_);

  // 2. send message queue which contains shared memory to Python Process Worker
  auto ret_status = msg_queue_->MsgSnd(kMasterSendDataMsg, shm_queue_->GetShmID(), shm_queue_->GetShmSize());

  RegisterShmIDAndMsgID(current_pid + "_" + std::to_string(worker_pid_) + "_PyFuncOp", shm_queue_->GetShmID(),
                        msg_queue_->msg_queue_id_);

  if (ret_status != Status::OK()) {
    return ret_status;
  }

  MS_LOG(INFO) << "Map thread " << std::to_string(thread_idx_)
               << " sends sample to python process worker: " << worker_pid_
               << " through shm_id: " << std::to_string(shm_queue_->GetShmID())
               << " with shm_size: " << std::to_string(shm_queue_->GetShmSize());

  // monitor thread
  // If no data is obtained from the worker process within the time
  // specified by get_multiprocessing_timeout_interval, a warning log is generated.
  monitor_exit_flag_ = false;
  auto monitor_thread =
    std::thread([this]() { MonitorLoop(&monitor_mtx_, &monitor_cv_, &monitor_exit_flag_, worker_pid_, "Map"); });

  // >> receive procedure >>
  // 1. get message queue which contains shared memory from Python Process Worker
  auto rcv_ret = msg_queue_->MsgRcv(kWorkerSendDataMsg);

  RegisterShmIDAndMsgID(current_pid + "_" + std::to_string(worker_pid_) + "_PyFuncOp", msg_queue_->shm_id_,
                        msg_queue_->msg_queue_id_);

  // got the data from worker process, end the monitor thread
  {
    std::lock_guard<std::mutex> lock(monitor_mtx_);
    monitor_exit_flag_ = true;
  }
  monitor_cv_.notify_all();
  monitor_thread.join();

  RETURN_IF_NOT_OK(rcv_ret);

  if (msg_queue_->MessageQueueState() == MessageState::kReleased) {
    RETURN_STATUS_UNEXPECTED("The msg queue had been released by worker process, map thread: " +
                             std::to_string(thread_idx_) + ", process worker: " + std::to_string(worker_pid_));
  }

  if (msg_queue_->GetErrorStatus()) {
    // got err from Python Process Worker
    auto ret = msg_queue_->DeserializeStatus();
    // for ds.config.set_error_samples_mode(...) scenario, we should clear the err message for next normal sample
    if (GlobalContext::config_manager()->error_samples_mode() != ErrorSamplesMode::kReturn) {
      msg_queue_->ClearErrMsg();
    }
    return ret;
  }
  MS_LOG(INFO) << "Map thread " << std::to_string(thread_idx_)
               << " receives sample from python process worker: " << worker_pid_
               << " through shm_id: " << std::to_string(msg_queue_->shm_id_)
               << " with shm_size: " << std::to_string(msg_queue_->shm_size_);

  // 2. construct shared memory to TensorRow
  RETURN_IF_NOT_OK(shm_queue_->ToTensorRow(output, msg_queue_->shm_id_, msg_queue_->shm_size_));
  return Status::OK();
}
#endif

Status PyFuncOp::ComputeWithThread(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  // map with multi threads
  uint64_t start_time = GetSyscnt();
  // Acquire Python GIL
  py::gil_scoped_acquire gil_acquire;
  RETURN_IF_NOT_OK(CollectOpInfo(this->Name(), "AcquireGIL", start_time));
  if (Py_IsInitialized() == 0) {
    return Status(StatusCode::kMDPythonInterpreterFailure, "Python Interpreter is finalized");
  }

  try {
    // Transform input tensor vector into numpy array vector
    py::object ret_py_obj;
    if (input.size() > 0) {
      py::tuple input_args(input.size());
      for (size_t i = 0; i < input.size(); i++) {
        if (input.at(i)->type().IsPython()) {
          py::dict new_data;
          RETURN_IF_NOT_OK(input.at(i)->GetDataAsPythonObject(&new_data));
          input_args[i] = new_data;
        } else {
          py::array new_data;
          RETURN_IF_NOT_OK(input.at(i)->GetDataAsNumpy(&new_data));
          // possible memcpy here
          input_args[i] = new_data;
        }
      }
      // Invoke python function
      ret_py_obj = this->py_func_ptr_(*input_args);
    } else {
      ret_py_obj = this->py_func_ptr_();
    }
    if (output_type_ != DataType::DE_UNKNOWN) {
      RETURN_IF_NOT_OK(CastOutput(ret_py_obj, output));
    } else {
      // scenario 1: map multi-processing, subprocess stop first and will get none
      // scenario 2: thread mode, user pyfunc return none
      if (ret_py_obj.is_none()) {
        std::string error_msg =
          "The subprocess of dataset may exit unexpected or be killed, "
          "main process will exit. If this is not an artificial operation, you can use "
          "mindspore.dataset.config.set_enable_watchdog(False) to block this error.";
        RETURN_STATUS_UNEXPECTED("Got None from Python object. " + error_msg);
      } else if (py::isinstance<py::tuple>(ret_py_obj)) {
        // In case of a n-m mapping, the return value will be a tuple of numpy arrays
        auto ret_py_tuple = ret_py_obj.cast<py::tuple>();
        // Iterate over two containers simultaneously for memory copy
        for (size_t i = 0; i < ret_py_tuple.size(); i++) {
          py::object ret_py_ele = ret_py_tuple[i];
          // Object is none if pyfunc timeout
          if (ret_py_ele.is_none()) {
            MS_LOG(INFO) << "Expected pyfunc to return NumPy array(s) or Python dict(s), but got None. "
                            "If python_multiprocessing is True, it may be due to pyfunc execution timeout.";
            return STATUS_ERROR(StatusCode::kMDTimeOut,
                                "Expect pyfunc to return numpy array(s), but got None. If python_multiprocessing is "
                                "True, it maybe due to pyfunc execution timeout.");
          } else if (py::isinstance<py::dict>(ret_py_ele)) {
            RETURN_IF_NOT_OK(ConvertPythonToTensor(ret_py_ele, output));
          } else {
            RETURN_IF_NOT_OK(ConvertNumpyToTensor(ret_py_ele, output));
          }
        }
      } else {
        // In case of a n-1 mapping, the return value will be a numpy array or a python object
        // Note that for Python dictionaries, only a reference will be stored in tensor.
        if (py::isinstance<py::dict>(ret_py_obj)) {
          RETURN_IF_NOT_OK(ConvertPythonToTensor(ret_py_obj, output));
        } else {
          RETURN_IF_NOT_OK(ConvertNumpyToTensor(ret_py_obj, output));
        }
      }
    }
  } catch (const py::error_already_set &e) {
    return Status(StatusCode::kMDPyFuncException, e.what());
  }
  return Status::OK();
}

Status PyFuncOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
#if !defined(_WIN32) && !defined(_WIN64)
  if (msg_queue_ != nullptr) {  // map with python_multiprocessing=True
    return ComputeWithWorker(input, output);
  }
#endif
  return ComputeWithThread(input, output);  // map with thread
}

Status PyFuncOp::CastOutput(const py::object &ret_py_obj, TensorRow *output) {
  RETURN_UNEXPECTED_IF_NULL(output);
  try {
    std::shared_ptr<Tensor> out;
    switch (output_type_) {
      case DataType::DE_INT32:
        RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({1}), DataType(DataType::DE_INT32), &out));
        RETURN_IF_NOT_OK(out->SetItemAt({0}, ret_py_obj.cast<int32_t>()));
        break;
      case DataType::DE_BOOL:
        RETURN_IF_NOT_OK(Tensor::CreateScalar(ret_py_obj.cast<bool>(), &out));
        break;
      default:
        RETURN_STATUS_UNEXPECTED("No cast for the specified DataType was found.");
    }
    output->push_back(out);
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED(e.what());
  }
  return Status::OK();
}

Status PyFuncOp::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  {
    py::gil_scoped_acquire gil_acquire;
    if (py_func_ptr_.attr("to_json")) {
      args = nlohmann::json::parse(py_func_ptr_.attr("to_json")().cast<std::string>());
    }
  }
  *out_json = args;
  return Status::OK();
}

Status PyFuncOp::from_json(nlohmann::json json_obj, std::vector<std::shared_ptr<TensorOperation>> *result) {
  RETURN_UNEXPECTED_IF_NULL(result);
  std::vector<std::shared_ptr<TensorOperation>> output;
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "tensor_op_name", kPyFuncOp));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "tensor_op_params", kPyFuncOp));
  std::string op_name = json_obj["tensor_op_name"];
  nlohmann::json op_params = json_obj["tensor_op_params"];
  std::string python_module = json_obj["python_module"];
  std::shared_ptr<TensorOperation> operation = nullptr;
  py::function py_func =
    py::module::import(python_module.c_str()).attr(op_name.c_str()).attr("from_json")(op_params.dump());
  operation = std::make_shared<transforms::PreBuiltOperation>(std::make_shared<PyFuncOp>(py_func));
  output.push_back(operation);
  *result = output;
  return Status::OK();
}

bool PyFuncOp::IsRandom() {
  bool random = true;
  if (py::hasattr(py_func_ptr_, "random") &&
      !static_cast<bool>(py::reinterpret_borrow<py::bool_>(py_func_ptr_.attr("random")))) {
    random = false;
  }
  return random;
}

#if !defined(_WIN32) && !defined(_WIN64)
void PyFuncOp::CreateMsgQueueAndShmQueue(const int32_t &thread_idx, const key_t &key) {
  MS_LOG(INFO) << "Create msg queue and shm queue for pyfunc map thread: " << std::to_string(thread_idx)
               << " with ftok_key: " << std::to_string(key);
  thread_idx_ = thread_idx;
  msg_queue_ = std::make_shared<MessageQueue>(key);
  msg_queue_->SetReleaseFlag(false);
  shm_queue_ = std::make_shared<SharedMemoryQueue>(key);
  shm_queue_->SetReleaseFlag(false);
}

Status PyFuncOp::GetOrCreateMessageQueueID() {
  RETURN_IF_NOT_OK(msg_queue_->GetOrCreateMessageQueueID());
  return Status::OK();
}

void PyFuncOp::SetProcessID(int32_t process_id) {
  MS_LOG(INFO) << "Set the process id: " << process_id << " to pyfunc map thread: " << thread_idx_;
  worker_pid_ = process_id;

  std::string current_pid = std::to_string(getpid());
  // register the shm_id & msg_id by MainProcessPID_WorkerPID
  RegisterShmIDAndMsgID(current_pid + "_" + std::to_string(worker_pid_) + "_PyFuncOp", shm_queue_->GetShmID(),
                        msg_queue_->msg_queue_id_);
}
#endif
}  // namespace dataset
}  // namespace mindspore
