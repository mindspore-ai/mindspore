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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_PY_FUNC_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_PY_FUNC_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/message_queue.h"
#include "minddata/dataset/core/shared_memory_queue.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {

Status ConvertNumpyToTensor(const py::object &py_obj, TensorRow *output);

Status ConvertPythonToTensor(const py::object &py_obj, TensorRow *output);

class PyFuncOp : public TensorOp {
 public:
  explicit PyFuncOp(const py::function &func);

  explicit PyFuncOp(const py::function &func, DataType::Type output_type);

  explicit PyFuncOp(std::shared_ptr<PyFuncOp> op);

  ~PyFuncOp() override;

  uint32_t NumInput() override { return 0; }

  uint32_t NumOutput() override { return 0; }

  // Compute function for n-n mapping.
  Status Compute(const TensorRow &input, TensorRow *output) override;

  /// \brief Function to convert a primitive type py::object to a TensorRow
  /// \notes Changes the py::object to a tensor with corresponding C++ DataType based on output_type_ and adds it to a
  ///    TensorRow. This function is used inside Compute.
  /// \param[in] ret_py_obj The python object we want to cast
  /// \param[output] The TensorRow output
  /// \return Status
  Status CastOutput(const py::object &ret_py_obj, TensorRow *output);

  std::string Name() const override { return kPyFuncOp; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::vector<std::shared_ptr<TensorOperation>> *result);

  /// \brief Check whether this pyfunc op is deterministic
  /// \return True if this pyfunc op is random
  bool IsRandom();

  Status ReleaseResource() override {
    {
      py::gil_scoped_acquire gil_acquire;
      if (py::hasattr(py_func_ptr_, "release_resource")) {
        // release the executor which is used in the PyFunc
        // the PyFunc maybe contains vision/nlp/audio transform
        (void)py_func_ptr_.attr("release_resource")();
      }
    }
    return Status::OK();
  }

#if !defined(_WIN32) && !defined(_WIN64)
  /// \brief Create message queue and shared memory queue
  // called in MapOp::SetPythonMp()
  void CreateMsgQueueAndShmQueue(const int32_t &thread_idx, const key_t &key);

  Status GetOrCreateMessageQueueID();

  /// \brief Set the process id when multiprocess mode
  // called in MapOp::Launch()
  void SetProcessID(int32_t process_id);
#endif

 private:
#if !defined(_WIN32) && !defined(_WIN64)
  /// \brief Execute the operations with python multiprocessing workers
  Status ComputeWithWorker(const TensorRow &input, TensorRow *output);
#endif

  /// \brief Execute the operations with C++ thread
  Status ComputeWithThread(const TensorRow &input, TensorRow *output);

  py::function py_func_ptr_;
  DataType::Type output_type_;

#if !defined(_WIN32) && !defined(_WIN64)
  int32_t worker_pid_;                       // process id of the worker
  int32_t thread_idx_;                       // the thread idx which corresponds to python process worker
  std::shared_ptr<MessageQueue> msg_queue_;  // MapOp with PyFunc in process mode will use msg_queue to transfer data
  std::shared_ptr<SharedMemoryQueue>
    shm_queue_;  // MapOp with PyFunc in process mode will use shm_queue to transfer data

  // monitor the worker process
  // the variables are hold in PyFuncOp which is corresponds one-to-one with the map thread
  std::mutex monitor_mtx_;
  std::condition_variable monitor_cv_;
  bool monitor_exit_flag_;
#endif
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_PY_FUNC_OP_H_
