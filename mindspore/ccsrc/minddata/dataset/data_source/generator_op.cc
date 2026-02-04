/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/data_source/generator_op.h"

#include <iomanip>
#include <utility>

#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/data_source/sampler/python_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/dataset/util/thread_bind_core_utils.h"

namespace mindspore {
namespace dataset {
GeneratorOp::GeneratorOp(const py::function &generator_function, std::vector<std::string> column_names,
                         std::vector<DataType> column_types, int32_t prefetch_size, int32_t connector_size,
                         std::shared_ptr<SamplerRT> sampler, int32_t num_parallel_workers, bool has_batch_sampler)
    : PipelineOp(connector_size, std::move(sampler)),
      column_names_(std::move(column_names)),
      column_types_(std::move(column_types)),
      prefetch_size_(prefetch_size),
      generator_counter_(0),
      num_parallel_workers_(num_parallel_workers),
      num_rows_sampled_{0},
      has_batch_sampler_(has_batch_sampler),
      num_samples_collected_(0),
      size_of_this_batch_(-1),
      batch_sizes_of_epoch_({}) {
  py::gil_scoped_acquire gil_acquire;
  generator_function_ = generator_function;
}

GeneratorOp::~GeneratorOp() {
  // we need to acquire gil before release python object
  py::gil_scoped_acquire gil_acquire;
  generator_ = py::object();
  generator_function_ = py::object();
}

void GeneratorOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nColumn names:\n";
    for (int i = 0; i < column_names_.size(); ++i) {
      out << "\n  " << column_names_[i];
    }
    out << "\n\n";
  }
}
// hand shake with Sampler, allow Sampler to call RandomAccessOp's functions to get NumRows
Status GeneratorOp::InitSampler() {
  if (sampler_ != nullptr) {
    // Let the sampler know if we are resetting the pipeline to a specific epoch (op_current_repeats_ > 0)
    // to mimic the behaviour in that state and have repeatability.
    // Note that number of repeats is used since in each epoch we may reset sampler multiple times.
    return sampler_->HandshakeRandomAccessOp(this, op_current_repeats_);
  }
  return Status::OK();
}

// Invoke the generatorFunction to get generator object
Status GeneratorOp::CreateGeneratorObject() {
  Status ret;
  {
    // Acquire Python GIL
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      RETURN_STATUS_ERROR(StatusCode::kMDPythonInterpreterFailure, "[Internal ERROR] Python Interpreter is finalized.");
    }
    try {
      py::array sample_ids;
      if (sampler_ != nullptr) {
        // Sampler is not null which means the source is RandomAccessible
        // get all samples and pass it to the Generator function
        RETURN_IF_NOT_OK(sampler_->GetAllIdsThenReset(&sample_ids));
        // If sampler is a user-defined python sampler, sample_ids will flow from python to c++ and back to python
        generator_ = generator_function_(sample_ids);
      } else {
        generator_ = generator_function_();
      }
    } catch (const py::error_already_set &e) {
      ret = Status(StatusCode::kMDPyFuncException, e.what());
    }
  }
  return ret;
}

Status GeneratorOp::Launch() {
  // Launch the python multiprocessing
  if (python_multiprocessing_runtime_) {
    py::gil_scoped_acquire gil_acquire;
    MS_LOG(DEBUG) << "Launch Python Multiprocessing for GeneratorOp: " << id();
    try {
      python_multiprocessing_runtime_->launch(id());
      py::dict worker_ids = python_multiprocessing_runtime_->get_worker_ids();
      if (worker_ids.empty()) {
        std::string msg = "The current thread/process id cannot be null.";
        RETURN_STATUS_ERROR(StatusCode::kMDPyFuncException, msg);
      }
      bool is_thread = py::cast<bool>(worker_ids["is_thread"]);
      std::vector<int64_t> worker_id = py::cast<std::vector<int64_t>>(worker_ids["worker_id"]);
      for (int i = 0; i < worker_id.size(); i++) {
        BindThreadCoreForMindDataOp("dataset::GeneratorOp", worker_id[i], is_thread);
      }
    } catch (py::error_already_set &e) {
      std::string traceback;
      try {
        // Construct python-like traceback
        py::list tb = py::module::import("traceback").attr("format_tb")(e.trace());
        traceback = "Traceback (most recent call last):\n";
        for (auto t : tb) {
          traceback += py::reinterpret_borrow<py::str>(t);
        }
        traceback += e.what();
      } catch (std::exception &) {
        // Back to original exception
        traceback = e.what();
      }

      RETURN_STATUS_ERROR(StatusCode::kMDPyFuncException, traceback);
    }
  }
  return DatasetOp::Launch();
}

Status GeneratorOp::Terminate() {
  // Terminate the python multiprocessing
  if (python_multiprocessing_runtime_) {
    MS_LOG(INFO) << "Terminate Python Multiprocessing for GeneratorOp: " << id();
    python_multiprocessing_runtime_->terminate();
  }
  return Status::OK();
}

// Reentrant init method.
Status GeneratorOp::Init() {
  RETURN_IF_NOT_OK(InitSampler());
  return CreateGeneratorObject();
}

Status GeneratorOp::PyRowToTensorRow(const py::object &py_data, TensorRow *tensor_row) {
  if (!py::isinstance<py::tuple>(py_data)) {
    RETURN_STATUS_ERROR(StatusCode::kMDPyFuncException,
                        "Invalid Python function, the 'source' of 'GeneratorDataset' should return a tuple of NumPy "
                        "arrays, but got " +
                          std::string(py_data.get_type().str()));
  }
  py::tuple py_row = py_data.cast<py::tuple>();
  // Check if returned number of columns matches with column names
  if (py_row.size() != column_names_.size()) {
    RETURN_STATUS_ERROR(
      StatusCode::kMDPyFuncException,
      "Invalid Python function, the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as "
      "specified in column_names, the size of column_names is:" +
        std::to_string(column_names_.size()) +
        " and number of returned NumPy array is:" + std::to_string(py_row.size()));
  }
  // Iterate over two containers simultaneously for memory copy
  for (int i = 0; i < py_row.size(); ++i) {
    py::object ret_py_ele = py_row[i];
    if (!py::isinstance<py::array>(ret_py_ele) && !py::isinstance<py::dict>(ret_py_ele)) {
      RETURN_STATUS_ERROR(
        StatusCode::kMDPyFuncException,
        "Invalid Python function, 'GeneratorDataset' should return a tuple of NumPy arrays or dictionaries, "
        "but got " +
          std::string(ret_py_ele.get_type().str()));
    }
    std::shared_ptr<Tensor> tensor;
    if (py::isinstance<py::dict>(ret_py_ele)) {
      RETURN_IF_NOT_OK(Tensor::CreateFromPythonObject(ret_py_ele.cast<py::dict>(), &tensor));
    } else {
      RETURN_IF_NOT_OK(Tensor::CreateFromNpArray(ret_py_ele.cast<py::array>(), &tensor));
    }
    if ((!column_types_.empty()) && (column_types_[i] != DataType::DE_UNKNOWN) &&
        (column_types_[i] != tensor->type())) {
      RETURN_STATUS_ERROR(StatusCode::kMDPyFuncException,
                          "Invalid Python function, type of returned data in 'GeneratorDataset' should be same with "
                          "specified column_types, but the type of returned data: " +
                            std::string(ret_py_ele.get_type().str()) +
                            ", specified column type: " + column_types_[i].ToString());
    }
    tensor_row->push_back(tensor);
  }
  return Status::OK();
}

Status GeneratorOp::CheckNumSamples() const {
  if (num_rows_sampled_ != -1 && num_rows_sampled_ != generator_counter_) {
    if (generator_counter_ == 0) {
      std::string msg =
        "Unable to fetch data from GeneratorDataset, try iterate the source function of GeneratorDataset or check"
        " value of num_epochs when create iterator.";
      RETURN_STATUS_ERROR(StatusCode::kMDPyFuncException, msg);
    }
    std::stringstream ss;
    ss << "The actual amount of data read from generator " << generator_counter_ << " is different from generator.len "
       << num_rows_sampled_ << ", you should adjust generator.len to make them match.";
    RETURN_STATUS_ERROR(StatusCode::kMDPyFuncException, ss.str());
  }
  return Status::OK();
}

Status GeneratorOp::CalculatedSampleId() {
  RETURN_IF_NOT_OK(InitSampler());
  py::gil_scoped_acquire gil_acquire;
  if (Py_IsInitialized() == 0) {
    RETURN_STATUS_ERROR(StatusCode::kMDPythonInterpreterFailure, "[Internal ERROR] Python Interpreter is finalized.");
  }
  try {
    py::array sample_ids;
    RETURN_IF_NOT_OK(sampler_->GetAllIdsThenReset(&sample_ids));
    auto buf_info = sample_ids.request();
    auto ptr_sample = static_cast<int64 *>(buf_info.ptr);
    for (size_t i = 0; i < buf_info.size; i++) {
      sample_ids_.push_back(ptr_sample[i]);
    }
  } catch (const py::error_already_set &e) {
    RETURN_STATUS_ERROR(StatusCode::kMDPyFuncException, e.what());
  }
  return Status::OK();
}

Status GeneratorOp::GetMappedIndex(int64_t index, int64_t *out_index) {
  if (index > sample_ids_.size() - 1) {
    RETURN_STATUS_UNEXPECTED("Index [" + std::to_string(index) + "] exceeded the number of data samples.");
  }
  if (index < 0) {
    RETURN_STATUS_UNEXPECTED("Index [" + std::to_string(index) + "] can not be a negative number.");
  }
  *out_index = sample_ids_[index];
  return Status::OK();
}

Status GeneratorOp::GetNextBatchSize() {
  if (has_batch_sampler_) {
    if (!batch_sizes_of_epoch_.empty()) {
      size_of_this_batch_ = batch_sizes_of_epoch_.front();
      batch_sizes_of_epoch_.pop_front();
      CHECK_FAIL_RETURN_UNEXPECTED(size_of_this_batch_ > 0, "Batch sampler should return a list, but got an integer.");
    }
  }
  return Status::OK();
}

Status GeneratorOp::GetNextEpochBatchSizes() {
  if (has_batch_sampler_) {
    auto batch_sampler = std::dynamic_pointer_cast<PythonSamplerRT>(sampler_);
    CHECK_FAIL_RETURN_UNEXPECTED(batch_sampler != nullptr,
                                 "Batch sampler of GeneratorDataset should be a Python sampler.");
    RETURN_IF_NOT_OK(batch_sampler->GetBatchSizes(&batch_sizes_of_epoch_));
    CHECK_FAIL_RETURN_UNEXPECTED(!batch_sizes_of_epoch_.empty(), "Batch sizes should not be empty.");
  }
  return Status::OK();
}

Status GeneratorOp::CheckAndSendEOB() {
  if (has_batch_sampler_) {
    if (num_samples_collected_ == size_of_this_batch_) {
      generator_counter_++;
      MS_LOG(DEBUG) << "Generator operator sends out EOB.";
      RETURN_IF_NOT_OK(out_connector_->SendEOB());
      num_samples_collected_ = 0;
    }
  }
  return Status::OK();
}

// Entry point for Generator, called by launch()
// Note that this function is very easy to break because of the Python GIL mechanism
// The master thread has the following workflow
//
// while !eof:
//      Try:
//          Prepare one data row                                   GIL, Can throw
//      Catch:
//          Fetch Python Exception                                    GIL
//          Check if Exception is StopIteration (EOE)                 GIL
//          Restore Python Exception                                  GIL
//          If not StopIteration:
//              Return Status PyFuncException
//
//      Push data buffer to connector                                 Block
//
//      if EOE
//          Push EOE                                                  Block
//          if more epoch:
//                Block until next epoch                              Block
//          else:
//                Push EOF                                            Block
//                eof = true
// Return Status OK
//
// Note that any modification of this function need to guarantee:
// 1. All "Require GIL" operations are protected by GIL
//    SegFault / Deadlock will occur if this condition is not fulfilled.
// 2. All "Block" operations are free from GIL, all block target are registered with tree.
//    Deadlock will occur if this condition is not fulfilled
// 3. No Python GC should be triggered outside of GIL.
//    SegFault will occur is this condition is not fulfilled
//
Status GeneratorOp::operator()() {
  // Handshake with TaskManager to synchronize thread creation
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(wp_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(Init());
  // calculate number samples after init the sampler
  num_rows_sampled_ = sampler_ ? sampler_->CalculateNumSamples(num_rows_) : num_rows_;

  RETURN_IF_NOT_OK(GetNextEpochBatchSizes());
  RETURN_IF_NOT_OK(GetNextBatchSize());

  bool eof = false;
  while (!eof) {
    // Create new row each iteration
    bool eoe = false;
    TensorRow new_row;
    {
      double row_timer_start = GetMilliTimeStamp();
      uint64_t start_time = GetSyscnt();
      py::gil_scoped_acquire gil_acquire;
      RETURN_IF_NOT_OK(CollectOpInfo(this->NameWithID(), "AcquireGIL", start_time));
      try {
        auto start = ProfilingTime::GetCurMilliSecond();
        start_time = GetSyscnt();
        RETURN_IF_NOT_OK(PyRowToTensorRow(generator_.attr("__next__")(), &new_row));
        RETURN_IF_NOT_OK(CollectOpInfo(this->NameWithID(), "__next__", start_time));
        new_row.TimerRecord(NameWithID(), RowTimer::kWorkerTime, {GetMilliTimeStamp() - row_timer_start});
        auto end = ProfilingTime::GetCurMilliSecond();
        if ((end - start) / num_parallel_workers_ > kGetItemTimeOutMilliSeconds) {
          MS_LOG(WARNING) << "Bad performance attention, it takes more than " +
                               std::to_string(kGetItemTimeOutMilliSeconds / 1000) +
                               " seconds to generator.__next__ new row, "
                               "which might cause `GetNext` timeout problem when sink_mode=True. You can increase the "
                               "parameter num_parallel_workers in GeneratorDataset / optimize the efficiency of "
                               "obtaining samples in the user-defined generator function.";
        }
        if (has_batch_sampler_) {
          num_samples_collected_++;
        } else {
          generator_counter_++;
        }
      } catch (py::error_already_set &e) {
        eoe = e.matches(PyExc_StopIteration);
        // Pop up non StopIteration Python Exception
        if (!eoe) {
          RETURN_IF_NOT_OK(
            CollectOpInfo(this->NameWithID(), "__next__", start_time, {{"TensorRowFlags", "Exception"}}));
          std::string traceback;
          try {
            // Construct python-like traceback
            py::list tb = py::module::import("traceback").attr("format_tb")(e.trace());
            traceback = "Traceback (most recent call last):\n";
            for (auto t : tb) {
              traceback += py::reinterpret_borrow<py::str>(t);
            }
            traceback += e.what();
          } catch (std::exception &) {
            // Back to original exception
            traceback = e.what();
          }

          RETURN_STATUS_ERROR(StatusCode::kMDPyFuncException, traceback);
        }
        RETURN_IF_NOT_OK(
          CollectOpInfo(this->NameWithID(), "__next__", start_time, {{"TensorRowFlags", "StopIteration"}}));

        // Check whether the number of samples is sufficient only when the first epoch
        if (op_current_repeats_ == 0) {
          RETURN_IF_NOT_OK(CheckNumSamples());
        }
      }
    }
    if (!new_row.empty()) {
      RETURN_IF_NOT_OK(out_connector_->Add(std::move(new_row)));
    }

    RETURN_IF_NOT_OK(CheckAndSendEOB());

    if (eoe) {
      // Push out EOE upon StopIteration exception from generator
      MS_LOG(DEBUG) << "Generator operator sends out EOE.";
      RETURN_IF_NOT_OK(out_connector_->SendEOE());
      if (has_batch_sampler_) {
        CHECK_FAIL_RETURN_UNEXPECTED(num_samples_collected_ == 0,
                                     "The sum of batch sizes should be equal to the total num samples.");
      }
      if (IsLastIteration()) {
        // If last repeat or not repeated, push out EOF and exit master loop
        MS_LOG(DEBUG) << "Generator operator sends out EOF.";
        RETURN_IF_NOT_OK(out_connector_->SendEOF());
        MS_LOG(DEBUG) << "Generator operator main execution loop complete.";
        eof = true;
      } else {
        // Waiting for repeatOp to start new epoch
        // If Reset() is called first by repeat op, this wait() will return right away.
        // If Reset() is not called yet, this wait() will block until reset.
        if (this->GetOpTotalRepeats() < 0) {
          RETURN_IF_NOT_OK(wp_.Wait());
          // Clear the status of the wait post
          wp_.Clear();
        } else {
          // Self-reset to start a new iteration
          RETURN_IF_NOT_OK(Reset());
        }
        RETURN_IF_NOT_OK(GetNextEpochBatchSizes());
      }
      UpdateRepeatAndEpochCounter();
    }
    if (num_samples_collected_ == 0 && !eof) {
      RETURN_IF_NOT_OK(GetNextBatchSize());
    }
  }
  return Status::OK();
}

Status GeneratorOp::Reset() {
  // Reset Op state
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  // Create new generator object
  RETURN_IF_NOT_OK(CreateGeneratorObject());
  // Once the master thread is waked up, that means a new epoch is started,
  // so the counter must be reset before master thread starts increasing it.
  generator_counter_ = 0;
  if (this->GetOpTotalRepeats() < 0) {
    // Wake up master thread
    wp_.Set();
  }
  return Status::OK();
}

void GeneratorOp::SetPythonMp(std::shared_ptr<PythonMultiprocessingRuntime> python_multiprocessing_runtime) {
  python_multiprocessing_runtime_ = std::move(python_multiprocessing_runtime);
}

Status GeneratorOp::ComputeColMap() {
  // Setup column names map (base class field)
  if (column_name_id_map_.empty()) {
    for (size_t i = 0; i < column_names_.size(); ++i) {
      column_name_id_map_[column_names_[i]] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status GeneratorOp::GetNextRowPullMode(TensorRow *const row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  if (!prepared_data_) {
    RETURN_IF_NOT_OK(Init());
    num_rows_sampled_ = sampler_ ? sampler_->CalculateNumSamples(num_rows_) : num_rows_;
    MS_LOG(DEBUG) << "num_rows_sampled: " << num_rows_sampled_;
    RETURN_IF_NOT_OK(GetNextEpochBatchSizes());
    prepared_data_ = true;
  }

  if (eof_received_) {
    *row = TensorRow(TensorRow::kFlagEOF);
    return Status::OK();
  }

  if (has_batch_sampler_) {
    if (num_samples_collected_ == 0) {
      RETURN_IF_NOT_OK(GetNextBatchSize());
    }
    if (num_samples_collected_ == size_of_this_batch_) {
      generator_counter_++;
      num_samples_collected_ = 0;
      *row = TensorRow(TensorRow::kFlagEOB);
      return Status::OK();
    }
  }

  bool eoe = false;
  TensorRow new_row;
  {
    py::gil_scoped_acquire gil_acquire;
    try {
      RETURN_IF_NOT_OK(PyRowToTensorRow(generator_.attr("__next__")(), &new_row));
      if (has_batch_sampler_) {
        num_samples_collected_++;
      } else {
        generator_counter_++;
      }
    } catch (py::error_already_set &e) {
      eoe = e.matches(PyExc_StopIteration);
      // Pop up non StopIteration Python Exception
      if (!eoe) {
        std::string traceback;
        try {
          // Construct python-like traceback
          py::list tb = py::module::import("traceback").attr("format_tb")(e.trace());
          traceback = "Traceback (most recent call last):\n";
          for (auto t : tb) {
            traceback += py::reinterpret_borrow<py::str>(t);
          }
          traceback += e.what();
        } catch (std::exception &) {
          // Back to original exception
          traceback = e.what();
        }

        RETURN_STATUS_ERROR(StatusCode::kMDPyFuncException, traceback);
      }

      // Check whether the number of samples is sufficient only when the first epoch
      if (op_current_repeats_ == 0) {
        RETURN_IF_NOT_OK(CheckNumSamples());
      }
    }
  }
  if (!new_row.empty()) {
    *row = std::move(new_row);
    return Status::OK();
  }

  if (eoe) {
    *row = TensorRow(TensorRow::kFlagEOE);
    if (has_batch_sampler_) {
      CHECK_FAIL_RETURN_UNEXPECTED(num_samples_collected_ == 0,
                                   "The sum of batch sizes should be equal to the total num samples.");
    }
    if (IsLastIteration()) {
      eof_received_ = true;
    } else {
      // Self-reset to start a new iteration
      RETURN_IF_NOT_OK(Reset());
      RETURN_IF_NOT_OK(GetNextEpochBatchSizes());
      UpdateRepeatAndEpochCounter();
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
