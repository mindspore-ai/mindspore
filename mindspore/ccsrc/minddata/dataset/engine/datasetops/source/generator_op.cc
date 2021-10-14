/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/generator_op.h"

#include <iomanip>

#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
GeneratorOp::GeneratorOp(py::function generator_function, std::vector<std::string> column_names,
                         std::vector<DataType> column_types, int32_t prefetch_size, int32_t connector_size,
                         std::shared_ptr<SamplerRT> sampler, uint32_t num_parallel_workers)
    : PipelineOp(connector_size, std::move(sampler)),
      generator_function_(generator_function),
      column_names_(column_names),
      column_types_(std::move(column_types)),
      prefetch_size_(prefetch_size),
      generator_counter_(0),
      num_parallel_workers_(num_parallel_workers) {}

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
  if (sampler_ != nullptr) return sampler_->HandshakeRandomAccessOp(this);
  return Status::OK();
}

// Invoke the generatorFunction to get generator object
Status GeneratorOp::CreateGeneratorObject() {
  Status ret;
  {
    // Acquire Python GIL
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kMDPythonInterpreterFailure, "Python Interpreter is finalized.");
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

// Reentrant init method.
Status GeneratorOp::Init() {
  RETURN_IF_NOT_OK(InitSampler());
  return CreateGeneratorObject();
}

Status GeneratorOp::PyRowToTensorRow(py::object py_data, TensorRow *tensor_row) {
  if (!py::isinstance<py::tuple>(py_data)) {
    return Status(StatusCode::kMDPyFuncException, __LINE__, __FILE__,
                  "Invalid data, Generator should return a tuple of NumPy arrays, currently returned is not a tuple.");
  }
  py::tuple py_row = py_data.cast<py::tuple>();
  // Check if returned number of columns matches with column names
  if (py_row.size() != column_names_.size()) {
    return Status(
      StatusCode::kMDPyFuncException, __LINE__, __FILE__,
      "Invalid data, Generator should return same number of NumPy arrays as specified in column_names, the size of"
      " column_names is:" +
        std::to_string(column_names_.size()) +
        " and number of returned NumPy array is:" + std::to_string(py_row.size()));
  }
  // Iterate over two containers simultaneously for memory copy
  for (int i = 0; i < py_row.size(); ++i) {
    py::object ret_py_ele = py_row[i];
    if (!py::isinstance<py::array>(ret_py_ele)) {
      return Status(StatusCode::kMDPyFuncException, __LINE__, __FILE__,
                    "Invalid data, Generator should return a tuple of NumPy arrays. Ensure each item in tuple that "
                    "returned by source function of GeneratorDataset be NumPy array.");
    }
    std::shared_ptr<Tensor> tensor;
    RETURN_IF_NOT_OK(Tensor::CreateFromNpArray(ret_py_ele.cast<py::array>(), &tensor));
    if ((!column_types_.empty()) && (column_types_[i] != DataType::DE_UNKNOWN) &&
        (column_types_[i] != tensor->type())) {
      return Status(StatusCode::kMDPyFuncException, __LINE__, __FILE__,
                    "Invalid data, type of returned data in GeneratorDataset is not same with specified column_types.");
    }
    tensor_row->push_back(tensor);
  }
  return Status(StatusCode::kSuccess, "");
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
  int64_t num_rows_sampled = sampler_ ? sampler_->CalculateNumSamples(num_rows_) : num_rows_;
  RETURN_IF_NOT_OK(Init());

  bool eof = false;
  while (!eof) {
    // Create new row each iteration
    bool eoe = false;
    TensorRow new_row;
    {
      py::gil_scoped_acquire gil_acquire;
      if (Py_IsInitialized() == 0) {
        return Status(StatusCode::kMDPythonInterpreterFailure, "Python Interpreter is finalized");
      }
      try {
        auto start = ProfilingTime::GetCurMilliSecond();
        RETURN_IF_NOT_OK(PyRowToTensorRow(generator_.attr("__next__")(), &new_row));
        auto end = ProfilingTime::GetCurMilliSecond();
        if ((end - start) / num_parallel_workers_ > kGetItemTimeOutMilliSeconds) {
          MS_LOG(WARNING) << "Bad performance attention, it takes more than 25 seconds to generator.__next__ new row, "
                             "which might cause `GetNext` timeout problem when sink_mode=True. You can increase the "
                             "parameter num_parallel_workers in GeneratorDataset / optimize the efficiency of "
                             "obtaining samples in the user-defined generator function.";
        }
        generator_counter_++;
      } catch (py::error_already_set &e) {
        eoe = e.matches(PyExc_StopIteration);
        // Restore exception to python
        e.restore();
        // Pop up non StopIteration Python Exception
        if (!eoe) {
          return Status(StatusCode::kMDPyFuncException, __LINE__, __FILE__, e.what());
        }
        if (num_rows_sampled != -1 && num_rows_sampled != generator_counter_) {
          if (generator_counter_ == 0) {
            std::string msg =
              "Unable to fetch data from GeneratorDataset, try iterate the source function of GeneratorDataset or check"
              " value of num_epochs when create iterator.";
            return Status(StatusCode::kMDPyFuncException, __LINE__, __FILE__, msg);
          }
          std::stringstream ss;
          ss << "The actual amount of data read from generator " << generator_counter_
             << " is different from generator.len " << num_rows_sampled
             << ", you should adjust generator.len to make them match.";
          return Status(StatusCode::kMDPyFuncException, __LINE__, __FILE__, ss.str());
        }
      }
    }
    if (!new_row.empty()) RETURN_IF_NOT_OK(out_connector_->Add(std::move(new_row)));

    if (eoe) {
      // Push out EOE upon StopIteration exception from generator
      MS_LOG(DEBUG) << "Generator operator sends out EOE.";
      RETURN_IF_NOT_OK(out_connector_->SendEOE());
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
      }
      UpdateRepeatAndEpochCounter();
    }
  }
  return Status::OK();
}

Status GeneratorOp::Reset() {
  // Reset Op state
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  // Create new generator object
  RETURN_IF_NOT_OK(CreateGeneratorObject());
  if (this->GetOpTotalRepeats() < 0) {
    // Wake up master thread
    wp_.Set();
  }
  generator_counter_ = 0;
  return Status(StatusCode::kSuccess, "GeneratorOp Reset Succeed");
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
}  // namespace dataset
}  // namespace mindspore
