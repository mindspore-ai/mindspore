/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "dataset/engine/datasetops/source/generator_op.h"
#include <iomanip>
#include "dataset/core/global_context.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/execution_tree.h"
#include "dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
GeneratorOp::Builder::Builder() {
  // Some arguments to the StorageOp constructor have a default argument that is taken
  // from the client config.
  build_buffer_size_ = kCfgRowsPerBuffer;
  build_op_connector_size_ = kCfgOpConnectorSize;
}

Status GeneratorOp::Builder::SanityCheck() {
  // Update queue size to fit the prefetch requirement
  MS_LOG(INFO) << "Generator operator sanity check, prefetch size is " << build_prefetch_size_ << ".";
  if (build_prefetch_size_ > 0) {
    build_op_connector_size_ = (build_prefetch_size_ + build_buffer_size_ - 1) / build_buffer_size_;
  }
  return Status::OK();
}

Status GeneratorOp::Builder::Build(std::shared_ptr<GeneratorOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<GeneratorOp>(build_generator_function_, build_column_names_, build_column_types_,
                                       build_prefetch_size_, build_buffer_size_, build_op_connector_size_);
  return (*ptr)->Init();
}

GeneratorOp::GeneratorOp(py::function generator_function, std::vector<std::string> column_names,
                         std::vector<DataType> column_types, int32_t prefetch_size, int32_t buffer_size,
                         int32_t connector_size)
    : PipelineOp(connector_size),
      generator_function_(generator_function),
      column_names_(column_names),
      column_types_(column_types),
      prefetch_size_(prefetch_size),
      buffer_size_(buffer_size),
      buffer_id_(0) {}

GeneratorOp::~GeneratorOp() { this->Dealloc(); }

void GeneratorOp::Print(std::ostream &out, bool show_all) const {
  // Always show the id and name as first line regardless if this summary or detailed print
  out << "(" << std::setw(2) << operator_id_ << ") <GeneratorOp>:";
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

void GeneratorOp::Dealloc() noexcept {
  // Setup GIL state
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();
  // GC the generator object within GIL
  (void)generator_.dec_ref();
  // Release GIL
  PyGILState_Release(gstate);
}

// Reentrant init method.
Status GeneratorOp::Init() {
  // Reset BufferID
  buffer_id_ = 0;
  // Setup column names map.
  if (column_names_map_.empty()) {
    for (int i = 0; i < column_names_.size(); ++i) {
      (void)column_names_map_.insert(std::make_pair(column_names_[i], i));
    }
  }
  Status ret;
  {
    // Acquire Python GIL
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kPythonInterpreterFailure, "Python Interpreter is finalized");
    }
    // Invoke the generatorFunction to get generator object
    try {
      generator_ = generator_function_();
    } catch (const py::error_already_set &e) {
      ret = Status(StatusCode::kPyFuncException, e.what());
    }
  }
  return ret;
}

Status GeneratorOp::PyRowToTensorRow(py::object py_data, TensorRow *tensor_row) {
  if (!py::isinstance<py::tuple>(py_data)) {
    return Status(StatusCode::kPyFuncException, __LINE__, __FILE__, "Generator should return a tuple of numpy arrays.");
  }
  py::tuple py_row = py_data.cast<py::tuple>();
  // Check if returned number of columns matches with column names
  if (py_row.size() != column_names_.size()) {
    return Status(StatusCode::kPyFuncException, __LINE__, __FILE__,
                  "Generator should return same number of numpy arrays as specified in column names.");
  }
  // Iterate over two containers simultaneously for memory copy
  for (int i = 0; i < py_row.size(); ++i) {
    py::object ret_py_ele = py_row[i];
    if (!py::isinstance<py::array>(ret_py_ele)) {
      return Status(StatusCode::kPyFuncException, __LINE__, __FILE__,
                    "Generator should return a tuple of numpy arrays.");
    }
    std::shared_ptr<Tensor> tensor;
    RETURN_IF_NOT_OK(Tensor::CreateTensor(&tensor, ret_py_ele.cast<py::array>()));
    if ((!column_types_.empty()) && (column_types_[i] != DataType::DE_UNKNOWN) &&
        (column_types_[i] != tensor->type())) {
      return Status(StatusCode::kPyFuncException, __LINE__, __FILE__, "Generator type check failed.");
    }
    tensor_row->push_back(tensor);
  }
  return Status(StatusCode::kOK, "");
}

Status GeneratorOp::FillBuffer(TensorQTable *tt) {
  for (int i = 0; i < buffer_size_; i++) {
    TensorRow row;
    RETURN_IF_NOT_OK(PyRowToTensorRow(generator_.attr("__next__")(), &row));
    tt->push_back(std::move(row));
  }
  return Status::OK();
}

// Entry point for Generator, called by launch()
// Note that this function is very easy to break because of the Python GIL mechanism
// The master thread has the following workflow
//
// while !eof:
//      Try:
//          Prepare one data buffer                                   GIL, Can throw
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
  std::unique_ptr<DataBuffer> fetched_buffer;
  bool eof = false;
  while (!eof) {
    // Create new buffer each iteration
    fetched_buffer = std::make_unique<DataBuffer>(buffer_id_++, DataBuffer::kDeBFlagNone);
    fetched_buffer->set_column_name_map(column_names_map_);
    std::unique_ptr<TensorQTable> fetched_table = std::make_unique<TensorQTable>();
    bool eoe = false;
    {
      py::gil_scoped_acquire gil_acquire;
      if (Py_IsInitialized() == 0) {
        return Status(StatusCode::kPythonInterpreterFailure, "Python Interpreter is finalized");
      }
      try {
        RETURN_IF_NOT_OK(FillBuffer(fetched_table.get()));
      } catch (py::error_already_set &e) {
        eoe = e.matches(PyExc_StopIteration);
        // Restore exception to python
        e.restore();
        // Pop up non StopIteration Python Exception
        if (!eoe) {
          return Status(StatusCode::kPyFuncException, __LINE__, __FILE__, e.what());
        }
      }
    }
    if (fetched_table->size() > 0) {
      fetched_buffer->set_tensor_table(std::move(fetched_table));
      RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(fetched_buffer)));
    }
    if (eoe) {
      // Push out EOE upon StopIteration exception from generator
      MS_LOG(INFO) << "Generator operator sends out EOE.";
      std::unique_ptr<DataBuffer> eoe_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
      RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eoe_buffer)));
      if (!BitTest(op_ctrl_flags_, kDeOpRepeated) || BitTest(op_ctrl_flags_, kDeOpLastRepeat)) {
        // If last repeat or not repeated, push out EOF and exit master loop
        MS_LOG(INFO) << "Generator operator sends out EOF.";
        std::unique_ptr<DataBuffer> eof_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF);
        RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(eof_buffer)));
        MS_LOG(INFO) << "Generator operator main execution loop complete.";
        eof = true;
      } else {
        // Waiting for repeatOp to start new epoch
        // If Reset() is called first by repeat op, this wait() will return right away.
        // If Reset() is not called yet, this wait() will block until reset.
        RETURN_IF_NOT_OK(wp_.Wait());
        // Clear the status of the wait post
        wp_.Clear();
      }
    }
  }
  return Status::OK();
}

Status GeneratorOp::Reset() {
  // Reset Op state
  RETURN_IF_NOT_OK(this->Init());
  // Wake up master thread
  wp_.Set();
  return Status(StatusCode::kOK, "GeneratorOp Reset Succeed");
}
}  // namespace dataset
}  // namespace mindspore
