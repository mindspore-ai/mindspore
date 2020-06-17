/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "dataset/engine/datasetops/source/sampler/python_sampler.h"

#include <memory>

namespace mindspore {
namespace dataset {

PythonSampler::PythonSampler(int64_t num_samples, py::object py_sampler_instance, int64_t samples_per_buffer)
    : Sampler(num_samples, samples_per_buffer), py_sampler_instance(py_sampler_instance), need_to_reset_(false) {}

Status PythonSampler::GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) {
  if (need_to_reset_) {
    (*out_buffer) = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }

    std::shared_ptr<Tensor> sample_ids;
    {
      py::gil_scoped_acquire gil_acquire;
      (*out_buffer) = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagNone);
      if (Py_IsInitialized() == 0) {
        return Status(StatusCode::kPythonInterpreterFailure, "Python Interpreter is finalized");
      }
      try {
        py::object py_ret = py_sampler_instance.attr("_get_indices")();
        py::array np_sample_ids = py_ret.cast<py::array>();
        Tensor::CreateTensor(&sample_ids, np_sample_ids);  // copy numpy to tensor

        if (HasChildSampler()) {
          for (auto it = sample_ids->begin<int64_t>(); it != sample_ids->end<int64_t>(); ++it) {
            int64_t associated_child_id = 0;
            RETURN_IF_NOT_OK(GetAssociatedChildId(&associated_child_id, associated_child_id));
            *it = associated_child_id;
          }
        }
      } catch (const py::error_already_set &e) {
        return Status(StatusCode::kPyFuncException, e.what());
      } catch (const py::cast_error &e) {
        return Status(StatusCode::kPyFuncException, "Python Sampler iterator should return integer index");
      }
    }
    TensorRow row(1, sample_ids);
    (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, row));
    need_to_reset_ = true;
  }
  return Status::OK();
}

Status PythonSampler::InitSampler() {
  CHECK_FAIL_RETURN_UNEXPECTED(num_rows_ > 0, "ERROR num_rows_ should be greater than 0");
  // Special value of 0 for num_samples means that the user wants to sample the entire set of data.
  // If the user asked to sample more rows than exists in the dataset, adjust the num_samples accordingly.
  if (num_samples_ == 0 || num_samples_ > num_rows_) {
    num_samples_ = num_rows_;
  }
  {
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kPythonInterpreterFailure, "Python Interpreter is finalized");
    }
    try {
      py_sampler_instance.attr("_handshake")(num_rows_, num_samples_);
    } catch (const py::error_already_set &e) {
      return Status(StatusCode::kPyFuncException, e.what());
    }
  }
  return Status::OK();
}

Status PythonSampler::ResetSampler() {
  CHECK_FAIL_RETURN_UNEXPECTED(need_to_reset_, "ERROR Reset() called not at end of an epoch");
  need_to_reset_ = false;
  py::gil_scoped_acquire gil_acquire;
  if (Py_IsInitialized() == 0) {
    return Status(StatusCode::kPythonInterpreterFailure, "Python Interpreter is finalized");
  }
  try {
    py_sampler_instance.attr("reset")();
  } catch (const py::error_already_set &e) {
    return Status(StatusCode::kPyFuncException, e.what());
  }

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->ResetSampler());
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
