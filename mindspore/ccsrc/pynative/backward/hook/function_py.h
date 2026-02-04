/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_PY_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_PY_H_

#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "pybind11/pybind11.h"
#include "include/utils/pynative/variable.h"
#include "include/utils/visible.h"
#include "pynative/backward/grad_utils.h"
#include "pynative/backward/hook/custom_function.h"

namespace mindspore::pynative::autograd {
namespace py = pybind11;

inline bool ensure_obj_tuple(py::object *obj) {
  if (py::isinstance<py::tuple>(*obj)) {
    return false;
  }
  py::tuple tuple = py::make_tuple(*obj);
  if (!tuple) {
    MS_LOG(EXCEPTION) << "tuple is null.";
  }
  *obj = tuple;
  return true;
}

using TensorPtrSet = std::unordered_set<tensor::TensorPtr>;
using TensorPtrList = std::vector<tensor::TensorPtr>;

struct FunctionContext {
  // The input of apply function
  ValuePtrList inputs;

  // The output of forward function in flatten format
  ValuePtrList flatten_outputs;

  // The input type of apply function input
  std::vector<InputType> input_value_grad_type;

  // Set of input tensors
  TensorPtrSet input_base_tensors;
  // Set of dirty tensors
  TensorPtrSet dirty_tensors;
  // Set of non_diff tensors
  TensorPtrSet non_diff_tensors;
  // to_save tensors
  TensorPtrList to_save_tensors;
  PyBackwardNodePtr grad_node;
};

struct FunctionBase {
  PyObject_HEAD
    // A python tuple return to use to indicate whether inputs need grad.
    PyObject *needs_input_grad;

  // The context carry tensors from forward function to backward function. Result of `save_for_backward` function.
  PyObject *saved_tensors;

  // The tensors that are not differentiable decided by use. Result of `mark_non_differentiable` function.
  PyObject *non_differentiable;

  // The tensor that have been modified. Result of `mark_dirty` function.
  PyObject *dirty_tensors;

  // The flag indicate whether to materialize none output grad tensors into
  // tensors full of zeros.
  bool materialize_grads = true;

  // True is the input is tensor
  std::vector<bool> is_tensor_input;

  // This is used for unpack saved tensors.
  std::weak_ptr<PyBackwardNode> weak_grad_node;
};

using FunctionPtr = std::shared_ptr<FunctionBase>;
}  // namespace mindspore::pynative::autograd
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_PY_H_
