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

#include "include/utils/tensor_utils.h"
#include <tuple>
#include <vector>
#include <algorithm>

namespace mindspore {
namespace tensor {
void SetPromise(const std::tuple<stub::StubNodePtr> &promises, const TensorPtr &tensor) {
  const auto &p = std::get<0>(promises);
  p->SetValue(tensor);
}

void FlattenOutputs(const ValuePtr &value, std::vector<TensorPtr> *outputs) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<Tensor>()) {
    outputs->emplace_back(value->cast<TensorPtr>());
  } else if (value->isa<ValueSequence>()) {
    auto seq = value->cast<ValueSequencePtr>();
    const auto &elements = seq->value();
    for (const auto &element : elements) {
      FlattenOutputs(element, outputs);
    }
  } else {
    MS_LOG(EXCEPTION) << "Not support type " << value->ToString();
  }
}

PyObject *TransformVectorOutput(const std::vector<TensorWrapper> &py_output) {
  PyObject *py_tuple = PyTuple_New(py_output.size());
  for (size_t i = 0; i < py_output.size(); i++) {
    PyTuple_SET_ITEM(py_tuple, i, py_output[i].value());
  }
  return py_tuple;
}

std::vector<stub::StubNodePtr> TransformVectorPromise(const std::vector<TensorWrapper> &py_output) {
  std::vector<stub::StubNodePtr> stubs;
  stubs.reserve(py_output.size());
  (void)std::transform(py_output.begin(), py_output.end(), std::back_inserter(stubs),
                       [](const TensorWrapper &wrapper) { return wrapper.MakeFuture(); });
  return stubs;
}

void SetPromise(const std::vector<stub::StubNodePtr> &t1, const std::vector<TensorPtr> &t2) {
  MS_ASSERT(t1.size() == t2.size());
  for (size_t i = 0; i < t1.size(); ++i) {
    t1[i]->SetValue(t2[i]);
  }
}

void SetException(const std::vector<stub::StubNodePtr> &t1) {
  (void)std::for_each(t1.begin(), t1.end(),
                      [](const stub::StubNodePtr &stub) { stub->SetException(std::current_exception()); });
}
}  // namespace tensor
}  // namespace mindspore
