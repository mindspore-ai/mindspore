/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "include/common/fallback.h"

#include <queue>

#include "include/common/utils/python_adapter.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace fallback {
static std::queue<py::object> py_execute_output_queue = std::queue<py::object>();

bool HasPyExecuteOutput() { return !py_execute_output_queue.empty(); }

py::object PopPyExecuteOutput() {
  auto output = py_execute_output_queue.front();
  MS_LOG(DEBUG) << "output: " << output;
  py_execute_output_queue.pop();
  return output;
}

void PushPyExecuteOutput(const py::object &output) {
  MS_LOG(DEBUG) << "output: " << output;
  py_execute_output_queue.push(output);
}

bool CheckListToMemory(const py::list &obj) {
  // A list object can be passed to raw memory and used by other operator if:
  //   1. The length of list is not empty.
  //   2. The list is not nested.
  //   3. The list only contains Scalar or Tensor elements.
  if (py::len(obj) == 0) {
    return false;
  }
  return std::all_of(obj.begin(), obj.end(), [](const auto &element) {
    return py::isinstance<py::bool_>(element) || py::isinstance<py::int_>(element) ||
           py::isinstance<py::float_>(element) || py::isinstance<tensor::Tensor>(element);
  });
}

abstract::AbstractListPtr GenerateAbstractList(const BaseShapePtr &base_shape, const TypePtr &type, bool is_dyn_shape) {
  // Generate AbstractList for PyExecute node.
  MS_EXCEPTION_IF_NULL(base_shape);
  MS_EXCEPTION_IF_NULL(type);
  if (!base_shape->isa<abstract::ListShape>()) {
    MS_INTERNAL_EXCEPTION(TypeError) << "For GenerateAbstractList, the shape should be list shape but got: "
                                     << base_shape->ToString();
  }
  if (!type->isa<List>()) {
    MS_INTERNAL_EXCEPTION(TypeError) << "For GenerateAbstractList, the type should be list but got: "
                                     << type->ToString();
  }
  auto shape_list = base_shape->cast_ptr<abstract::ListShape>();
  MS_EXCEPTION_IF_NULL(shape_list);
  auto type_list = type->cast_ptr<List>();
  MS_EXCEPTION_IF_NULL(type_list);
  if (shape_list->size() != type_list->size()) {
    MS_INTERNAL_EXCEPTION(ValueError) << "For GenerateAbstractList, the shape and type size should be the same, "
                                      << "but got shape size: " << shape_list->size()
                                      << " and type size: " << type_list->size();
  }
  AbstractBasePtrList ptr_list;
  for (size_t it = 0; it < shape_list->size(); ++it) {
    auto element_shape = (*shape_list)[it];
    auto element_type = (*type_list)[it];
    bool is_external = type->isa<External>();
    bool is_tensor_or_scalar = element_type->isa<Number>() || element_type->isa<TensorType>();
    if (!is_external && is_tensor_or_scalar) {
      (void)ptr_list.emplace_back(abstract::MakeAbstract(element_shape, element_type));
    } else {
      // is_dyn_shape will be deleted after list PyExecute is not run re-infer.
      if (is_dyn_shape) {
        (void)ptr_list.emplace_back(std::make_shared<abstract::AbstractAny>());
      } else {
        const auto &infer_shape = std::make_shared<abstract::Shape>(ShapeVector({1}));
        (void)ptr_list.emplace_back(abstract::MakeAbstract(infer_shape, kFloat64));
      }
    }
  }
  return std::make_shared<abstract::AbstractList>(ptr_list);
}
}  // namespace fallback
}  // namespace mindspore
