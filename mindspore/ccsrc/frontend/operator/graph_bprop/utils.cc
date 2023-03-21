/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/operator/graph_bprop/utils.h"
#include <memory>
#include "frontend/operator/ops_front_infer_function.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/parse/resolve.h"

namespace mindspore {
namespace graph_bprop {
ValuePtr GetAndCheckAttr(const PrimitivePtr &prim, const std::string &key) {
  auto value = prim->GetAttr(key);
  if (value == nullptr) {
    MS_LOG(EXCEPTION) << "The attr(" << key << ") of Primitive[" << prim->name() << "] not exist";
  }
  return value;
}

PrimitivePtr NewPrimitive(const PrimitivePtr &prim, const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  return NewPrimitive(prim->name(), attrs);
}

PrimitivePtr NewPrimitive(const std::string &prim_name, const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  auto prim = std::make_shared<Primitive>(prim_name, attrs);
  return prim;
}

FuncGraphPtr NewGraph(const AbstractBasePtrList &abs_list) {
  auto fg = std::make_shared<FuncGraph>();
  for (const auto &abs : abs_list) {
    auto para = fg->add_parameter();
    para->set_abstract(abs);
  }
  return fg;
}

void CheckArgSize(const std::vector<AnfNodePtr> &parameters, const AbstractBasePtrList &input_abs,
                  const PrimitivePtr &prim, size_t expected_size) {
  if (parameters.size() != expected_size) {
    MS_LOG(EXCEPTION) << "The parameter size of bprop graph of prim[" << prim->name() << "] should be " << expected_size
                      << ", but got " << parameters.size();
  }

  if (input_abs.size() != expected_size) {
    MS_LOG(EXCEPTION) << "The input abstract size of bprop graph of prim[" << prim->name() << "] should be "
                      << expected_size << ", but got " << input_abs.size();
  }
}

TypeId GetTensorDType(const AbstractBasePtr &abs) {
  auto tensor_abs = dyn_cast_ptr<abstract::AbstractTensor>(abs);
  if (tensor_abs == nullptr) {
    MS_LOG(EXCEPTION) << "The abstract should be AbstractTensor, but got " << abs->ToString();
  }
  MS_EXCEPTION_IF_NULL(tensor_abs->element());
  return tensor_abs->element()->BuildType()->type_id();
}

AnfNodePtr GetClassType(const std::string &package, const std::string &class_name) {
  auto module = python_adapter::GetPyModule(package);
  if (!module || py::isinstance<py::none>(module)) {
    MS_LOG(EXCEPTION) << "Can not get python module: " << package;
  }
  auto attr = module.attr(class_name.c_str());
  return NewValueNode(std::make_shared<parse::ClassType>(attr, package + "." + class_name));
}

bool ConvertToTensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  if (abs != nullptr) {
    return abs->isa<abstract::AbstractTensor>();
  }
  if (!node->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The node " << node->DebugString() << " should have been set abstract.";
  }
  auto value = node->cast_ptr<ValueNode>()->value();
  MS_EXCEPTION_IF_NULL(value);
  abs = value->ToAbstract();
  return abs->isa<abstract::AbstractTensor>();
}
}  // namespace graph_bprop
}  // namespace mindspore
