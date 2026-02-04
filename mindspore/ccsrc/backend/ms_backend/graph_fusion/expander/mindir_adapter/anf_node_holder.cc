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

#include "backend/ms_backend/graph_fusion/expander/mindir_adapter/anf_node_holder.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_callback.h"
#include "abstract/abstract_value.h"
#include "ops_utils/op_constants.h"

namespace mindspore::graphkernel::expander {
ValuePtr AnfNodeHolder::GetValue() {
  auto v = node_->cast<ValueNodePtr>();
  if (v == nullptr) {
    auto abstract = node_->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    return abstract->BuildValue();
  }
  return v->value();
}

// todo, supports multi-output nodes
BaseShapePtr AnfNodeHolderWithDeviceInfo::GetShapePtr() { return nullptr; }
ShapeVector AnfNodeHolderWithDeviceInfo::GetShape() { return Callback::Instance()->GetOutputShape(node_, 0); }
TypePtr AnfNodeHolderWithDeviceInfo::GetDtype() { return TypeIdToType(Callback::Instance()->GetOutputType(node_, 0)); }
std::string AnfNodeHolderWithDeviceInfo::GetFormat() { return Callback::Instance()->GetOutputFormat(node_, 0); }
BaseShapePtr AnfNodeHolderWithHostInfo::GetShapePtr() { return nullptr; }
ShapeVector AnfNodeHolderWithHostInfo::GetShape() { return Callback::Instance()->GetOutputInferShape(node_, 0); }
TypePtr AnfNodeHolderWithHostInfo::GetDtype() {
  return TypeIdToType(Callback::Instance()->GetOutputInferType(node_, 0));
}
std::string AnfNodeHolderWithHostInfo::GetFormat() { return kOpFormat_DEFAULT; }
}  // namespace mindspore::graphkernel::expander
