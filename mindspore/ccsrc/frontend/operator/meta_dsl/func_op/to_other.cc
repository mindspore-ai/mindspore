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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/to_other.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore::prim {
namespace {
TypeId GetNodeTypeId(const AnfNodePtr &node) {
  const auto &node_abs = node->abstract();
  MS_EXCEPTION_IF_NULL(node_abs);
  auto node_abs_type = node_abs->BuildType();
  MS_EXCEPTION_IF_NULL(node_abs_type);
  auto node_type = node_abs_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(node_type);
  return node_type->element()->type_id();
}
}  // namespace

BeginFunction(ToOther, x, other, non_blocking, copy) {
  auto x_type_id = GetNodeTypeId(x);
  auto other_type_id = GetNodeTypeId(other);

  if (x_type_id != other_type_id) {
    MS_LOG(DEBUG) << "Insert cast for primitive " << prim().get() << " " << prim()->ToString() << " x:" << x->ToString()
                  << " other:" << other->ToString();
    Return(Call(Prim(Cast), x, Value(MakeValue(static_cast<int64_t>(other_type_id)))));
  } else {
    MS_LOG(DEBUG) << "No need insert cast for primitive " << prim().get() << " " << prim()->ToString()
                  << " x:" << x->ToString() << " other:" << other->ToString();
    Return(x);
  }
}
EndFunction(ToOther)
}  // namespace mindspore::prim
