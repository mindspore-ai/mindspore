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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/gmm.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/gmm_common_utils.h"

namespace mindspore::prim {
void CheckGmmInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  CheckGroupTypeValue(input_args[kIndex4], "Gmm");
}

BeginFunction(Gmm, x, weight, bias, group_list, group_type, group_list_type) {
  auto split_item = Value(3);
  auto none = NewValueNode(kNone);
  auto out = Call(Prim(GroupedMatmulV2), x, weight, bias, none, none, none, none, group_list, split_item, group_type);
  Return(out);
}
EndFunction(Gmm)
}  // namespace mindspore::prim
