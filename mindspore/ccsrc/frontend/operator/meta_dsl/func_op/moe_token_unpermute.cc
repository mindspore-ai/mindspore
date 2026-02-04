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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/moe_token_unpermute.h"
#include "ir/dtype/type.h"
#include "ir/dtype/tensor_type.h"
#include "utils/shape_utils.h"
#include "primitive/array_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "primitive/sequence_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"

namespace mindspore::prim {
void CheckMoeTokenUnpermuteInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  auto permuted_type_ptr = input_args[kIndex0]->GetType();
  auto permuted_tensor_type = permuted_type_ptr->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(permuted_tensor_type);
  auto permuted_ele = permuted_tensor_type->element();
  MS_EXCEPTION_IF_NULL(permuted_ele);
  auto permuted_element_type = permuted_ele->type_id();
  primitive->AddAttr("permuted_type_id", MakeValue(static_cast<int64_t>(permuted_element_type)));

  auto probs_type_ptr = input_args[kIndex2]->GetType();
  if (probs_type_ptr->isa<TensorType>()) {
    auto probs_tensor_type = probs_type_ptr->cast<TensorTypePtr>();
    auto ele = probs_tensor_type->element();
    auto probs_element_type = ele->type_id();
    primitive->AddAttr("probs_type_id", MakeValue(static_cast<int64_t>(probs_element_type)));
  } else {
    primitive->AddAttr("probs_type_id", MakeValue(static_cast<int64_t>(kTypeUnknown)));
  }
}

BeginFunction(MoeTokenUnpermute, permuted_tokens, sorted_indices, probs, padded_mode, restore_shape) {
  auto cond = IsNone(probs);
  auto cond_f = [&]() {
    const auto &primitive = prim();
    auto probs_type_id = primitive->GetAttr("probs_type_id");
    MS_EXCEPTION_IF_NULL(probs_type_id);
    auto permuted_type_id = primitive->GetAttr("permuted_type_id");
    auto unpermuted_token_casted = Call(Prim(Cast), permuted_tokens, Value(probs_type_id));
    auto out =
      Call(Prim(InnerMoeTokenUnpermute), unpermuted_token_casted, sorted_indices, probs, padded_mode, restore_shape);
    auto out_cast = Call(Prim(Cast), out, Value(permuted_type_id));
    Return(out_cast);
  };

  auto cond_t = [&]() {
    auto out = Call(Prim(InnerMoeTokenUnpermute), permuted_tokens, sorted_indices, probs, padded_mode, restore_shape);
    Return(out);
  };

  Return(If(cond, cond_t, cond_f));
}
EndFunction(MoeTokenUnpermute)
}  // namespace mindspore::prim
