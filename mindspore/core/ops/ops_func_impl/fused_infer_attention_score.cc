/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/fused_infer_attention_score.h"

#include <set>
#include <string>
#include <map>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "ops/op_enum.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr FusedInferAttentionScoreFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto attention_out_shape = input_args[kIndex0]->GetShape()->Clone();
  auto softmax_lse_shape = attention_out_shape;
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList({attention_out_shape, softmax_lse_shape}));
}

TypePtr FusedInferAttentionScoreFuncImpl::InferType(const PrimitivePtr &prim,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto attention_out_type = input_args[kIndex0]->BuildType();
  auto softmax_lse_shape = attention_out_type;
  return std::make_shared<Tuple>(std::vector<TypePtr>{attention_out_type, softmax_lse_shape});
}
}  // namespace ops
}  // namespace mindspore
