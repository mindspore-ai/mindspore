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

#include "infer/ops_func_impl/multinomial_ext.h"
#include "ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore::ops {
ShapeArray MultinomialExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  auto &x = input_infos[kInputIndex0];
  std::optional<int64_t> num_samples_opt = input_infos[kInputIndex1]->GetScalarValue<int64_t>();
  if (x->IsDynamicRank()) {
    return {{abstract::TensorShape::kShapeRankAny}};
  }
  int64_t num_samples = -1;
  if (num_samples_opt.has_value()) {
    num_samples = num_samples_opt.value();
  }
  auto x_shape = x->GetShape();
  const int64_t x_rank_max = 2;
  const int64_t x_rank_min = 1;
  if (x_shape.size() > x_rank_max || x_shape.size() < x_rank_min) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', input[x] dimension must be 1 or 2, but got rank "
                             << x_shape.size() << ".";
  }

  std::vector<int64_t> output_shape;
  if (x_shape.size() == x_rank_max) {
    output_shape.push_back(x_shape[0]);
  }
  output_shape.push_back(num_samples);
  return {output_shape};
}

std::vector<TypeId> MultinomialExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  return {kNumberTypeInt64};
}
REGISTER_SIMPLE_INFER(kNameMultinomialExt, MultinomialExtFuncImpl);
}  // namespace mindspore::ops
