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

#include "ops/ops_func_impl/upsample_forward_base.h"
#include "ops/ops_func_impl/upsample.h"

namespace mindspore {
namespace ops {
BaseShapePtr UpsampleForwardBaseFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  const size_t image_rank = GetImageRank();
  return UpsampleForwardInferShape(primitive, input_args, image_rank);
}

TypePtr UpsampleForwardBaseFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args.at(0));
  auto input_type = input_args[0]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);
  return input_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
