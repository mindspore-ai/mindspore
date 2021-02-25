/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/rank.h"

namespace mindspore {
namespace ops {
namespace {
TypePtr RankInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto Rank_prim = prim->cast<PrimRankPtr>();
  MS_EXCEPTION_IF_NULL(Rank_prim);
  auto op_name = Rank_prim->name();
  auto infer_dtype = input_args[0]->BuildType();
  CheckAndConvertUtils::CheckSubClass("x", infer_dtype, {TypeIdToType(kObjectTypeTensorType)}, op_name);
  return TypeIdToType(kMetaTypeNone);
}
}  // namespace
AbstractBasePtr RankInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  std::vector<int64_t> infer_shape;
  return std::make_shared<abstract::AbstractTensor>(RankInferType(primitive, input_args), infer_shape);
}
REGISTER_PRIMITIVE_C(kNameRank, Rank);
}  // namespace ops
}  // namespace mindspore
