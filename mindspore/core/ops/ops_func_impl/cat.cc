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
#include "ops/ops_func_impl/cat.h"
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"
#include "utils/ms_context.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr CatFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  AbstractBasePtrList elements = input_args;
  if (input_args[0]->isa<abstract::AbstractSequence>()) {
    elements = input_args[0]->cast<abstract::AbstractSequencePtr>()->elements();
  }
  auto element0 = elements[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(element0);
  auto element0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kShape];
  auto element0_rank = element0_shape.size();
  auto dim = GetValue<int64_t>(input_args[1]->BuildValue());
  CheckAndConvertUtils::CheckInRange<int64_t>("Cat axis", dim, kIncludeBoth,
                                              {-SizeToLong(element0_rank), SizeToLong(element0_rank) - 1}, prim_name);
  auto axis = dim < 0 ? LongToSize(dim + SizeToLong(element0_rank)) : LongToSize(dim);
  int64_t all_shp = element0_shape[axis];
  for (size_t i = 1; i < elements.size(); ++i) {
    std::string elementi = "element" + std::to_string(i);
    auto elementi_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape())[kShape];
    all_shp = all_shp == -1 || elementi_shape[axis] == -1 ? -1 : all_shp + elementi_shape[axis];
  }
  auto ret_shape = element0_shape;
  ret_shape[axis] = all_shp;
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr CatFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  AbstractBasePtrList elements = input_args;
  if (input_args[0]->isa<abstract::AbstractSequence>()) {
    elements = input_args[0]->cast<abstract::AbstractSequencePtr>()->elements();
  }
  return elements[0]->BuildType();
}
}  // namespace ops
}  // namespace mindspore
