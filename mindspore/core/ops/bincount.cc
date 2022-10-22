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

#include <set>

#include "ops/bincount.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BincountInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto arr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShapeTrack())[kShape];
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShapeTrack())[kShape];
  int64_t arr_size = 1;
  int64_t weight_size = 1;
  if (w_shape.size() != 0) {
    auto weights_num = std::accumulate(w_shape.begin(), w_shape.end(), int64_t(1), std::multiplies{});
    (void)CheckAndConvertUtils::CheckInteger("size of weights", weights_num, kNotEqual, 0, primitive->name());
  }
  for (size_t i = 0; i < arr_shape.size(); ++i) {
    arr_size *= arr_shape[i];
  }
  for (size_t i = 0; i < w_shape.size(); ++i) {
    weight_size *= w_shape[i];
  }
  (void)CheckAndConvertUtils::CheckInteger("size of array and weights", arr_size, kEqual, weight_size,
                                           primitive->name());
  auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShapeTrack())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("size", size_shape.size(), kEqual, 0, primitive->name());
  auto size_value_ptr = input_args[kInputIndex1]->BuildValue();
  MS_EXCEPTION_IF_NULL(size_value_ptr);
  if (!size_value_ptr->isa<AnyValue>() && !size_value_ptr->isa<None>()) {
    if (!size_value_ptr->isa<tensor::Tensor>()) {
      MS_EXCEPTION(ValueError) << "For primitive[" << primitive->name() << "], the input argument[size]"
                               << " must be a tensor, but got " << size_value_ptr->ToString();
    }
    auto out_shape = CheckAndConvertUtils::CheckTensorIntValue("size", size_value_ptr, primitive->name());
    (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero("size", out_shape, primitive->name());
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    std::vector<int64_t> out_shape;
    (void)out_shape.emplace_back(-1);
    return std::make_shared<abstract::Shape>(out_shape);
  }
}
TypePtr BincountInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::set<TypePtr> valid_type = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("array", input_args[kInputIndex0]->BuildType(), valid_type,
                                                   primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("size", input_args[kInputIndex1]->BuildType(), valid_type,
                                                   primitive->name());
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kInt32, kInt64};
  auto weights_type = input_args[kInputIndex2]->BuildType();
  return CheckAndConvertUtils::CheckTensorTypeValid("weights", weights_type, valid_types, primitive->name());
}
}  // namespace

AbstractBasePtr BincountInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = BincountInferType(primitive, input_args);
  auto infer_shape = BincountInferShape(primitive, input_args);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape);
}

MIND_API_OPERATOR_IMPL(Bincount, BaseOperator);
REGISTER_HOST_DEPENDS(kNameBincount, {1});
REGISTER_PRIMITIVE_EVAL_IMPL(Bincount, prim::kPrimBincount, BincountInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
