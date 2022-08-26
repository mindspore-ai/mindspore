/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/bucketize.h"

#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BucketizeInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto out_shape = x_shape;
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr BucketizeInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", x_type, common_valid_types, prim_name);
  return std::make_shared<TensorType>(kInt32);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Bucketize, BaseOperator);

void Bucketize::Init(const std::vector<float> &boundaries) { this->set_boundaries(boundaries); }

void Bucketize::set_boundaries(const std::vector<float> &boundaries) {
  (void)this->AddAttr(kBoundaries, api::MakeValue(boundaries));
}

std::vector<float> Bucketize::get_boundaries() const {
  auto value_ptr = GetAttr(kBoundaries);
  return GetValue<std::vector<float>>(value_ptr);
}

AbstractBasePtr BucketizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = BucketizeInferType(primitive, input_args);
  auto infer_shape = BucketizeInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Bucketize, prim::kPrimBucketize, BucketizeInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
