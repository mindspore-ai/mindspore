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
#include <vector>
#include <memory>
#include <string>
#include <map>
#include "ops/extract_glimpse.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ExtractGlimpseInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto offsets_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  const int kDimensionOne = 1;
  const int kDimensionTwo = 2;
  const int kDimensionFour = 4;
  if (input_shape.size() != kDimensionFour) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the shape of parameter "
                      << "'x' must be 4, but got " << input_shape.size() << ".";
  }
  if (size_shape.size() != kDimensionOne) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the shape of parameter "
                      << "'size' must be 1, but got " << size_shape.size() << ".";
  }
  if (offsets_shape.size() != kDimensionTwo) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the shape of parameter "
                      << "'offsets' must be 2, but got " << offsets_shape.size() << ".";
  }
  if (offsets_shape[1] != kDimensionTwo) {
    MS_EXCEPTION(ValueError) << "The second dimension of offsets must be 2, "
                             << "but got " << offsets_shape[1] << ".";
  }
  if (offsets_shape[0] != input_shape[0]) {
    MS_EXCEPTION(ValueError) << "The first dimension of offsets must be consistent with "
                             << "the first dimension of x, "
                             << "but got " << offsets_shape[0] << ".";
  }
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);

  int64_t batch_cnt = input_shape[0];
  int64_t channels = input_shape.back();
  if (!input_args[1]->BuildValue()->isa<AnyValue>() && !input_args[1]->BuildValue()->isa<None>()) {
    auto size_value = input_args[1]->BuildValue();
    MS_EXCEPTION_IF_NULL(size_value);
    auto size_value_tensor = size_value->cast<tensor::TensorPtr>();
    if (size_value_tensor == nullptr) {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', the input size must be const Tensor.";
    }
    int32_t *size_data = static_cast<int32_t *>(size_value_tensor->data_c());
    int32_t g_height = size_data[0], g_width = size_data[1];
    if (g_height == 0 || g_width == 0) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the value of parameter "
                               << "'size' must be greater than zero, but got [0, 0].";
    }
    int64_t output_elements_num = batch_cnt * g_height * g_width * channels;
    if (output_elements_num > max_length) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', The number of elements of output must be less than max length: " << max_length
                               << ", but got " << output_elements_num
                               << "! The shape of  output should be reduced or max_length should be increased";
    }
    std::vector<int64_t> output_shape{batch_cnt, g_height, g_width, channels};
    return std::make_shared<abstract::Shape>(output_shape);
  } else {
    const int64_t image_size = max_length / (batch_cnt * channels);
    const int64_t dim_size = static_cast<int64_t>(std::pow(image_size, 0.5));
    ShapeVector output_shape{batch_cnt, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, channels};
    ShapeVector shape_min(output_shape);
    ShapeVector shape_max(output_shape);
    shape_min[1] = 0;
    shape_min[kDimensionTwo] = 0;
    shape_max[1] = dim_size;
    shape_max[kDimensionTwo] = dim_size;
    return std::make_shared<abstract::Shape>(output_shape, shape_min, shape_max);
  }
}
TypePtr ExtractGlimpseInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const int kMagicNumber = 2;
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  if (!input_args[0]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For " << primitive->name() << ", the input x only support tensor!";
  }
  if (!input_args[1]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For " << primitive->name() << ", the input size only support tensor!";
  }
  if (!input_args[kMagicNumber]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For " << primitive->name() << ", the input offsets only support tensor!";
  }
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), {kFloat32}, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("size", input_args[1]->BuildType(), {kInt32}, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("offsets", input_args[kMagicNumber]->BuildType(), {kFloat32},
                                                   primitive->name());
  auto res = input_args[0]->BuildType();
  return res;
}
}  // namespace
MIND_API_BASE_IMPL(ExtractGlimpse, PrimitiveC, BaseOperator);
AbstractBasePtr ExtractGlimpseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = ExtractGlimpseInferType(primitive, input_args);
  auto infer_shape = ExtractGlimpseInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ExtractGlimpse, prim::kPrimExtractGlimpse, ExtractGlimpseInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
