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

#include <vector>
#include <memory>
#include <string>
#include <algorithm>

#include "ops/extract_glimpse.h"
#include "utils/check_convert_utils.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/number.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
inline bool CheckShapePositiveTool(const std::vector<int64_t> &input_shape) {
  if (input_shape.size() != 0) {
    if (std::all_of(input_shape.begin(), input_shape.end(), [](int64_t i) { return i > 0; })) {
      return true;
    }
  }
  return false;
}

abstract::ShapePtr ExtractGlimpseInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto offsets_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  if (IsDynamicRank(size_shape)) {
    return std::make_shared<abstract::Shape>(size_shape);
  }
  const int kDimensionOne = 1;
  const int kDimensionTwo = 2;
  const int kDimensionFour = 4;
  if (!IsDynamicRank(input_shape) && (input_shape.size() != kDimensionFour)) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the shape of parameter "
                      << "'x' must be 4, but got " << input_shape.size() << ".";
  }
  if (size_shape.size() != kDimensionOne) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the shape of parameter "
                      << "'size' must be 1, but got " << size_shape.size() << ".";
  }
  if (!IsDynamicRank(offsets_shape) && (offsets_shape.size() != kDimensionTwo)) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the shape of parameter "
                      << "'offsets' must be 2, but got " << offsets_shape.size() << ".";
  }
  if (CheckShapePositiveTool(input_shape) && CheckShapePositiveTool(offsets_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("shape offsets", offsets_shape[1], kGreaterEqual, kDimensionTwo,
                                             prim_name);
    if (offsets_shape[1] != kDimensionTwo) {
      MS_EXCEPTION(ValueError) << "The second dimension of offsets must be 2, "
                               << "but got " << offsets_shape[1] << ".";
    }
    if (offsets_shape[0] != input_shape[0]) {
      MS_EXCEPTION(ValueError) << "The first dimension of offsets must be consistent with "
                               << "the first dimension of x, "
                               << "but got " << offsets_shape[0] << ".";
    }
  }
  int32_t g_height = -1;
  int32_t g_width = -1;
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
    g_height = size_data[0];
    g_width = size_data[1];
    if (g_height == 0 || g_width == 0) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the value of parameter "
                               << "'size' must be greater than zero, but got [0, 0].";
    }
  }
  std::vector<int64_t> output_shape{batch_cnt, g_height, g_width, channels};
  return std::make_shared<abstract::Shape>(output_shape);
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

void ExtractGlimpse::Init(const bool centered, const bool normalized, const bool uniform_noise, const string noise) {
  set_centered(centered);
  set_normalized(normalized);
  set_uniform_noise(uniform_noise);
}
void ExtractGlimpse::set_centered(const bool centered) { (void)this->AddAttr("centered", api::MakeValue(centered)); }
bool ExtractGlimpse::get_centered() const { return GetValue<bool>(GetAttr("centered")); }
void ExtractGlimpse::set_normalized(const bool normalized) {
  (void)this->AddAttr(kNormalized, api::MakeValue(normalized));
}
bool ExtractGlimpse::get_normalized() const { return GetValue<bool>(GetAttr(kNormalized)); }
void ExtractGlimpse::set_uniform_noise(const bool uniform_noise) {
  (void)this->AddAttr("uniform_noise", api::MakeValue(uniform_noise));
}
bool ExtractGlimpse::get_uniform_noise() const { return GetValue<bool>(GetAttr("uniform_noise")); }
std::string ExtractGlimpse::get_noise() const {
  auto value_ptr = GetAttr("noise");
  return GetValue<std::string>(value_ptr);
}

MIND_API_OPERATOR_IMPL(ExtractGlimpse, BaseOperator);
AbstractBasePtr ExtractGlimpseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = ExtractGlimpseInferType(primitive, input_args);
  auto infer_shape = ExtractGlimpseInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGExtractGlimpseInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ExtractGlimpseInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ExtractGlimpseInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ExtractGlimpseInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ExtractGlimpse, prim::kPrimExtractGlimpse, AGExtractGlimpseInfer, false);
}  // namespace ops
}  // namespace mindspore
