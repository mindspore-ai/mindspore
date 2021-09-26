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

#include "ops/extract_volume_patches.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ExtractVolumePatchesInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int MAX_SHAPE = 2048;
  const int d = 2;
  const int w = 4;
  (void)CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 1, primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  (void)CheckAndConvertUtils::CheckInteger("input shape", x_shape.size(), kEqual, 5, primitive->name());
  auto x_v = x_shape[2] * x_shape[3] * x_shape[4];
  (void)CheckAndConvertUtils::CheckInteger("x_d * x_h * x_w", x_v, kLessEqual, MAX_SHAPE, primitive->name());
  std::vector<int64_t> kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));
  std::vector<int64_t> strides = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStrides));
  (void)CheckAndConvertUtils::CheckInteger("kernel_size_length", kernel_size.size(), kEqual, 5, primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("strides_length", strides.size(), kEqual, 5, primitive->name());
  auto padding = GetValue<std::string>(primitive->GetAttr(kPadding));
  for (auto &item : strides) {
    (void)CheckAndConvertUtils::Check("strides", item, kGreaterThan, "zero", 0, primitive->name());
  }
  for (auto &item : kernel_size) {
    (void)CheckAndConvertUtils::Check("kernel_size", item, kGreaterThan, "zero", 0, primitive->name());
  }
  std::vector<int64_t> y_shape(5);
  int64_t padding_needed = 0;
  y_shape[0] = x_shape[0];
  y_shape[1] = x_shape[1] * kernel_size[2] * kernel_size[3] * kernel_size[4];
  if (padding == "VALID") {
    for (int i = d; i <= w; ++i) {
      (void)CheckAndConvertUtils::CheckInteger(
        "padding = VALID, input[" + std::to_string(i) + "] - kernel_size[" + std::to_string(i) + "]",
        x_shape[i] - kernel_size[i], kGreaterEqual, 0, primitive->name());
      y_shape[i] = 1 + (x_shape[i] - kernel_size[i]) / strides[i];
    }
  } else {
    for (int i = d; i <= w; ++i) {
      y_shape[i] = (x_shape[i] + strides[i] - 1) / strides[i];
      int64_t output_size = y_shape[i];
      padding_needed = (output_size - 1) * strides[i] + kernel_size[i] - x_shape[i];
      (void)CheckAndConvertUtils::CheckInteger("padding = ((input[" + std::to_string(i) + "] + strides[" +
                                                 std::to_string(i) + "] - 1) / strides[" + std::to_string(i) +
                                                 "]) - 1) * strides[" + std::to_string(i) + "] + kernel_size[" +
                                                 std::to_string(i) + "] - input[" + std::to_string(i) + "]",
                                               padding_needed, kGreaterEqual, 0, primitive->name());
    }
  }
  if (y_shape[3] != 1 || y_shape[4] != 1) {
    (void)CheckAndConvertUtils::CheckInteger("input_w + pad_l + pad_r - kernel_w - stride_w",
                                             x_shape[4] + padding_needed - kernel_size[4] - strides[4], kGreaterEqual,
                                             0, primitive->name());
  }
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr ExtractVolumePatchesInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  (void)CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 1, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
}
}  // namespace
AbstractBasePtr ExtractVolumePatchesInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto type = ExtractVolumePatchesInferType(primitive, input_args);
  auto shape = ExtractVolumePatchesInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ExtractVolumePatches, prim::kPrimExtractVolumePatches, ExtractVolumePatchesInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
