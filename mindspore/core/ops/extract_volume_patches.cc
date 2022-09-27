/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "ir/dtype/number.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ExtractVolumePatchesInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int MAX_SHAPE = 2048;
  const int d = 2;
  const int w = 4;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1, primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  constexpr int64_t shape_size = 5;
  (void)CheckAndConvertUtils::CheckInteger("input shape", SizeToLong(x_shape.size()), kEqual, shape_size,
                                           primitive->name());
  auto x_v = x_shape[2] * x_shape[3] * x_shape[4];
  (void)CheckAndConvertUtils::CheckInteger("x_d * x_h * x_w", x_v, kLessEqual, MAX_SHAPE, primitive->name());
  std::vector<int64_t> kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));
  std::vector<int64_t> strides = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStrides));
  constexpr int64_t kernel_size_num = 5;
  (void)CheckAndConvertUtils::CheckInteger("kernel_size_length", SizeToLong(kernel_size.size()), kEqual,
                                           kernel_size_num, primitive->name());
  constexpr int64_t strides_num = 5;
  (void)CheckAndConvertUtils::CheckInteger("strides_length", SizeToLong(strides.size()), kEqual, strides_num,
                                           primitive->name());
  auto padding = GetValue<std::string>(primitive->GetAttr(kPadding));
  for (auto &item : strides) {
    CheckAndConvertUtils::Check("strides", item, kGreaterThan, 0, primitive->name());
  }
  for (auto &item : kernel_size) {
    CheckAndConvertUtils::Check("kernel_size", item, kGreaterThan, 0, primitive->name());
  }
  std::vector<int64_t> y_shape(5);
  int64_t padding_needed = 0;
  y_shape[0] = x_shape[0];
  y_shape[1] = x_shape[1] * kernel_size[2] * kernel_size[3] * kernel_size[4];
  if (padding == "VALID") {
    for (int i = d; i <= w; ++i) {
      (void)CheckAndConvertUtils::CheckInteger(
        "padding = VALID, input[" + std::to_string(i) + "] - kernel_size[" + std::to_string(i) + "]",
        x_shape[IntToSize(i)] - kernel_size[IntToSize(i)], kGreaterEqual, 0, primitive->name());
      y_shape[IntToSize(i)] = 1 + (x_shape[IntToSize(i)] - kernel_size[IntToSize(i)]) / strides[IntToSize(i)];
    }
  } else {
    for (int i = d; i <= w; ++i) {
      y_shape[IntToSize(i)] = (x_shape[IntToSize(i)] + strides[IntToSize(i)] - 1) / strides[IntToSize(i)];
      int64_t output_size = y_shape[IntToSize(i)];
      padding_needed = (output_size - 1) * strides[IntToSize(i)] + kernel_size[IntToSize(i)] - x_shape[IntToSize(i)];
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
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt8,   kInt16, kInt32,
                                         kInt64,   kUInt8,   kUInt16,  kUInt32, kUInt64};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
}
}  // namespace

void ExtractVolumePatches::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
                                const std::string &padding) {
  set_kernel_size(kernel_size);
  set_strides(strides);
  set_padding(padding);
}

void ExtractVolumePatches::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)AddAttr(kKernelSize, api::MakeValue(kernel_size));
}

void ExtractVolumePatches::set_strides(const std::vector<int64_t> &strides) {
  (void)AddAttr(kStrides, api::MakeValue(strides));
}

void ExtractVolumePatches::set_padding(const std::string &padding) { (void)AddAttr(kPadding, api::MakeValue(padding)); }

std::vector<int64_t> ExtractVolumePatches::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> ExtractVolumePatches::get_strides() const {
  auto value_ptr = GetAttr(kStrides);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::string ExtractVolumePatches::get_padding() const {
  auto value_ptr = GetAttr(kPadding);
  return GetValue<std::string>(value_ptr);
}

MIND_API_OPERATOR_IMPL(ExtractVolumePatches, BaseOperator);
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
