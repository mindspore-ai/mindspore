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

#include "ops/stft.h"
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr STFTInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t k2DInputDims = 2;
  constexpr int64_t k1DWindowDims = 1;
  constexpr int64_t k1DSignalInput = 1;
  constexpr int64_t k2DSignalInput = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  CheckAndConvertUtils::CheckInRange<int64_t>("x_rank", SizeToLong(x_shape.size()), kIncludeBoth,
                                              {k1DSignalInput, k2DSignalInput}, op_name);
  int64_t len = x_shape.back();

  int64_t n_fft = GetValue<int64_t>(primitive->GetAttr(kNFft));
  CheckAndConvertUtils::CheckInRange<int64_t>("n_fft", n_fft, kIncludeRight, {0, len}, op_name);

  int64_t hop_length = GetValue<int64_t>(primitive->GetAttr(kHopLength));
  (void)CheckAndConvertUtils::CheckInteger("hop_length", hop_length, kGreaterThan, 0, op_name);

  int64_t win_length = GetValue<int64_t>(primitive->GetAttr(kWinLength));
  CheckAndConvertUtils::CheckInRange<int64_t>("win_length", win_length, kIncludeRight, {0, n_fft}, op_name);

  auto window_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("window_rank", SizeToLong(window_shape.size()), kEqual, k1DWindowDims,
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("window_shape", window_shape[0], kEqual, win_length, op_name);

  std::vector<int64_t> out_shape = {};
  if (x_shape.size() == k2DInputDims) {
    (void)out_shape.emplace_back(x_shape[0]);
  }
  int64_t n_frames = 1 + (len - n_fft) / hop_length;
  int64_t fft_length = n_fft;
  bool onesided = GetValue<bool>(primitive->GetAttr(kOnesided));
  if (onesided) {
    // Only real part because symmetric.
    constexpr int64_t k2FolderNum = 2;
    fft_length = n_fft / k2FolderNum + 1;
  }
  (void)out_shape.emplace_back(fft_length);
  (void)out_shape.emplace_back(n_frames);
  bool ret_complex = GetValue<bool>(primitive->GetAttr(kReturnComplex));
  if (!ret_complex) {
    // Split complex into real and image.
    constexpr int64_t k2DRealOutput = 2;
    (void)out_shape.emplace_back(k2DRealOutput);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr STFTInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t k1DSignalInput = 1;
  constexpr int64_t k2DSignalInput = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_dtype = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_dtype);
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, op_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  CheckAndConvertUtils::CheckInRange<int64_t>("x_rank", SizeToLong(x_shape.size()), kIncludeBoth,
                                              {k1DSignalInput, k2DSignalInput}, op_name);

  auto window_dtype = input_args[1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("window", window_dtype, valid_types, op_name);
  auto window_tensor = window_dtype->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(window_tensor);
  MS_EXCEPTION_IF_NULL(window_tensor->element());
  TypeId window_tensor_id = window_tensor->element()->type_id();
  bool onesided = GetValue<bool>(primitive->GetAttr(kOnesided));
  if (window_tensor_id == kNumberTypeComplex64 || window_tensor_id == kNumberTypeComplex128) {
    if (onesided) {
      MS_EXCEPTION(ValueError) << "For onesided should be false if window is complex,"
                               << "but got " << onesided;
    }
  }

  auto x_tensor = x_dtype->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(x_tensor->element());
  TypeId x_tensor_id = x_tensor->element()->type_id();
  if (x_tensor_id == kNumberTypeComplex64 || x_tensor_id == kNumberTypeComplex128) {
    if (onesided) {
      MS_EXCEPTION(ValueError) << "For onesided should be false if input is complex,"
                               << "but got " << onesided;
    }
  }

  bool ret_complex = GetValue<bool>(primitive->GetAttr(kReturnComplex));
  if (x_tensor_id == kNumberTypeFloat64 || x_tensor_id == kNumberTypeComplex128 ||
      window_tensor_id == kNumberTypeFloat64 || window_tensor_id == kNumberTypeComplex128) {
    TensorTypePtr complex128_tensor_type = std::make_shared<TensorType>(kComplex128);
    TensorTypePtr float64_tensor_type = std::make_shared<TensorType>(kFloat64);
    return ret_complex ? complex128_tensor_type : float64_tensor_type;
  }
  TensorTypePtr complex64_tensor_type = std::make_shared<TensorType>(kComplex64);
  TensorTypePtr float32_tensor_type = std::make_shared<TensorType>(kFloat32);
  return ret_complex ? complex64_tensor_type : float32_tensor_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(STFT, BaseOperator);
AbstractBasePtr STFTInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = STFTInferType(primitive, input_args);
  auto infer_shape = STFTInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(STFT, prim::kPrimSTFT, STFTInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
