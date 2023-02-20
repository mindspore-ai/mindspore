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

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
constexpr size_t kSTFTIndex0 = 0;
constexpr size_t kSTFTIndex1 = 1;
namespace {
abstract::ShapePtr STFTInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kSTFTIndex0]->GetShapeTrack())[kShape];
  auto window_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kSTFTIndex1]->GetShapeTrack())[kShape];
  if (batch_rank == 0) {
    CheckAndConvertUtils::CheckInRange<int64_t>("x_rank", SizeToLong(x_shape.size()), kIncludeBoth,
                                                {kSTFT1DSignalInput, kSTFT2DSignalInput}, op_name);
    (void)CheckAndConvertUtils::CheckInteger("window_rank", SizeToLong(window_shape.size()), kEqual, kSTFT1DWindowDims,
                                             op_name);
  } else {
    CheckAndConvertUtils::CheckInRange<int64_t>("x_rank", SizeToLong(x_shape.size()), kIncludeBoth,
                                                {kSTFT1DSignalInput + batch_rank, kSTFT2DSignalInput + batch_rank},
                                                op_name);
    (void)CheckAndConvertUtils::CheckInteger("window_rank", SizeToLong(window_shape.size()), kEqual,
                                             kSTFT1DWindowDims + batch_rank, op_name);
  }

  int64_t len = x_shape.back();

  int64_t n_fft = GetValue<int64_t>(primitive->GetAttr(kNFft));
  CheckAndConvertUtils::CheckInRange<int64_t>("n_fft", n_fft, kIncludeRight, {0, len}, op_name);

  int64_t hop_length = GetValue<int64_t>(primitive->GetAttr(kHopLength));
  (void)CheckAndConvertUtils::CheckInteger("hop_length", hop_length, kGreaterThan, 0, op_name);

  int64_t win_length = GetValue<int64_t>(primitive->GetAttr(kWinLength));
  CheckAndConvertUtils::CheckInRange<int64_t>("win_length", win_length, kIncludeRight, {0, n_fft}, op_name);

  (void)CheckAndConvertUtils::CheckInteger("window_shape", window_shape.back(), kEqual, win_length, op_name);

  std::vector<int64_t> out_shape = {};
  for (size_t index = 0; index < LongToSize(batch_rank); index++) {
    (void)CheckAndConvertUtils::CheckInteger("batch_shape", x_shape[index], kEqual, window_shape[index], op_name);
    (void)out_shape.emplace_back(x_shape[index]);
  }
  if (x_shape.size() - LongToSize(batch_rank) == kSTFT2DInputDims) {
    (void)out_shape.emplace_back(x_shape[LongToSize(batch_rank)]);
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
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_dtype = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_dtype);
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, op_name);

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

void STFT::Init(int64_t n_fft, int64_t hop_length, int64_t win_length, bool normalized, bool onesided,
                bool return_complex) {
  this->set_n_fft(n_fft);
  this->set_hop_length(hop_length);
  this->set_win_length(win_length);
  this->set_normalized(normalized);
  this->set_onesided(onesided);
  this->set_return_complex(return_complex);
}

void STFT::set_n_fft(int64_t n_fft) { (void)this->AddAttr(kNFft, api::MakeValue(n_fft)); }

void STFT::set_hop_length(int64_t hop_length) { (void)this->AddAttr(kHopLength, api::MakeValue(hop_length)); }

void STFT::set_win_length(int64_t win_length) { (void)this->AddAttr(kWinLength, api::MakeValue(win_length)); }

void STFT::set_normalized(bool normalized) { (void)this->AddAttr(kNormalized, api::MakeValue(normalized)); }

void STFT::set_onesided(bool onesided) { (void)this->AddAttr(kOnesided, api::MakeValue(onesided)); }

void STFT::set_return_complex(bool return_complex) {
  (void)this->AddAttr(kReturnComplex, api::MakeValue(return_complex));
}

int64_t STFT::get_n_fft() const {
  auto value_ptr = this->GetAttr(kNFft);
  return GetValue<int64_t>(value_ptr);
}

int64_t STFT::get_hop_length() const {
  auto value_ptr = this->GetAttr(kHopLength);
  return GetValue<int64_t>(value_ptr);
}

int64_t STFT::get_win_length() const {
  auto value_ptr = this->GetAttr(kWinLength);
  return GetValue<int64_t>(value_ptr);
}

bool STFT::get_normalized() const {
  auto value_ptr = this->GetAttr(kNormalized);
  return GetValue<bool>(value_ptr);
}

bool STFT::get_onesided() const {
  auto value_ptr = this->GetAttr(kOnesided);
  return GetValue<bool>(value_ptr);
}

bool STFT::get_return_complex() const {
  auto value_ptr = this->GetAttr(kReturnComplex);
  return GetValue<bool>(value_ptr);
}

// AG means auto generated
class MIND_API AGSTFTInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return STFTInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return STFTInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return STFTInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(STFT, prim::kPrimSTFT, AGSTFTInfer, false);
}  // namespace ops
}  // namespace mindspore
