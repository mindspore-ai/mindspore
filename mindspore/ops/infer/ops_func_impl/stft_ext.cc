/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/stft_ext.h"

#include <utility>
#include <vector>

#include "abstract/dshape.h"
#include "mindapi/base/type_id.h"
#include "mindspore/ops/op_def/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kSTFTExtInputIdx = 0;
constexpr size_t kSTFTExtWindowIdx = 1;
constexpr size_t kSTFTExtNFftIdx = 2;
constexpr size_t kSTFTExtHopLengthIdx = 3;
constexpr size_t kSTFTExtWinLengthIdx = 4;
constexpr size_t kSTFTExtNormalizedIdx = 5;
constexpr size_t kSTFTExtOnesidedIdx = 6;
constexpr size_t kSTFTExtReturnComplexIdx = 7;
constexpr int64_t kSTFTExtTwo = 2;

bool IsRealOrComplexFloat(TypeId type_id) {
  return type_id == kNumberTypeFloat32 || type_id == kNumberTypeFloat64 || type_id == kNumberTypeComplex64 ||
         type_id == kNumberTypeComplex128;
}

bool IsHighPrecision(TypeId type_id) {
  return type_id == kNumberTypeFloat64 || type_id == kNumberTypeComplex128;
}

bool IsComplex(TypeId type_id) { return type_id == kNumberTypeComplex64 || type_id == kNumberTypeComplex128; }
}  // namespace

ShapeArray STFTExtFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &input_info = input_infos[kSTFTExtInputIdx];
  auto &window_info = input_infos[kSTFTExtWindowIdx];

  if (input_info->IsDynamicRank()) {
    return {ShapeVector{abstract::Shape::kShapeRankAny}};
  }
  auto input_shape = input_info->GetShape();
  auto input_rank = SizeToLong(input_shape.size());
  MS_CHECK_VALUE(input_rank == 1 || input_rank == kSTFTExtTwo,
                 CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>("rank of input", input_rank, kIncludeBoth, {1, 2},
                                                                      primitive));

  auto n_fft_opt = input_infos[kSTFTExtNFftIdx]->GetScalarValue<int64_t>();
  auto hop_length_opt = input_infos[kSTFTExtHopLengthIdx]->GetScalarValue<int64_t>();
  auto win_length_opt = input_infos[kSTFTExtWinLengthIdx]->GetScalarValue<int64_t>();
  auto onesided_opt = input_infos[kSTFTExtOnesidedIdx]->GetScalarValue<bool>();
  auto return_complex_opt = input_infos[kSTFTExtReturnComplexIdx]->GetScalarValue<bool>();

  if (!n_fft_opt.has_value() || !hop_length_opt.has_value() || !onesided_opt.has_value() ||
      !return_complex_opt.has_value()) {
    return {ShapeVector{abstract::Shape::kShapeRankAny}};
  }

  int64_t n_fft = n_fft_opt.value();
  int64_t hop_length = hop_length_opt.value();
  bool onesided = onesided_opt.value();
  bool return_complex = return_complex_opt.value();

  int64_t len = input_shape.back();
  if (len > 0) {
    MS_CHECK_VALUE(n_fft > 0 && n_fft <= len,
                   CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>("n_fft", n_fft, kIncludeRight, {0, len},
                                                                        primitive));
  }
  MS_CHECK_VALUE(hop_length > 0,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("hop_length", hop_length, kGreaterThan, 0, primitive));

  if (win_length_opt.has_value()) {
    int64_t win_length = win_length_opt.value();
    MS_CHECK_VALUE(win_length > 0 && win_length <= n_fft,
                   CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>("win_length", win_length, kIncludeRight,
                                                                        {0, n_fft}, primitive));
    if (!window_info->IsDynamicRank()) {
      auto window_shape = window_info->GetShape();
      MS_CHECK_VALUE(SizeToLong(window_shape.size()) == 1,
                     CheckAndConvertUtils::FormatCheckIntegerMsg("rank of window", SizeToLong(window_shape.size()),
                                                                 kEqual, 1, primitive));
      if (window_shape[0] > 0) {
        MS_CHECK_VALUE(window_shape[0] == win_length,
                       CheckAndConvertUtils::FormatCheckIntegerMsg("window size", window_shape[0], kEqual, win_length,
                                                                   primitive));
      }
    }
  }

  bool is_batched = (input_rank == kSTFTExtTwo);
  int64_t batch = is_batched ? input_shape[0] : 0;
  int64_t fft_length = onesided ? (n_fft / kSTFTExtTwo + 1) : n_fft;
  int64_t n_frames = (len < 0) ? abstract::Shape::kShapeDimAny : (1 + (len - n_fft) / hop_length);

  ShapeVector out_shape;
  if (is_batched) {
    out_shape.emplace_back(batch);
  }
  out_shape.emplace_back(fft_length);
  out_shape.emplace_back(n_frames);
  if (!return_complex) {
    out_shape.emplace_back(kSTFTExtTwo);
  }
  return {out_shape};
}

std::vector<TypeId> STFTExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  const auto &op_name = primitive->name();
  auto input_type = input_infos[kSTFTExtInputIdx]->GetType();
  if (!IsRealOrComplexFloat(input_type)) {
    MS_EXCEPTION(TypeError) << "For " << op_name
                            << ", the input dtype must be one of float32/float64/complex64/complex128, but got "
                            << TypeIdToString(input_type);
  }

  TypeId window_type = input_infos[kSTFTExtWindowIdx]->GetType();
  if (!IsRealOrComplexFloat(window_type)) {
    MS_EXCEPTION(TypeError) << "For " << op_name
                            << ", the window dtype must be one of float32/float64/complex64/complex128, but got "
                            << TypeIdToString(window_type);
  }

  auto onesided_opt = input_infos[kSTFTExtOnesidedIdx]->GetScalarValue<bool>();
  if (onesided_opt.has_value() && onesided_opt.value()) {
    if (IsComplex(input_type)) {
      MS_EXCEPTION(ValueError) << "For " << op_name << ", onesided must be False if input is complex.";
    }
    if (IsComplex(window_type)) {
      MS_EXCEPTION(ValueError) << "For " << op_name << ", onesided must be False if window is complex.";
    }
  }

  bool high_precision = IsHighPrecision(input_type) || IsHighPrecision(window_type);
  auto return_complex_opt = input_infos[kSTFTExtReturnComplexIdx]->GetScalarValue<bool>();
  bool return_complex = return_complex_opt.has_value() ? return_complex_opt.value()
                                                       : (IsComplex(input_type) || IsComplex(window_type));

  TypeId out_type;
  if (high_precision) {
    out_type = return_complex ? kNumberTypeComplex128 : kNumberTypeFloat64;
  } else {
    out_type = return_complex ? kNumberTypeComplex64 : kNumberTypeFloat32;
  }
  return {out_type};
}
}  // namespace ops
}  // namespace mindspore
