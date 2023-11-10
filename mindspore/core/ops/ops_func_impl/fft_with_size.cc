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

#include "ops/ops_func_impl/fft_with_size.h"
#include <set>
#include <memory>
#include <unordered_map>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/auto_generate/gen_enum_def.h"

namespace mindspore {
namespace ops {
const int64_t kDimNum = 2;
void CheckSignalNDim(const PrimitivePtr &primitive, const int64_t signal_ndim,
                     const std::vector<int64_t> &input_shape) {
  const int64_t kSignalRankMin = 1, kSignalRankMax = 3;
  MS_CHECK_VALUE(kSignalRankMin <= signal_ndim && signal_ndim <= kSignalRankMax,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("signal_ndim", signal_ndim, kIncludeBoth,
                                                             {kSignalRankMin, kSignalRankMax}, primitive));

  MS_CHECK_VALUE(SizeToLong(input_shape.size()) >= signal_ndim,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("x's dimension ", SizeToLong(input_shape.size()),
                                                             kGreaterEqual, signal_ndim, primitive));
  return;
}

void CheckSignalSizeValid(const PrimitivePtr &primitive, const int64_t signal_ndim,
                          const ArrayValue<int64_t> &signal_sizes_array, const std::vector<int64_t> &input_shape) {
  if (signal_sizes_array.size() == 0) {
    /* signal_sizes is empty */
    return;
  }
  if (IsDynamicShape(input_shape)) {
    /* input_shape has -1 */
    return;
  }

  std::vector<int64_t> signal_sizes;
  for (size_t i = 0; i < signal_sizes_array.size(); i++) {
    if (signal_sizes_array.IsValueUnknown(i)) {
      signal_sizes.push_back(abstract::TensorShape::kShapeDimAny);
    } else {
      signal_sizes.push_back(signal_sizes_array[i]);
    }
  }

  // irfft, signal_sizes without batch dimension.
  if (signal_sizes.size() != LongToUlong(signal_ndim)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', "
                             << "signal_sizes is expected to be empty (default)"
                             << " or of signal_ndim=" << signal_ndim << "D, but got signal_sizes=" << signal_sizes;
  }

  std::vector<int64_t> valid_size_even(input_shape.end() - signal_ndim, input_shape.end());
  valid_size_even.back() = (input_shape.back() - 1) * kDimNum;
  auto valid_size_odd = valid_size_even;
  valid_size_odd.back() = valid_size_even.back() + 1;
  auto batch_rank = SizeToLong(input_shape.size()) - signal_ndim;
  for (size_t i = 0; i < LongToUlong(signal_ndim) - 1; i++) {
    if (signal_sizes[i] != abstract::TensorShape::kShapeDimAny && signal_sizes[i] != input_shape[i + batch_rank]) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', "
                               << "got invalid signal_sizes: " << ToString(signal_sizes) << ", a valid one should be "
                               << ToString(valid_size_even) << ", or " << ToString(valid_size_odd) << ".";
    }
  }

  if (signal_sizes.back() != abstract::TensorShape::kShapeDimAny &&
      signal_sizes.back() / kDimNum + 1 != input_shape.back()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', "
                             << "got invalid signal_sizes: " << ToString(signal_sizes) << ", a valid one should be "
                             << ToString(valid_size_even) << ", or " << ToString(valid_size_odd) << ".";
  }
  return;
}

/*
 * fft/ifft mode: !real || !onesided
 *                shape: input
 *     rfft mode: real && onesided && !inverse
 *                shape: input.back = input_shape.back() / 2 + 1;
 *    irfft mode: real && onesided && inverse
 *                shape: input.back = signal_sizes.back()                   // when signal_sizes is valid
 *                       input.back = (input_shape.back() - 1) * kDimNum;   // when signal_sizes is empty
 */
BaseShapePtr FFTWithSizeFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();
  if (IsDynamicRank(input_shape)) {
    ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(dyn_output);
  }
  auto y_shape = input_shape;
  y_shape.back() = abstract::TensorShape::kShapeDimAny;

  /* get real and onesided value */
  auto real_v = GetScalarValue<bool>(input_args[kInputIndex3]->GetValue());
  auto onesided_v = GetScalarValue<bool>(input_args[kInputIndex5]->GetValue());
  if (!real_v.has_value() || !onesided_v.has_value()) {
    return std::make_shared<abstract::TensorShape>(y_shape);
  }
  bool real = real_v.value();
  bool onesided = onesided_v.value();
  if (!real || !onesided) {
    return input_shape_ptr->Clone();
  }

  /* get inverse value */
  auto inverse_v = GetScalarValue<bool>(input_args[kInputIndex2]->GetValue());
  if (!inverse_v.has_value()) {
    return std::make_shared<abstract::TensorShape>(y_shape);
  }
  bool inverse = inverse_v.value();
  if (!inverse) {
    y_shape.back() = input_shape.back() / kDimNum + 1;
    return std::make_shared<abstract::TensorShape>(y_shape);
  }

  /* get signal_sizes value */
  auto signal_sizes_v = GetArrayValue<int64_t>(input_args[kInputIndex6]->GetValue());
  if (!signal_sizes_v.has_value()) {
    return std::make_shared<abstract::TensorShape>(y_shape);
  }
  auto signal_sizes_array = signal_sizes_v.value();

  /* get signal_ndim value */
  auto signal_ndim_v = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  if (signal_ndim_v.has_value()) {
    int64_t signal_ndim = signal_ndim_v.value();
    CheckSignalNDim(primitive, signal_ndim, input_shape);
    CheckSignalSizeValid(primitive, signal_ndim, signal_sizes_array, input_shape);
  }

  /* calculate output shape */
  if (signal_sizes_array.size() == 0) {
    y_shape.back() = (input_shape.back() - 1) * kDimNum;
  } else if (signal_sizes_array.IsValueUnknown(signal_sizes_array.size() - 1)) {
    y_shape.back() = abstract::TensorShape::kShapeDimAny;
  } else {
    y_shape.back() = signal_sizes_array[signal_sizes_array.size() - 1];
  }
  return std::make_shared<abstract::TensorShape>(y_shape);
}

std::set<TypePtr> get_input_types(std::unordered_map<TypeId, TypePtr> types) {
  std::set<TypePtr> keys;
  for (const auto &t : types) {
    keys.insert(TypeIdToType(t.first));
  }
  return keys;
}
const std::unordered_map<TypeId, TypePtr> kRfftTypes{
  {kNumberTypeFloat32, kComplex64}, {kNumberTypeFloat64, kComplex128}, {kNumberTypeUInt8, kComplex64},
  {kNumberTypeInt8, kComplex64},    {kNumberTypeInt16, kComplex64},    {kNumberTypeInt32, kComplex64},
  {kNumberTypeInt64, kComplex64},   {kNumberTypeBool, kComplex64}};
const std::unordered_map<TypeId, TypePtr> kFftTypes{{kNumberTypeComplex64, kComplex64},
                                                    {kNumberTypeComplex128, kComplex128}};
const std::unordered_map<TypeId, TypePtr> kIrfftTypes{{kNumberTypeComplex64, kFloat32},
                                                      {kNumberTypeComplex128, kFloat64}};

/*
 * fft/ifft mode: !real
 *                dtype: complex -> complex
 *    irfft mode: real && inverse
 *                dtype: complex -> common
 *     rfft mode: real && !inverse
 *                dtype: common  -> complex
 */
TypePtr FFTWithSizeFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto input_type = input_args[kInputIndex0]->GetType()->cast<TensorTypePtr>()->element();
  auto inverse_v = GetScalarValue<bool>(input_args[kInputIndex2]->GetValue());
  auto real_v = GetScalarValue<bool>(input_args[kInputIndex3]->GetValue());

  if (!inverse_v.has_value() || !real_v.has_value()) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', got invalid inverse or real, dtype is unknown.";
  }
  auto real = real_v.value();
  auto inverse = inverse_v.value();

  TypePtr out_type(nullptr);
  if (real) {
    if (!inverse) {
      auto valid_types = get_input_types(kRfftTypes);
      (void)CheckAndConvertUtils::CheckTypeValidWithMoreInfo("x", input_type, "in rfft mode", valid_types, prim_name);
      out_type = kRfftTypes.at(input_type->type_id());
    } else {
      auto valid_types = get_input_types(kIrfftTypes);
      (void)CheckAndConvertUtils::CheckTypeValidWithMoreInfo("x", input_type, "in irfft mode", valid_types, prim_name);
      out_type = kIrfftTypes.at(input_type->type_id());
    }
  } else {
    auto valid_types = get_input_types(kFftTypes);
    (void)CheckAndConvertUtils::CheckTypeValidWithMoreInfo("x", input_type, "in fft/ifft mode", valid_types, prim_name);
    out_type = kFftTypes.at(input_type->type_id());
  }

  return std::make_shared<TensorType>(out_type);
}
}  // namespace ops
}  // namespace mindspore
