/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/max_pool_with_mask.h"
#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <set>
#include "ops_utils/op_constants.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
inline int64_t MaskComputeSize(int64_t in_value, const ArrayValue<int64_t> &kernel_size,
                               const ArrayValue<int64_t> &strides, const ArrayValue<int64_t> &pads,
                               const ArrayValue<int64_t> &dilation, size_t index, bool ceil_mode) {
  int64_t out_value = 0;
  const int64_t factor = 2;
  if (in_value == abstract::Shape::kShapeDimAny) {
    out_value = abstract::Shape::kShapeDimAny;
  } else if (kernel_size.IsValueUnknown(index) || strides.IsValueUnknown(index) || pads.IsValueUnknown(index) ||
             dilation.IsValueUnknown(index)) {
    out_value = abstract::Shape::kShapeDimAny;
  } else {
    auto out_d =
      (static_cast<double>(in_value + factor * pads[index] - dilation[index] * (kernel_size[index] - 1) - 1) /
       static_cast<double>(strides[index])) +
      1;
    if (ceil_mode) {
      out_value = static_cast<int>(ceil(out_d));
      if ((out_value - 1) * strides[index] >= in_value + pads[index]) {
        --out_value;
      }
    } else {
      out_value = static_cast<int>(floor(out_d));
    }
    if (out_value <= 0) {
      MS_EXCEPTION(ValueError) << "The index[" << index + kIndex2 << "] of input is [" << out_value
                               << "], which is invalid shape of MaxPoolWithMask.";
    }
  }
  return out_value;
}

inline void MaskCheckPositiveVector(const string &arg_name, const ArrayValue<int64_t> &array, const string &prim_name,
                                    bool exclude_zeros) {
  for (size_t i = 0; i < array.size(); ++i) {
    if (exclude_zeros) {
      if (MS_UNLIKELY(array[i] <= 0)) {
        MS_LOG(EXCEPTION) << "For " << prim_name << ", '" << arg_name << "' must be positive, but it's "
                          << array.ToString() << ".";
      }
    } else {
      if (MS_UNLIKELY(array[i] < 0)) {
        MS_LOG(EXCEPTION) << "For " << prim_name << ", '" << arg_name << "' must be not negetive, but it's "
                          << array.ToString() << ".";
      }
    }
  }
}
}  // namespace
TypeIdList MaxPoolWithMaskFuncImpl::InferType(const PrimitivePtr &, const InferInfoPtrList &input_infos) const {
  auto input_type = input_infos[kIndex0]->GetType();
  return {input_type, kNumberTypeInt8};
}

ShapeArray MaxPoolWithMaskFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  const size_t kAttrH = 0;
  const size_t kAttrW = 1;
  const int64_t kInputShapeSizeMin = 3;
  const int64_t kInputShapeSizeMax = 4;
  const int64_t kAttrsSize = 2;

  const auto &x_info = input_infos[kIndex0];
  auto x_shape = x_info->GetShape();
  if (MS_UNLIKELY(x_info->IsDynamicRank())) {
    // The output rank depends on the input rank (3D and 4D inputs map to different output ranks), so keep the
    // output rank unknown instead of guessing 4D here.
    ShapeVector shape{abstract::Shape::kShapeRankAny};
    return {shape, shape};
  }

  (void)CheckAndConvertUtils::CheckInRange("input x rank", SizeToLong(x_shape.size()), kIncludeBoth,
                                           {kInputShapeSizeMin, kInputShapeSizeMax}, primitive->name());
  const bool has_batch_dim = x_shape.size() == LongToSize(kInputShapeSizeMax);
  auto batch = has_batch_dim ? x_shape[kIndex0] : 1;
  auto channel = has_batch_dim ? x_shape[kIndex1] : x_shape[kIndex0];

  const auto &kernel_size_info = input_infos[kIndex1];
  auto kernel_size_array_opt = kernel_size_info->GetArrayValue<int64_t>();

  std::optional<ArrayValue<int64_t>> strides_array_opt;
  if (input_infos[kIndex2]->IsNone()) {
    strides_array_opt = kernel_size_array_opt;
  } else {
    strides_array_opt = input_infos[kIndex2]->GetArrayValue<int64_t>();
  }

  auto pads_array_opt = input_infos[kIndex3]->GetArrayValue<int64_t>();
  auto dilation_array_opt = input_infos[kIndex4]->GetArrayValue<int64_t>();
  auto ceil_mode_scalar_opt = input_infos[kIndex5]->GetScalarValue<bool>();
  if (!kernel_size_array_opt.has_value() || !strides_array_opt.has_value() || !pads_array_opt.has_value() ||
      !dilation_array_opt.has_value() || !ceil_mode_scalar_opt.has_value()) {
    ShapeVector dyn_output =
      has_batch_dim ? ShapeVector{batch, channel, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny}
                    : ShapeVector{channel, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny};
    return {dyn_output, dyn_output};
  }

  const auto &kernel_size_array = kernel_size_array_opt.value();
  const auto &strides_array = strides_array_opt.value();
  const auto &pads_array = pads_array_opt.value();
  const auto &dilation_array = dilation_array_opt.value();
  auto ceil_mode_scalar = ceil_mode_scalar_opt.value();

  (void)CheckAndConvertUtils::CheckInteger("kernel_size rank", SizeToLong(kernel_size_array.size()), kEqual, kAttrsSize,
                                           primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("strides rank", SizeToLong(strides_array.size()), kEqual, kAttrsSize,
                                           primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("pads rank", SizeToLong(pads_array.size()), kEqual, kAttrsSize,
                                           primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("dilation rank", SizeToLong(dilation_array.size()), kEqual, kAttrsSize,
                                           primitive->name());

  auto H_in = x_shape[has_batch_dim ? kIndex2 : kIndex1];
  auto W_in = x_shape[has_batch_dim ? kIndex3 : kIndex2];
  auto H_out =
    MaskComputeSize(H_in, kernel_size_array, strides_array, pads_array, dilation_array, kAttrH, ceil_mode_scalar);
  auto W_out =
    MaskComputeSize(W_in, kernel_size_array, strides_array, pads_array, dilation_array, kAttrW, ceil_mode_scalar);

  ShapeVector output_shape =
    has_batch_dim ? ShapeVector{batch, channel, H_out, W_out} : ShapeVector{channel, H_out, W_out};
  ShapeVector argmax_shape =
    has_batch_dim ? ShapeVector{batch, channel, kernel_size_array[kAttrH] * kernel_size_array[kAttrW],
                                (static_cast<int>(ceil(static_cast<double>(H_out * W_out) / 16)) + 1) * 2 * 16}
                  : ShapeVector{channel, kernel_size_array[kAttrH] * kernel_size_array[kAttrW],
                                (static_cast<int>(ceil(static_cast<double>(H_out * W_out) / 16)) + 1) * 2 * 16};
  return {output_shape, argmax_shape};
}

int32_t MaxPoolWithMaskFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  int32_t check_status = OP_CHECK_SUCCESS;

  const size_t kAttrH = 0;
  const size_t kAttrW = 1;

  auto kernel_size_array_opt = input_infos[kIndex1]->GetArrayValue<int64_t>();
  std::optional<ArrayValue<int64_t>> strides_array_opt;
  if (input_infos[kIndex2]->IsNone()) {
    strides_array_opt = kernel_size_array_opt;
  } else {
    strides_array_opt = input_infos[kIndex2]->GetArrayValue<int64_t>();
  }
  auto pads_array_opt = input_infos[kIndex3]->GetArrayValue<int64_t>();
  auto dilation_array_opt = input_infos[kIndex4]->GetArrayValue<int64_t>();
  if (MS_UNLIKELY(!kernel_size_array_opt.has_value() || !strides_array_opt.has_value() || !pads_array_opt.has_value() ||
                  !dilation_array_opt.has_value())) {
    check_status = OP_CHECK_RETRY;
  } else {
    const auto &kernel_size_array = kernel_size_array_opt.value();
    const auto &strides_array = strides_array_opt.value();
    const auto &pads_array = pads_array_opt.value();
    const auto &dilation_array = dilation_array_opt.value();
    if (MS_UNLIKELY(kernel_size_array.HasUnknownValue() || strides_array.HasUnknownValue() ||
                    pads_array.HasUnknownValue() || dilation_array.HasUnknownValue())) {
      check_status = OP_CHECK_RETRY;
    } else {
      MaskCheckPositiveVector(kKernelSize, kernel_size_array, primitive->name(), true);
      MaskCheckPositiveVector(kStrides, strides_array, primitive->name(), true);
      MaskCheckPositiveVector(kPads, pads_array, primitive->name(), false);
      MaskCheckPositiveVector(kDilation, dilation_array, primitive->name(), true);

      double half_factor = 0.5;
      if ((pads_array[kAttrH] > static_cast<int64_t>(static_cast<double>(kernel_size_array[kAttrH]) * half_factor)) ||
          (pads_array[kAttrW] > static_cast<int64_t>(static_cast<double>(kernel_size_array[kAttrW]) * half_factor))) {
        MS_EXCEPTION(ValueError)
          << "It is required that the `pads` is no more than half of the `kernel_size`, but gets pads("
          << pads_array[kAttrH] << ", " << pads_array[kAttrW] << ") and kernel_size(" << kernel_size_array[kAttrH]
          << ", " << kernel_size_array[kAttrW] << ").";
      }

      auto context = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context);
      const auto &dilation_vector = dilation_array.ToVector();
      if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice &&
          std::any_of(dilation_vector.begin(), dilation_vector.end(),
                      [](const int64_t &value) { return value != 1; })) {
        MS_EXCEPTION(ValueError) << "While running in Ascend, the attribute of `dilation` of '" << primitive->name()
                                 << "' is required to be all one, but got (" << dilation_vector[kAttrH] << ", "
                                 << dilation_vector[kAttrW] << ").";
      }
    }
  }
  return check_status;
}
}  // namespace ops
}  // namespace mindspore
