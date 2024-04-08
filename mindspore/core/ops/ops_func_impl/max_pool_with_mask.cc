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

#include "ops/ops_func_impl/max_pool_with_mask.h"
#include <algorithm>
#include <memory>
#include <string>
#include <set>
#include "include/common/utils/utils.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
TypePtr MaxPoolWithMaskFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto output_dtype = input_args[kIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_args[kIndex0]->GetType(), {kFloat16, kFloat32},
                                                   primitive->name());
  std::vector<TypePtr> type_list = {output_dtype, kInt8};
  return std::make_shared<Tuple>(type_list);
}

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

BaseShapePtr MaxPoolWithMaskFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  const size_t kAttrH = 0;
  const size_t kAttrW = 1;
  const int64_t kInputShapeSize = 4;
  const int64_t kAttrsSize = 2;
  auto x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  if (IsDynamicRank(x_shape)) {
    std::vector<abstract::BaseShapePtr> shape_list = {std::make_shared<abstract::Shape>(std::vector<int64_t>{
                                                        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                                                        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny}),
                                                      std::make_shared<abstract::Shape>(std::vector<int64_t>{
                                                        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                                                        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny})};
    return std::make_shared<abstract::TupleShape>(shape_list);
  }
  (void)CheckAndConvertUtils::CheckInteger("input x rank", SizeToLong(x_shape.size()), kEqual, kInputShapeSize,
                                           primitive->name());
  auto batch = x_shape[kIndex0];
  auto channel = x_shape[kIndex1];

  auto kernel_size = input_args[kIndex1]->GetValue();
  auto kernel_size_array_opt = GetArrayValue<int64_t>(kernel_size);
  ValuePtr strides;
  if (input_args[kIndex2]->GetType()->type_id() == kMetaTypeNone) {
    strides = kernel_size;
  } else {
    strides = input_args[kIndex2]->GetValue();
  }
  auto strides_array_opt = GetArrayValue<int64_t>(strides);
  auto pads = input_args[kIndex3]->GetValue();
  auto pads_array_opt = GetArrayValue<int64_t>(pads);
  auto dilation = input_args[kIndex4]->GetValue();
  auto dilation_array_opt = GetArrayValue<int64_t>(dilation);
  auto ceil_mode = input_args[kIndex5]->GetValue();
  auto ceil_mode_scalar_opt = GetScalarValue<bool>(ceil_mode);
  if (!kernel_size_array_opt.has_value() || !strides_array_opt.has_value() || !pads_array_opt.has_value() ||
      !dilation_array_opt.has_value() || !ceil_mode_scalar_opt.has_value()) {
    ShapeVector dyn_output{batch, channel, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny};
    std::vector<abstract::BaseShapePtr> shape_list = {std::make_shared<abstract::Shape>(dyn_output),
                                                      std::make_shared<abstract::Shape>(dyn_output)};
    return std::make_shared<abstract::TupleShape>(shape_list);
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
  auto H_in = x_shape[kIndex2];
  auto W_in = x_shape[kIndex3];
  auto H_out =
    MaskComputeSize(H_in, kernel_size_array, strides_array, pads_array, dilation_array, kAttrH, ceil_mode_scalar);
  auto W_out =
    MaskComputeSize(W_in, kernel_size_array, strides_array, pads_array, dilation_array, kAttrW, ceil_mode_scalar);
  ShapeVector output_shape = {x_shape[kIndex0], x_shape[kIndex1], H_out, W_out};
  ShapeVector argmax_shape = {x_shape[kIndex0], x_shape[kIndex1], kernel_size_array[kAttrH] * kernel_size_array[kAttrW],
                              (static_cast<int>(ceil(static_cast<double>(H_out * W_out) / 16)) + 1) * 2 * 16};

  std::vector<abstract::BaseShapePtr> shape_list = {std::make_shared<abstract::Shape>(output_shape),
                                                    std::make_shared<abstract::Shape>(argmax_shape)};
  return std::make_shared<abstract::TupleShape>(shape_list);
}
int32_t MaxPoolWithMaskFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  int32_t check_status = OP_CHECK_SUCCESS;

  const size_t kAttrH = 0;
  const size_t kAttrW = 1;
  auto kernel_size = input_args[kIndex1]->GetValue();
  auto kernel_size_array_opt = GetArrayValue<int64_t>(kernel_size);
  ValuePtr strides;
  if (input_args[kIndex2]->GetType()->type_id() == kMetaTypeNone) {
    strides = kernel_size;
  } else {
    strides = input_args[kIndex2]->GetValue();
  }
  auto strides_array_opt = GetArrayValue<int64_t>(strides);
  auto pads = input_args[kIndex3]->GetValue();
  auto pads_array_opt = GetArrayValue<int64_t>(pads);
  auto dilation = input_args[kIndex4]->GetValue();
  auto dilation_array_opt = GetArrayValue<int64_t>(dilation);

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
