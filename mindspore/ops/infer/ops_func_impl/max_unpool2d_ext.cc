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

#include "infer/ops_func_impl/max_unpool2d_ext.h"
#include <set>
#include <memory>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

#include "infer/ops_func_impl/reduce_arithmetic.h"

namespace mindspore {
namespace ops {
namespace {
// Legal length of kernel_size / stride / padding tuples in MaxUnpool2d:
//   - scalar form: a single int, broadcast to (H, W)
//   - (H, W) form: two ints, one per spatial dim
constexpr size_t kSpatialAttrScalarLen = 1;
constexpr size_t kSpatialAttrHWLen = 2;

// Legal length of `output_size` argument in MaxUnpool2d:
//   0 -> not provided (None/empty), let kernel/stride/padding decide the shape
//   2 -> (H, W)
//   3 -> (C, H, W)
//   4 -> (N, C, H, W)
constexpr size_t kOutputSizeNone = 0;
constexpr size_t kOutputSizeHW = 2;
constexpr size_t kOutputSizeCHW = 3;
constexpr size_t kOutputSizeNCHW = 4;

inline bool CheckInputIsDynamic(const InferInfoPtrList &input_infos) {
  return input_infos[kInputIndex0]->IsDynamic() && input_infos[kInputIndex1]->IsDynamic();
}

ShapeArray CalOutShape(const InferInfoPtr &ksize_ptr, const InferInfoPtr &strides_ptr, const InferInfoPtr &pads_ptr,
                       ShapeVector out_shape, size_t last_dim) {
  auto ksize_opt = ksize_ptr->GetArrayValue<int64_t>();
  auto strides_opt = (strides_ptr->IsNone()) ? ksize_opt : strides_ptr->GetArrayValue<int64_t>();
  auto pads_opt = pads_ptr->GetArrayValue<int64_t>();
  ShapeVector strides;
  if (!(ksize_opt.has_value() && strides_opt.has_value() && pads_opt.has_value())) {
    out_shape[last_dim - kDim1] = -1;
    out_shape[last_dim] = -1;
  } else if (ksize_opt.value().HasUnknownValue() || strides_opt.value().HasUnknownValue() ||
             pads_opt.value().HasUnknownValue()) {
    out_shape[last_dim - kDim1] = -1;
    out_shape[last_dim] = -1;
  } else {
    auto ksize = ksize_opt.value().ToVector();
    strides = strides_opt.value().ToVector();
    auto pads = pads_opt.value().ToVector();

    auto AdjustVectorLength = [](std::vector<int64_t> &vec) {
      if (vec.size() == 1) {
        vec.push_back(vec[0]);
      }
    };

    AdjustVectorLength(ksize);
    AdjustVectorLength(strides);
    AdjustVectorLength(pads);

    auto size = out_shape.size();
    int64_t out_h = static_cast<int64_t>((out_shape[size - kDim2] - 1) * strides[kInputIndex0] -
                                         kDim2 * pads[kInputIndex0] + ksize[kInputIndex0]);
    int64_t out_w = static_cast<int64_t>((out_shape[size - kDim1] - 1) * strides[kInputIndex1] -
                                         kDim2 * pads[kInputIndex1] + ksize[kInputIndex1]);
    out_shape[last_dim - 1] = out_h;
    out_shape[last_dim] = out_w;
  }
  return {out_shape, strides};
}

void CheckMaxUnpool2DExtAttr(const PrimitivePtr &primitive, const std::string &attr_name, const InferInfoPtr &attr_info,
                             bool allow_zero) {
  if (attr_info->IsNone()) {
    return;
  }
  auto attr_opt = attr_info->GetArrayValue<int64_t>();
  if (!attr_opt.has_value()) {
    return;
  }
  const auto attr_value = attr_opt.value();
  const auto attr_size = attr_value.size();
  if (attr_size != kSpatialAttrScalarLen && attr_size != kSpatialAttrHWLen) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << primitive->name() << "], '" << attr_name
                             << "' must be an int or a tuple with length " << kSpatialAttrHWLen << ", but got length "
                             << attr_size << ".";
  }
  if (attr_value.HasUnknownValue()) {
    return;
  }
  for (const auto &value : attr_value.ToVector()) {
    if ((allow_zero && value < 0) || (!allow_zero && value <= 0)) {
      MS_EXCEPTION(ValueError) << "For Primitive[" << primitive->name() << "], '" << attr_name << "' must be "
                               << (allow_zero ? "non-negative" : "greater than 0") << ", but got " << value << ".";
    }
  }
}

void CheckMaxUnpool2DExtOutputSize(const PrimitivePtr &primitive, const InferInfoPtr &output_size_info) {
  if (output_size_info->IsNone()) {
    return;
  }
  auto output_size_opt = output_size_info->GetArrayValue<int64_t>();
  if (!output_size_opt.has_value()) {
    return;
  }
  const auto output_size = output_size_opt.value();
  const auto output_size_len = output_size.size();
  if (output_size_len != kOutputSizeNone && output_size_len != kOutputSizeHW && output_size_len != kOutputSizeCHW &&
      output_size_len != kOutputSizeNCHW) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << primitive->name()
                             << "], 'output_size' must have length 0, 2, 3 or 4, but got length " << output_size_len
                             << ".";
  }
}

inline void SetSpatialDimsUnknown(ShapeVector *out_shape, int64_t last_dim) {
  (*out_shape)[last_dim - kDim1] = -1;
  (*out_shape)[last_dim] = -1;
}

// Output spatial dims when `output_size` is absent (None or empty): derive from
// kernel_size / stride / padding via CalOutShape, collapsing to unknown when
// input is dynamic.
ShapeVector ComputeOutShapeFromAttrsOnly(const InferInfoPtrList &input_infos, ShapeVector out_shape, int64_t last_dim) {
  if (CheckInputIsDynamic(input_infos)) {
    SetSpatialDimsUnknown(&out_shape, last_dim);
    return out_shape;
  }
  return CalOutShape(input_infos[kInputIndex2], input_infos[kInputIndex3], input_infos[kInputIndex4],
                     std::move(out_shape), static_cast<size_t>(last_dim))[0];
}

// `output_size` is usable for direct assignment iff it carries enough dims and
// every dim is statically known.
inline bool HasUsableOutputSize(const ArrayValue<int64_t> &output_size) {
  return output_size.size() >= kDim2 && !output_size.HasUnknownValue();
}

// Output spatial dims when input is dynamic and `output_size` may or may not be
// usable. Falls back to unknown spatial dims when output_size is partially
// known.
ShapeVector ComputeOutShapeDynamicWithOutputSize(const ArrayValue<int64_t> &output_size, ShapeVector out_shape,
                                                 int64_t last_dim) {
  if (!HasUsableOutputSize(output_size)) {
    SetSpatialDimsUnknown(&out_shape, last_dim);
    return out_shape;
  }
  const auto size = output_size.size();
  out_shape[last_dim - kDim1] = output_size[size - kDim2];
  out_shape[last_dim] = output_size[size - kDim1];
  return out_shape;
}

// Output spatial dims when input is static and `output_size` is provided. The
// supplied output_size must fall within the (min, max) range implied by
// kernel_size / stride / padding, otherwise an explicit ValueError is raised.
ShapeVector ComputeOutShapeStaticWithOutputSize(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                                const ArrayValue<int64_t> &output_size, ShapeVector out_shape,
                                                int64_t last_dim) {
  if (!HasUsableOutputSize(output_size)) {
    SetSpatialDimsUnknown(&out_shape, last_dim);
    return out_shape;
  }
  auto cal_out = CalOutShape(input_infos[kInputIndex2], input_infos[kInputIndex3], input_infos[kInputIndex4],
                             std::move(out_shape), static_cast<size_t>(last_dim));
  out_shape = cal_out[0];
  const auto &strides_out = cal_out[1];
  const auto min_size_h = out_shape[last_dim - kDim1] - strides_out[kInputIndex0];
  const auto max_size_h = out_shape[last_dim - kDim1] + strides_out[kInputIndex0];
  const auto min_size_w = out_shape[last_dim] - strides_out[kInputIndex1];
  const auto max_size_w = out_shape[last_dim] + strides_out[kInputIndex1];
  const auto size = output_size.size();
  const bool h_in_range = min_size_h < output_size[size - kDim2] && output_size[size - kDim2] < max_size_h;
  const bool w_in_range = min_size_w < output_size[size - kDim1] && output_size[size - kDim1] < max_size_w;
  if (!h_in_range || !w_in_range) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << primitive->name() << "], invalid output_size "
                             << output_size.ToString() << ", expected spatial dims between " << min_size_h << " and "
                             << max_size_h << ", " << min_size_w << " and " << max_size_w << ".";
  }
  out_shape[last_dim - kDim1] = output_size[size - kDim2];
  out_shape[last_dim] = output_size[size - kDim1];
  return out_shape;
}
}  // namespace

ShapeArray MaxUnpool2DExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  auto in_shape = input_infos[kInputIndex0]->GetShape();
  if (input_infos[kInputIndex0]->IsDynamicRank()) {
    return {in_shape};
  }
  const int64_t input_num_dims = SizeToLong(in_shape.size());
  const auto last_dim = input_num_dims - 1;
  ShapeVector out_shape = input_infos[kInputIndex0]->IsDynamic() ? input_infos[kInputIndex1]->GetShape() : in_shape;
  CheckAndConvertUtils::CheckInRange("dim of input", input_num_dims, kIncludeBoth, {3, 4}, primitive->name());
  CheckMaxUnpool2DExtAttr(primitive, "kernel_size", input_infos[kInputIndex2], false);
  CheckMaxUnpool2DExtAttr(primitive, "stride", input_infos[kInputIndex3], false);
  CheckMaxUnpool2DExtAttr(primitive, "padding", input_infos[kInputIndex4], true);
  CheckMaxUnpool2DExtOutputSize(primitive, input_infos[kInputIndex5]);

  // Branch 1: `output_size` not supplied → derive spatial dims from attrs.
  if (input_infos[kInputIndex5]->IsNone()) {
    return {ComputeOutShapeFromAttrsOnly(input_infos, std::move(out_shape), last_dim)};
  }
  auto output_shape_opt = input_infos[kInputIndex5]->GetArrayValue<int64_t>();
  // Branch 2: `output_size` is variable / unknown at infer time.
  if (!output_shape_opt.has_value()) {
    SetSpatialDimsUnknown(&out_shape, last_dim);
    return {out_shape};
  }
  // Branch 3: `output_size` is an empty sequence — semantically same as None.
  if (output_shape_opt.value().size() == 0) {
    return {ComputeOutShapeFromAttrsOnly(input_infos, std::move(out_shape), last_dim)};
  }
  // Branch 4 / 5: dynamic vs static input with usable output_size.
  if (CheckInputIsDynamic(input_infos)) {
    return {ComputeOutShapeDynamicWithOutputSize(output_shape_opt.value(), std::move(out_shape), last_dim)};
  }
  return {ComputeOutShapeStaticWithOutputSize(primitive, input_infos, output_shape_opt.value(), std::move(out_shape),
                                              last_dim)};
}

std::vector<TypeId> MaxUnpool2DExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  return {input_infos[0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
