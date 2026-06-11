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
  if (attr_size != 1 && attr_size != 2) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << primitive->name() << "], '" << attr_name
                             << "' must be an int or a tuple with length 2, but got length " << attr_size << ".";
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
  if (output_size_len != 0 && output_size_len != 2 && output_size_len != 3 && output_size_len != 4) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << primitive->name()
                             << "], 'output_size' must have length 0, 2, 3 or 4, but got length " << output_size_len
                             << ".";
  }
}
}  // namespace

static inline bool CheckInputIsDynamic(const InferInfoPtrList &input_infos) {
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

ShapeArray MaxUnpool2DExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  auto in_shape = input_infos[kInputIndex0]->GetShape();

  if (input_infos[kInputIndex0]->IsDynamicRank()) {
    return {in_shape};
  }
  const int64_t input_num_dims = SizeToLong(in_shape.size());
  auto last_dim = input_num_dims - 1;
  ShapeVector out_shape = input_infos[kInputIndex0]->IsDynamic() ? input_infos[kInputIndex1]->GetShape() : in_shape;
  CheckAndConvertUtils::CheckInRange("dim of input", input_num_dims, kIncludeBoth, {3, 4}, primitive->name());
  CheckMaxUnpool2DExtAttr(primitive, "kernel_size", input_infos[kInputIndex2], false);
  CheckMaxUnpool2DExtAttr(primitive, "stride", input_infos[kInputIndex3], false);
  CheckMaxUnpool2DExtAttr(primitive, "padding", input_infos[kInputIndex4], true);
  CheckMaxUnpool2DExtOutputSize(primitive, input_infos[kInputIndex5]);
  if (input_infos[kInputIndex5]->IsNone()) {
    if (CheckInputIsDynamic(input_infos)) {
      out_shape[last_dim - kDim1] = -1;
      out_shape[last_dim] = -1;
      return {out_shape};
    }
    return {CalOutShape(input_infos[kInputIndex2], input_infos[kInputIndex3], input_infos[kInputIndex4], out_shape,
                        last_dim)[0]};
  }
  auto output_shape_opt = input_infos[kInputIndex5]->GetArrayValue<int64_t>();
  if (!output_shape_opt.has_value()) {
    out_shape[last_dim - kDim1] = -1;
    out_shape[last_dim] = -1;
    return {out_shape};
  }
  size_t size = output_shape_opt.value().size();
  if (size == 0) {
    if (CheckInputIsDynamic(input_infos)) {
      out_shape[last_dim - kDim1] = -1;
      out_shape[last_dim] = -1;
      return {out_shape};
    }
    return {CalOutShape(input_infos[kInputIndex2], input_infos[kInputIndex3], input_infos[kInputIndex4], out_shape,
                        last_dim)[0]};
  }

  if (CheckInputIsDynamic(input_infos)) {
    if (output_shape_opt.has_value() && output_shape_opt.value().size() >= kDim2 &&
        !output_shape_opt.value().HasUnknownValue()) {
      auto output_shape_val = output_shape_opt.value();
      out_shape[last_dim - kDim1] = output_shape_val[size - kDim2];
      out_shape[last_dim] = output_shape_val[size - kDim1];
    } else {
      out_shape[last_dim - kDim1] = -1;
      out_shape[last_dim] = -1;
    }
    return {out_shape};
  }
  if (output_shape_opt.has_value() && output_shape_opt.value().size() >= kDim2 &&
      !output_shape_opt.value().HasUnknownValue()) {
    auto cal_out =
      CalOutShape(input_infos[kInputIndex2], input_infos[kInputIndex3], input_infos[kInputIndex4], out_shape, last_dim);
    out_shape = cal_out[0];
    auto strides_out = cal_out[1];
    auto min_size_h = out_shape[last_dim - kDim1] - strides_out[kInputIndex0];
    auto max_size_h = out_shape[last_dim - kDim1] + strides_out[kInputIndex0];
    auto min_size_w = out_shape[last_dim] - strides_out[kInputIndex1];
    auto max_size_w = out_shape[last_dim] + strides_out[kInputIndex1];
    auto attr_output_shape = output_shape_opt.value();
    if ((min_size_h < attr_output_shape[size - kDim2] && attr_output_shape[size - kDim2] < max_size_h) &&
        (min_size_w < attr_output_shape[size - kDim1] && attr_output_shape[size - kDim1] < max_size_w)) {
      out_shape[last_dim - kDim1] = attr_output_shape[size - kDim2];
      out_shape[last_dim] = attr_output_shape[size - kDim1];
    } else {
      MS_EXCEPTION(ValueError) << "For Primitive[" << primitive->name() << "], invalid output_size "
                               << attr_output_shape.ToString() << ", expected spatial dims between " << min_size_h
                               << " and " << max_size_h << ", " << min_size_w << " and " << max_size_w << ".";
    }
  } else {
    out_shape[last_dim - kDim1] = -1;
    out_shape[last_dim] = -1;
  }
  return {out_shape};
}

std::vector<TypeId> MaxUnpool2DExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  return {input_infos[0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
