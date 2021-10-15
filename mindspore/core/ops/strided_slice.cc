/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/strided_slice.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <bitset>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> TenToTwo(int64_t num) {
  std::vector<int64_t> output;
  if (num == 0) {
    output.push_back(0);
    return output;
  }
  while (num) {
    output.push_back(num % 2);
    num /= 2;
  }

  return output;
}

int64_t get_stride_with_not_zero(int64_t start_pos, int64_t end_pos, int64_t strides) {
  int64_t slicing_length = 0;
  if (strides != 0) {
    slicing_length = 1 + (end_pos + 1 - start_pos) / strides;
  } else {
    MS_EXCEPTION(ValueError) << "the strides must be non-zero but got " << strides;
  }
  return slicing_length;
}

void EllipsisInferShape(const PrimitivePtr &primitive, const std::vector<int64_t> &x_shape,
                        const std::vector<int64_t> &begin_v, const std::vector<int64_t> &end_v,
                        const std::vector<int64_t> &strides_v, std::vector<int64_t> *infer_shape, size_t i, size_t j,
                        bool has_ellipsis) {
  if (!has_ellipsis) {
    return;
  }
  MS_EXCEPTION_IF_NULL(primitive);
  auto strided_slice_prim = primitive->cast<PrimStridedSlicePtr>();
  MS_EXCEPTION_IF_NULL(strided_slice_prim);
  size_t x_rank = x_shape.size();
  size_t slice_len = begin_v.size();
  std::vector<int64_t> begin_pos = TenToTwo(GetValue<int64_t>(primitive->GetAttr(kBeginMask)));
  std::vector<int64_t> end_pos = TenToTwo(GetValue<int64_t>(primitive->GetAttr(kEndMask)));
  std::vector<int64_t> new_axis_pos = TenToTwo(GetValue<int64_t>(primitive->GetAttr(kNewAxisMask)));
  std::vector<int64_t> shrink_axis_pos = TenToTwo(GetValue<int64_t>(primitive->GetAttr(kShrinkAxisMask)));
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(new_axis_pos.size()), kGreaterEqual,
                                           SizeToLong(slice_len), primitive->name());

  size_t num = 0;
  for (size_t n = j + 1; n < slice_len; n++) {
    if (new_axis_pos[n] == 1) {
      num++;
    }
  }

  size_t ellipsis_occupied_dims = x_rank - i - (slice_len - (j + 1)) + num;
  (void)infer_shape->insert(infer_shape->end(), x_shape.begin() + SizeToInt(i),
                            x_shape.begin() + SizeToLong(i + ellipsis_occupied_dims));
  j += 1;
  i += ellipsis_occupied_dims;

  while (i < x_rank || j < slice_len) {
    int64_t x_dim_size = x_shape[i];
    int64_t start = begin_v[j];
    int64_t finish = end_v[j];
    int64_t strides = strides_v[j];
    if (j < begin_pos.size() || j < slice_len) {
      start = strides_v[j] < 0 ? -1 : 0;
    }
    if (j < end_pos.size() && end_pos[j] == 1) {
      finish = strides_v[j] < 0 ? -(x_shape[i] + 1) : x_shape[i];
    }
    if (j < new_axis_pos.size() && new_axis_pos[j] == 1) {
      infer_shape->push_back(1);
      j += 1;
      continue;
    }
    if (j < shrink_axis_pos.size() && shrink_axis_pos[j] == 1) {
      if ((-x_shape[i] <= start && start < x_shape[i]) || strides < 0) {
        MS_EXCEPTION(ValueError) << "when shrink axis, the stride cannot be negative number";
      }
      j += 1;
      i += 1;
      continue;
    }
    int64_t slicing_length = strided_slice_prim->compute_slicing_length(start, finish, strides, x_dim_size);
    infer_shape->push_back(slicing_length);
    i += 1;
    j += 1;
  }
  return;
}

const std::vector<int64_t> CheckAndGetValidStrides(const AbstractBasePtr &stride_arg) {
  MS_EXCEPTION_IF_NULL(stride_arg);
  auto temp_strides = stride_arg->cast<abstract::AbstractTuplePtr>()->BuildValue();
  MS_EXCEPTION_IF_NULL(temp_strides);
  auto strides = GetValue<std::vector<int64_t>>(temp_strides);
  if (std::any_of(strides.begin(), strides.end(), [](int64_t stride) { return stride == 0; })) {
    MS_EXCEPTION(ValueError) << "StridedSlice's input strides cannot contain 0.";
  }
  return strides;
}

std::vector<int64_t> ComputeInferShape(const PrimitivePtr &primitive, const std::vector<int64_t> &begin_v,
                                       const std::vector<int64_t> &end_v, const std::vector<int64_t> &x_shape,
                                       const std::vector<int64_t> &strides_v) {
  std::vector<int64_t> begin_pos = TenToTwo(GetValue<int64_t>(primitive->GetAttr(kBeginMask)));
  std::vector<int64_t> end_pos = TenToTwo(GetValue<int64_t>(primitive->GetAttr(kEndMask)));
  std::vector<int64_t> ellipsis_pos = TenToTwo(GetValue<int64_t>(primitive->GetAttr(kEllipsisMask)));
  std::vector<int64_t> new_axis_pos = TenToTwo(GetValue<int64_t>(primitive->GetAttr(kNewAxisMask)));
  std::vector<int64_t> shrink_axis_pos = TenToTwo(GetValue<int64_t>(primitive->GetAttr(kShrinkAxisMask)));
  size_t i = 0;
  size_t j = 0;
  int64_t start;
  int64_t finish;
  int64_t strides;
  int64_t slicing_length;
  bool has_ellipsis = false;
  std::vector<int64_t> infer_shape;
  size_t slice_len = begin_v.size();
  size_t x_rank = x_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("end_v size", SizeToLong(end_v.size()), kGreaterEqual, SizeToLong(slice_len),
                                           primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("strides_v size", SizeToLong(strides_v.size()), kGreaterEqual,
                                           SizeToLong(slice_len), primitive->name());
  while (i < x_rank || j < slice_len) {
    int64_t x_dim_size = x_shape[i];
    if (j < slice_len) {
      start = begin_v[j];
      finish = end_v[j];
      strides = strides_v[j];
      if (j < ellipsis_pos.size() && ellipsis_pos[j] == 1) {
        has_ellipsis = true;
        break;
      }
      if (j < begin_pos.size() && begin_pos[j] == 1) {
        start = strides_v[j] < 0 ? -1 : 0;
      }
      if (j < end_pos.size() && end_pos[j] == 1) {
        finish = strides_v[j] < 0 ? -(x_shape[i] + 1) : x_shape[i];
      }
      if (j < new_axis_pos.size() && new_axis_pos[j] == 1) {
        infer_shape.push_back(1);
        j += 1;
        continue;
      }
      if (j < shrink_axis_pos.size() && shrink_axis_pos[j] == 1) {
        if ((-x_shape[i] <= start && start < x_shape[i]) || strides < 0) {
          MS_EXCEPTION(ValueError) << "when shrink axis, the stride cannot be negative number";
        }
        j += 1;
        i += 1;
        continue;
      }
    } else {
      start = 0;
      finish = x_shape[0];
      strides = 1;
    }
    auto strided_slice_prim = primitive->cast<PrimStridedSlicePtr>();
    MS_EXCEPTION_IF_NULL(strided_slice_prim);
    slicing_length = strided_slice_prim->compute_slicing_length(start, finish, strides, x_dim_size);
    infer_shape.push_back(slicing_length);
    i += 1;
    j += 1;
  }
  EllipsisInferShape(primitive, x_shape, begin_v, end_v, strides_v, &infer_shape, i, j, has_ellipsis);
  return infer_shape;
}

abstract::ShapePtr StridedSliceInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto tuple_begin_v = input_args[kInputIndex1]->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_begin_v);
  auto temp_begin_v = tuple_begin_v->BuildValue();
  MS_EXCEPTION_IF_NULL(temp_begin_v);
  auto begin_v = GetValue<std::vector<int64_t>>(temp_begin_v);

  auto tuple_end_v = input_args[kInputIndex2]->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_end_v);
  auto temp_end_v = tuple_end_v->BuildValue();
  MS_EXCEPTION_IF_NULL(temp_end_v);
  auto end_v = GetValue<std::vector<int64_t>>(temp_end_v);
  auto strides_v = CheckAndGetValidStrides(input_args[kInputIndex3]);

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto min_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kMinShape];
  auto max_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kMaxShape];
  auto ret_in_shape = ComputeInferShape(primitive, begin_v, end_v, x_shape, strides_v);
  if (min_shape.empty() || max_shape.empty()) {
    return std::make_shared<abstract::Shape>(ret_in_shape);
  }
  auto ret_min_shape = ComputeInferShape(primitive, begin_v, end_v, min_shape, strides_v);
  auto ret_max_shape = ComputeInferShape(primitive, begin_v, end_v, max_shape, strides_v);
  return std::make_shared<abstract::Shape>(ret_in_shape, ret_min_shape, ret_max_shape);
}

TypePtr StridedSliceInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const int64_t x_index = 0;
  return CheckAndConvertUtils::GetInputTensorType(input_args, x_index, primitive->name());
}
}  // namespace

void StridedSlice::set_begin_mask(const int64_t begin_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kBeginMask, begin_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kBeginMask, MakeValue(begin_mask));
}
int64_t StridedSlice::get_begin_mask() const {
  auto value_ptr = GetAttr(kBeginMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::set_end_mask(const int64_t end_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kEndMask, end_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kEndMask, MakeValue(end_mask));
}
int64_t StridedSlice::get_end_mask() const {
  auto value_ptr = GetAttr(kEndMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::set_ellipsis_mask(const int64_t ellipsis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kEllipsisMask, ellipsis_mask, kGreaterEqual, 0, this->name());
  std::bitset<sizeof(int64_t) * 8> bs(ellipsis_mask);
  std::ostringstream buffer;
  if (bs.count() > 1) {
    buffer << "For" << this->name() << ", only support one ellipsis in the index, but got " << this->get_end_mask();
    MS_EXCEPTION(ValueError) << buffer.str();
  }
  (void)this->AddAttr(kEllipsisMask, MakeValue(ellipsis_mask));
}
int64_t StridedSlice::get_ellipsis_mask() const {
  auto value_ptr = GetAttr(kEllipsisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::set_new_axis_mask(const int64_t new_axis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kNewAxisMask, new_axis_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kNewAxisMask, MakeValue(new_axis_mask));
}
int64_t StridedSlice::get_new_axis_mask() const {
  auto value_ptr = GetAttr(kNewAxisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::set_shrink_axis_mask(const int64_t shrink_axis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kShrinkAxisMask, shrink_axis_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kShrinkAxisMask, MakeValue(shrink_axis_mask));
}
int64_t StridedSlice::get_shrink_axis_mask() const {
  auto value_ptr = GetAttr(kShrinkAxisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::Init(const int64_t begin_mask, const int64_t end_mask, const int64_t ellipsis_mask,
                        const int64_t new_axis_mask, const int64_t shrink_axis_mask) {
  this->set_begin_mask(begin_mask);
  this->set_end_mask(end_mask);
  this->set_ellipsis_mask(ellipsis_mask);
  this->set_new_axis_mask(new_axis_mask);
  this->set_shrink_axis_mask(shrink_axis_mask);
}

int64_t StridedSlice::compute_slicing_length(int64_t start_pos, int64_t end_pos, int64_t strides, int64_t x_dim) const {
  int64_t slicing_length = 0;
  if (strides > 0) {
    if ((start_pos >= x_dim) || end_pos < -x_dim) {
      slicing_length = 0;
    } else {
      if (-x_dim <= start_pos && start_pos < 0) {
        start_pos += x_dim;
      }
      if (start_pos < -x_dim) {
        start_pos = 0;
      }
      if (-x_dim <= end_pos && end_pos < 0) {
        end_pos += x_dim;
      }
      if (end_pos > x_dim) {
        end_pos = x_dim;
      }
      if (start_pos > end_pos) {
        slicing_length = 0;
      } else {
        slicing_length = 1 + (end_pos - 1 - start_pos) / strides;
      }
    }
  } else {
    if (start_pos < -x_dim || end_pos >= x_dim) {
      slicing_length = 0;
    } else {
      if (start_pos > 0 && start_pos < x_dim) {
        start_pos += -x_dim;
      }
      if (start_pos >= x_dim) {
        start_pos = -1;
      }
      if (end_pos >= 0 && end_pos < x_dim) {
        end_pos += -x_dim;
      }
      if (end_pos < -x_dim - 1) {
        end_pos = -x_dim - 1;
      }
      if (start_pos <= end_pos) {
        slicing_length = 0;
      } else {
        slicing_length = get_stride_with_not_zero(start_pos, end_pos, strides);
      }
    }
  }
  return slicing_length;
}

AbstractBasePtr StridedSliceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  return std::make_shared<abstract::AbstractTensor>(StridedSliceInferType(primitive, input_args),
                                                    StridedSliceInferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameStridedSlice, StridedSlice);
}  // namespace ops
}  // namespace mindspore
