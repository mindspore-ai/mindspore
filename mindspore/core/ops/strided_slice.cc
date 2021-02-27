/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
abstract::ShapePtr StridedSliceInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto strided_slice_prim = primitive->cast<PrimStridedSlicePtr>();
  MS_EXCEPTION_IF_NULL(strided_slice_prim);
  auto prim_name = strided_slice_prim->name();
  auto temp_begin_v = input_args[1]->cast<abstract::AbstractTuplePtr>()->BuildValue();
  auto begin_v = GetValue<std::vector<int64_t>>(temp_begin_v);
  auto temp_end_v = input_args[2]->cast<abstract::AbstractTuplePtr>()->BuildValue();
  auto end_v = GetValue<std::vector<int64_t>>(temp_end_v);
  auto temp_strides_v = input_args[3]->cast<abstract::AbstractTuplePtr>()->BuildValue();
  auto strides_v = GetValue<std::vector<int64_t>>(temp_strides_v);

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  int64_t x_rank = x_shape.size();
  int64_t slice_len = begin_v.size();
  std::vector<int64_t> begin_pos = strided_slice_prim->TenToTwo(strided_slice_prim->get_begin_mask());
  std::vector<int64_t> end_pos = strided_slice_prim->TenToTwo(strided_slice_prim->get_end_mask());
  std::vector<int64_t> ellipsis_pos = strided_slice_prim->TenToTwo(strided_slice_prim->get_ellipsis_mask());
  std::vector<int64_t> new_axis_pos = strided_slice_prim->TenToTwo(strided_slice_prim->get_new_axis_mask());
  std::vector<int64_t> shrink_axis_pos = strided_slice_prim->TenToTwo(strided_slice_prim->get_shrink_axis_mask());

  int64_t i = 0;
  int64_t j = 0;
  int64_t start;
  int64_t finish;
  int64_t strides;
  int64_t slicing_length;
  bool has_ellipsis = false;
  std::vector<int64_t> infer_shape;
  while (i < x_rank || j < slice_len) {
    if (j < slice_len) {
      start = begin_v[j];
      finish = end_v[j];
      strides = strides_v[j];
      if (j < (int64_t)ellipsis_pos.size() && ellipsis_pos[j] == 1) {
        has_ellipsis = true;
        break;
      }
      if (j < (int64_t)begin_pos.size() && begin_pos[j] == 1) {
        start = strides_v[j] < 0 ? -1 : 0;
      }
      if (j < (int64_t)end_pos.size() && end_pos[j] == 1) {
        finish = strides_v[j] < 0 ? -(x_shape[i] + 1) : x_shape[i];
      }
      if (j < (int64_t)new_axis_pos.size() && new_axis_pos[j] == 1) {
        infer_shape.push_back(1);
        j += 1;
        continue;
      }
      if (j < (int64_t)shrink_axis_pos.size() && shrink_axis_pos[j] == 1) {
        if (((-x_shape[i] <= start && start < x_shape[i]) == false) || strides < 0) {
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
    slicing_length = strided_slice_prim->compute_slicing_length(start, finish, strides, x_shape, i);
    infer_shape.push_back(slicing_length);
    i += 1;
    j += 1;
  }

  int64_t num = 0;
  for (int64_t n = j + 1; n < slice_len; n++) {
    if (new_axis_pos[n] == 1) {
      num++;
    }
  }
  if (has_ellipsis) {
    int64_t ellipsis_occupied_dims = x_rank - i - (slice_len - (j + 1)) + num;
    infer_shape.insert(infer_shape.end(), x_shape.begin() + i, x_shape.begin() + i + ellipsis_occupied_dims);
    j += 1;
    i += ellipsis_occupied_dims;

    while (i < x_rank || j < slice_len) {
      start = begin_v[j];
      finish = end_v[j];
      strides = strides_v[j];
      if (j < (int64_t)begin_pos.size() || j < slice_len) {
        start = strides_v[j] < 0 ? -1 : 0;
      }
      if (j < (int64_t)end_pos.size() && end_pos[j] == 1) {
        finish = strides_v[j] < 0 ? -(x_shape[i] + 1) : x_shape[i];
      }
      if (j < (int64_t)new_axis_pos.size() && new_axis_pos[j] == 1) {
        infer_shape.push_back(1);
        j += 1;
        continue;
      }
      if (j < (int64_t)shrink_axis_pos.size() && shrink_axis_pos[j] == 1) {
        if (((-x_shape[i] <= start && start < x_shape[i]) == false) || strides < 0) {
          MS_EXCEPTION(ValueError) << "when shrink axis, the stride cannot be negative number";
        }
        j += 1;
        i += 1;
        continue;
      }
      slicing_length = strided_slice_prim->compute_slicing_length(start, finish, strides, x_shape, i);
      infer_shape.push_back(slicing_length);
      i += 1;
      j += 1;
    }
  }
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr StridedSliceInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  return infer_type;
}
}  // namespace

void StridedSlice::set_begin_mask(const int64_t begin_mask) {
  CheckAndConvertUtils::CheckInteger(kBeginMask, begin_mask, kGreaterEqual, 0, this->name());
  this->AddAttr(kBeginMask, MakeValue(begin_mask));
}
int64_t StridedSlice::get_begin_mask() const {
  auto value_ptr = GetAttr(kBeginMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::set_end_mask(const int64_t end_mask) {
  CheckAndConvertUtils::CheckInteger(kEndMask, end_mask, kGreaterEqual, 0, this->name());
  this->AddAttr(kEndMask, MakeValue(end_mask));
}
int64_t StridedSlice::get_end_mask() const {
  auto value_ptr = GetAttr(kEndMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::set_ellipsis_mask(const int64_t ellipsis_mask) {
  CheckAndConvertUtils::CheckInteger(kEllipsisMask, ellipsis_mask, kGreaterEqual, 0, this->name());
  std::bitset<sizeof(int64_t) * 8> bs(ellipsis_mask);
  std::ostringstream buffer;
  if (bs.count() > 1) {
    buffer << "For" << this->name() << ", only support one ellipsis in the index, but got " << this->get_end_mask();
    MS_EXCEPTION(ValueError) << buffer.str();
  }
  this->AddAttr(kEllipsisMask, MakeValue(ellipsis_mask));
}
int64_t StridedSlice::get_ellipsis_mask() const {
  auto value_ptr = GetAttr(kEllipsisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::set_new_axis_mask(const int64_t new_axis_mask) {
  CheckAndConvertUtils::CheckInteger(kNewAxisMask, new_axis_mask, kGreaterEqual, 0, this->name());
  this->AddAttr(kNewAxisMask, MakeValue(new_axis_mask));
}
int64_t StridedSlice::get_new_axis_mask() const {
  auto value_ptr = GetAttr(kNewAxisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSlice::set_shrink_axis_mask(const int64_t shrink_axis_mask) {
  CheckAndConvertUtils::CheckInteger(kShrinkAxisMask, shrink_axis_mask, kGreaterEqual, 0, this->name());
  this->AddAttr(kShrinkAxisMask, MakeValue(shrink_axis_mask));
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

std::vector<int64_t> StridedSlice::TenToTwo(int64_t num) {
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

int64_t StridedSlice::compute_slicing_length(int64_t start_pos, int64_t end_pos, int64_t strides,
                                             std::vector<int64_t> x_shape, int64_t i) {
  if (i > (int64_t)x_shape.size()) {
    MS_EXCEPTION(ValueError) << "For 'StridedSlice', When their is no new axis, "
                                "the index length must be less or equal than the dim of x.";
  }
  int64_t x_dim = x_shape[i];
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
      if (0 < start_pos && start_pos < x_dim) {
        start_pos += -x_dim;
      }
      if (start_pos >= x_dim) {
        start_pos = -1;
      }
      if (0 <= end_pos && end_pos < x_dim) {
        end_pos += -x_dim;
      }
      if (end_pos < -x_dim - 1) {
        end_pos = -x_dim - 1;
      }
      if (start_pos <= end_pos) {
        slicing_length = 0;
      } else {
        slicing_length = 1 + (end_pos + 1 - start_pos) / strides;
      }
    }
  }
  return slicing_length;
}

AbstractBasePtr StridedSliceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(StridedSliceInferType(primitive, input_args),
                                                    StridedSliceInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameStridedSlice, StridedSlice);
}  // namespace ops
}  // namespace mindspore
