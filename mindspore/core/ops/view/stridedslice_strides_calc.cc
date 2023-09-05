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
#include "ops/view/stridedslice_strides_calc.h"
#include <vector>
#include <memory>

namespace mindspore::ops {
constexpr size_t kStridedSliceInputsNum = 4;
void ConvertNegToPos(std::vector<int64_t> *begin, std::vector<int64_t> *end, const std::vector<int64_t> &tensor_shape) {
  if (begin->size() != tensor_shape.size()) {
    MS_EXCEPTION(ValueError) << "Convert shape size is not equal";
  }
  if (end->size() != tensor_shape.size()) {
    MS_EXCEPTION(ValueError) << "Convert shape size is not equal";
  }
  for (size_t i = 0; i < tensor_shape.size(); ++i) {
    if ((*begin)[i] < 0) {
      (*begin)[i] += tensor_shape[i];
    }
    if ((*end)[i] < 0) {
      (*end)[i] += tensor_shape[i];
    }
    if ((*begin)[i] < 0) {
      (*begin)[i] = 0;
    } else if ((*begin)[i] >= tensor_shape[i]) {
      (*begin)[i] = tensor_shape[i];
    }
    if ((*end)[i] < (*begin)[i]) {
      (*end)[i] = (*begin)[i];
    } else if ((*end)[i] >= tensor_shape[i]) {
      (*end)[i] = tensor_shape[i];
    }
  }
}
void VectorEmplace(std::vector<int64_t> *vec, size_t number, size_t dst_size) {
  if ((*vec).size() >= dst_size) {
    return;
  }
  for (size_t i = 0; i < dst_size - (*vec).size(); ++i) {
    (void)vec->emplace_back(number);
  }
}

bool CheckAttrIsNull(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto begin_mask = GetValue<int64_t>(primitive->GetAttr(kBeginMask));
  auto end_mask = GetValue<int64_t>(primitive->GetAttr(kEndMask));
  auto ellipsis_mask = GetValue<int64_t>(primitive->GetAttr(kEllipsisMask));
  auto new_axis_mask = GetValue<int64_t>(primitive->GetAttr(kNewAxisMask));
  auto shrink_axis_mask = GetValue<int64_t>(primitive->GetAttr(kShrinkAxisMask));
  if (begin_mask == 0 && end_mask == 0 && ellipsis_mask == 0 && new_axis_mask == 0 && shrink_axis_mask == 0) {
    return true;
  }
  return false;
}

TensorStorageInfoPtrList StridedSliceCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (!CheckAttrIsNull(prim) || inputs.size() != kStridedSliceInputsNum) {
    return {};
  }
  auto input_tensor = inputs[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto size = input_tensor->shape().size();
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto old_storage_offset = old_tensor_info->old_offset;
  if (inputs[1]->isa<tensor::Tensor>() || inputs[2]->isa<tensor::Tensor>() || inputs[2]->isa<tensor::Tensor>()) {
    return {};
  }
  auto begin = GetValue<std::vector<int64_t>>(inputs[1]);
  auto end = GetValue<std::vector<int64_t>>(inputs[2]);
  auto step = GetValue<std::vector<int64_t>>(inputs[3]);
  if (IsDynamic(step) || begin.size() != end.size() || begin.size() != step.size() || HasZero(step)) {
    return {};
  }
  VectorEmplace(&begin, 0, size);
  VectorEmplace(&end, size, size);
  VectorEmplace(&step, 1, size);
  ConvertNegToPos(&begin, &end, old_shape);

  for (size_t i = 0; i < begin.size(); ++i) {
    old_storage_offset += begin[i] * old_strides[i];
  }
  ShapeVector new_shape;
  auto new_strides = old_strides;
  for (size_t i = 0; i < size; ++i) {
    auto dim = DynamicDimWrap(i, input_tensor->shape().size());
    auto real_end = end[dim] > old_shape[dim] ? old_shape[dim] : end[dim];
    auto len = real_end - begin[dim];
    if (len <= 0) {
      (void)new_shape.emplace_back(0);
    } else {
      auto shape_dim = (len + step[dim] - 1) / step[dim];
      (void)new_shape.emplace_back(shape_dim);
    }
    new_strides[dim] *= step[dim];
  }

  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, old_storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}

REG_VIEW_STRIDES_CALC_FUN(StridedSlice, StridedSliceCalc);
}  // namespace mindspore::ops
