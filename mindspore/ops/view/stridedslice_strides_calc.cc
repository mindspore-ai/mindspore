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
#include "view/stridedslice_strides_calc.h"
#include <vector>
#include <memory>
#include "ops_utils/op_constants.h"
#include "utils/check_convert_utils.h"
#include "ops/op_def_utils.h"

namespace mindspore::ops {
constexpr size_t kStridedSliceCalcInputsNumWithoutMask = 4;
constexpr size_t kStridedSliceCalcInputsNumWithMask = 9;
void ConvertNegToPos(std::vector<int64_t> *begin, std::vector<int64_t> *end, const std::vector<int64_t> &tensor_shape) {
  MS_CHECK_VALUE(begin->size() == tensor_shape.size() && end->size() == tensor_shape.size(),
                 CheckAndConvertUtils::FormatCommMsg("Convert shape size is not equal"));
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
    if ((*begin)[i] == (*end)[i]) {
      (*begin)[i] = 0;
      (*end)[i] = 0;
    }
  }
}
void VectorEmplace(std::vector<int64_t> *vec, size_t number, size_t dst_size) {
  if ((*vec).size() >= dst_size) {
    return;
  }
  auto num = dst_size - vec->size();
  for (size_t i = 0; i < num; ++i) {
    (void)vec->emplace_back(number);
  }
}

void VectorEmplace(std::vector<int64_t> *vec, std::vector<int64_t> *number_vec, size_t dst_size) {
  if ((*vec).size() >= dst_size) {
    return;
  }

  MS_CHECK_VALUE(number_vec->size() == dst_size,
                 CheckAndConvertUtils::FormatCommMsg("dst_size is not equal to number_vec.size(), dst_size:", dst_size,
                                                     ",  number_vec.size():", number_vec->size()));

  auto begin = vec->size();
  for (size_t i = begin; i < dst_size; ++i) {
    (void)vec->emplace_back(number_vec->at(i));
  }
}

bool CheckMaskIsZero(const std::vector<ValuePtr> &inputs) {
  auto begin_mask = GetValue<int64_t>(inputs[kIndex4]);
  auto end_mask = GetValue<int64_t>(inputs[kIndex5]);
  auto ellipsis_mask = GetValue<int64_t>(inputs[kIndex6]);
  auto new_axis_mask = GetValue<int64_t>(inputs[kIndex7]);
  auto shrink_axis_mask = GetValue<int64_t>(inputs[kIndex8]);

  if (begin_mask == 0 && end_mask == 0 && ellipsis_mask == 0 && new_axis_mask == 0 && shrink_axis_mask == 0) {
    return true;
  }
  return false;
}

bool CheckStridedSliceInputs(const std::vector<ValuePtr> &inputs) {
  bool nullptr_input_exist =
    std::any_of(inputs.cbegin(), inputs.cend(), [](const ValuePtr &v) { return v == nullptr; });
  if (nullptr_input_exist) {
    return false;
  }

  if (inputs.size() == kStridedSliceCalcInputsNumWithoutMask) {
    return true;
  }

  if (inputs.size() != kStridedSliceCalcInputsNumWithMask) {
    return false;
  }

  return CheckMaskIsZero(inputs);
}

TensorStorageInfoPtrList StridedSliceStridesCalc(const OldTensorInfoPtr old_tensor_info, size_t size,
                                                 std::vector<int64_t> *shape, std::vector<int64_t> *begin,
                                                 std::vector<int64_t> *end, std::vector<int64_t> *step) {
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto old_storage_offset = old_tensor_info->old_offset;

  VectorEmplace(begin, size_t(0), size);
  VectorEmplace(end, shape, size);
  VectorEmplace(step, 1, size);
  ConvertNegToPos(begin, end, old_shape);

  for (size_t i = 0; i < begin->size(); ++i) {
    old_storage_offset += LongToSize(begin->at(i) * old_strides[i]);
  }
  ShapeVector new_shape;
  auto new_strides = old_strides;
  for (size_t i = 0; i < size; ++i) {
    auto dim = DynamicDimWrap(i, shape->size());
    auto real_end = end->at(dim) > old_shape[dim] ? old_shape[dim] : end->at(dim);
    auto len = real_end - begin->at(dim);
    if (len <= 0) {
      (void)new_shape.emplace_back(0);
    } else {
      auto shape_dim = (len + step->at(dim) - 1) / step->at(dim);
      (void)new_shape.emplace_back(shape_dim);
    }
    new_strides[dim] *= step->at(dim);
  }

  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, old_storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}

TensorStorageInfoPtrList StridedSliceCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (!CheckStridedSliceInputs(inputs)) {
    return {};
  }
  if (inputs[kInputIndex1]->isa<tensor::Tensor>() || inputs[kInputIndex2]->isa<tensor::Tensor>() ||
      inputs[kInputIndex3]->isa<tensor::Tensor>()) {
    return {};
  }
  auto begin = GetValue<std::vector<int64_t>>(inputs[kInputIndex1]);
  auto end = GetValue<std::vector<int64_t>>(inputs[kInputIndex2]);
  auto step = GetValue<std::vector<int64_t>>(inputs[kInputIndex3]);
  if (IsDynamic(step) || begin.size() != end.size() || begin.size() != step.size() || HasZero(step)) {
    return {};
  }
  auto input_tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto size = input_tensor->shape().size();
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  auto shape = input_tensor->shape();
  return StridedSliceStridesCalc(old_tensor_info, size, &shape, &begin, &end, &step);
}

REG_VIEW_STRIDES_CALC_FUN(StridedSlice, StridedSliceCalc);
}  // namespace mindspore::ops
