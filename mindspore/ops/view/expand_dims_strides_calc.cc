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

#include "view/expand_dims_strides_calc.h"
#include <memory>
#include <utility>
#include <vector>

namespace mindspore::ops {
TensorStorageInfoPtrList ExpandDimsStrideCalc(const std::vector<int64_t> &old_shape,
                                              const std::vector<int64_t> &old_strides,
                                              const TensorStorageInfoPtr &storage_info, const int64_t &axis) {
  MS_LOG(DEBUG) << "ExpandDims: input shape " << old_shape << ", input stride " << old_strides << ", storage_info "
                << (storage_info != nullptr ? storage_info->ToString() : "null") << ", dim " << axis;
  auto [ori_shape, ori_strides, storage_offset] = GetOriShapeStridesAndOffset(old_shape, old_strides, storage_info);

  bool is_contiguous = storage_info ? storage_info->is_contiguous : true;

  int64_t old_dim = static_cast<int64_t>(old_shape.size());
  int64_t axis_new = DynamicDimWrap(axis, old_dim + 1);

  auto new_shape = old_shape;
  auto new_strides = old_strides;
  int64_t new_stride = axis_new >= old_dim ? 1 : old_shape[axis_new] * old_strides[axis_new];
  (void)new_shape.insert(new_shape.begin() + axis_new, 1);
  (void)new_strides.insert(new_strides.begin() + axis_new, new_stride);

  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(std::move(new_shape), std::move(new_strides), storage_offset,
                                        std::move(ori_shape), std::move(ori_strides), is_contiguous);

  MS_LOG(DEBUG) << "ExpandDims: output storage_info " << new_storage_info->ToString();
  return {std::move(new_storage_info)};
}

TensorStorageInfoPtrList ExpandDimsBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                                 const int64_t &axis) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  return ExpandDimsStrideCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(), axis);
}

TensorStorageInfoPtrList ExpandDimsCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (!inputs[kInputIndex0]->isa<tensor::Tensor>() || !inputs[kInputIndex1]->isa<IntegerImm>()) {
    return {};
  }
  auto input_tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto axis = GetValue<int64_t>(inputs[kInputIndex1]);
  return ExpandDimsBasicTypeCalc(input_tensor, axis);
}

REG_VIEW_STRIDES_CALC_FUN(ExpandDims, ExpandDimsCalc);
}  // namespace mindspore::ops
