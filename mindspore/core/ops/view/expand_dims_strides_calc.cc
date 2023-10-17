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

#include "ops/view/expand_dims_strides_calc.h"
#include <memory>

namespace mindspore::ops {
constexpr size_t kExpandDimsInputsNum = 2;

TensorStorageInfoPtrList ExpandDimsCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kExpandDimsInputsNum) || !inputs[0]->isa<tensor::Tensor>() ||
      !inputs[1]->isa<IntegerImm>()) {
    return {};
  }

  auto input_tensor = inputs[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto old_storage_offset = old_tensor_info->old_offset;

  auto new_shape = old_shape;
  auto new_strides = old_strides;

  int64_t dim_size = SizeToLong(new_shape.size());
  auto axis = GetValue<int64_t>(inputs[1]);
  axis = DynamicDimWrap(axis, new_shape.size() + 1);
  int64_t tmp_strides = axis >= dim_size ? 1 : new_shape[axis] * new_strides[axis];
  (void)new_strides.insert(new_strides.begin() + axis, tmp_strides);
  (void)new_shape.insert(new_shape.begin() + axis, 1);
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, old_storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}

REG_VIEW_STRIDES_CALC_FUN(ExpandDims, ExpandDimsCalc);
}  // namespace mindspore::ops
