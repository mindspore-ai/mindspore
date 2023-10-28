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

#include "ops/view/selectview_strides_calc.h"
#include <vector>
#include <memory>

namespace mindspore::ops {
constexpr size_t kGatherInputsNum = 3;

TensorStorageInfoPtrList SelectViewCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kGatherInputsNum)) {
    return {};
  }
  auto input_tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  int64_t dim;
  if (inputs[kInputIndex2]->isa<tensor::Tensor>()) {
    auto dim_tensor = inputs[kInputIndex2]->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(dim_tensor);
    dim = *(static_cast<int64_t *>(dim_tensor->data_c()));
  } else {
    dim = GetValue<int64_t>(inputs[kInputIndex2]);
  }
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto old_storage_offset = old_tensor_info->old_offset;
  dim = DynamicDimWrap(dim, old_shape.size());
  auto index = GetValue<int64_t>(inputs[kInputIndex1]);
  old_storage_offset += LongToSize(index * old_strides[dim]);
  auto new_strides = old_strides;
  auto new_shape = old_shape;
  (void)new_shape.erase(new_shape.begin() + dim);
  (void)new_strides.erase(new_strides.begin() + dim);
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, old_storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}

REG_VIEW_STRIDES_CALC_FUN(SelectView, SelectViewCalc);
}  // namespace mindspore::ops
