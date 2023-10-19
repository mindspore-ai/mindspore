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

#include "ops/view/broadcast_to_strides_calc.h"
#include <memory>
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
constexpr size_t kBroadCastToInputsNum = 1;
bool BroadcastToCheck(const std::vector<int64_t> &input_x, const std::vector<int64_t> &x_shape) {
  CheckAndConvertUtils::Check("x shape", SizeToLong(x_shape.size()), kLessEqual, SizeToLong(input_x.size()),
                              "BroadcastTo");
  auto outer_dim_offset = input_x.size() - x_shape.size();
  bool flag = true;
  if (input_x.end() == find(input_x.begin(), input_x.end(), -1)) {
    flag = false;
  } else {
    flag = true;
  }
  if (flag) {
    for (size_t i = 0; i < input_x.size(); i++) {
      if (input_x[i] == -1) {
        if (i < outer_dim_offset) {
          return false;
        }
      }
    }
  }
  for (size_t i = 0; i < x_shape.size(); i++) {
    if (input_x[i + outer_dim_offset] == -1) {
      continue;
    }
    if (input_x[i + outer_dim_offset] != x_shape[i] && x_shape[i] != 1) {
      return false;
    }
  }
  return true;
}

TensorStorageInfoPtrList BroadCastToCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kBroadCastToInputsNum) || !inputs[0]->isa<tensor::Tensor>()) {
    return {};
  }

  auto input_tensor = inputs[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto value_ptr = prim->GetAttr(kShape);
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto input_x = GetValue<std::vector<int64_t>>(value_ptr);
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto old_storage_offset = old_tensor_info->old_offset;
  if (!BroadcastToCheck(input_x, old_shape)) {
    return {};
  }
  int64_t ndim = SizeToInt(input_x.size());
  int64_t tensor_ndim = SizeToInt(old_shape.size());
  std::vector<int64_t> new_strides(ndim);
  if (tensor_ndim == 0) {
    auto new_storage_info =
      std::make_shared<TensorStorageInfo>(input_x, new_strides, old_storage_offset, old_tensor_info->ori_shape,
                                          old_tensor_info->ori_strides, IsContiguous(input_x, new_strides));
    return {new_storage_info};
  }
  std::vector<int64_t> new_shape(ndim);
  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dim = tensor_ndim - 1 - offset;
    auto size = (dim >= 0) ? old_shape[dim] : 1;
    auto stride = (dim >= 0) ? old_strides[dim] : new_shape[i + 1] * new_strides[i + 1];
    auto target_size = input_x[i];
    if (target_size == -1) {
      target_size = size;
    }
    if (size != target_size) {
      size = target_size;
      stride = 0;
    }
    new_shape[i] = size;
    new_strides[i] = stride;
  }
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, old_storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}

REG_VIEW_STRIDES_CALC_FUN(BroadcastTo, BroadCastToCalc);
}  // namespace mindspore::ops
