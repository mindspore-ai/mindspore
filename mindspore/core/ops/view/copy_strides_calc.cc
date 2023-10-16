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

#include "ops/view/copy_strides_calc.h"
#include <vector>
#include <memory>

namespace mindspore::ops {
constexpr size_t kCopyInputsNum = 2;

TensorStorageInfoPtrList CopyWithSliceCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kCopyInputsNum)) {
    MS_LOG(EXCEPTION) << "inputs num is invalid, num:" << inputs.size();
  }
  auto self_tensor = inputs[0]->cast<tensor::TensorPtr>();
  auto src_tensor = inputs[1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(self_tensor);
  MS_EXCEPTION_IF_NULL(src_tensor);

  auto storage_info = self_tensor->storage_info();
  if (storage_info == nullptr) {
    const auto &old_shape = self_tensor->shape();
    const auto &old_strides = GetOriStrides(old_shape);
    storage_info = std::make_shared<TensorStorageInfo>(old_shape, old_strides, old_shape, old_strides,
                                                       IsContiguous(old_shape, old_strides));
    self_tensor->set_storage_info(storage_info);
  }

  if (src_tensor->storage_info() == nullptr) {
    const auto &old_shape = src_tensor->shape();
    const auto &old_strides = GetOriStrides(old_shape);
    auto src_storage_info = std::make_shared<TensorStorageInfo>(old_shape, old_strides, old_shape, old_strides,
                                                                IsContiguous(old_shape, old_strides));
    src_tensor->set_storage_info(src_storage_info);
  }

  MS_EXCEPTION_IF_NULL(storage_info);
  if (storage_info->shape != src_tensor->storage_info()->shape) {
    MS_LOG(EXCEPTION) << "storage_info->shape is not equal to src_tensor->shape(), storage_info->shape:"
                      << storage_info->shape << ", src_tensor->shape:" << src_tensor->storage_info()->shape;
  }

  if (self_tensor->data_type() != src_tensor->data_type()) {
    MS_LOG(EXCEPTION) << "self_tensor->data_type is not equal to src_tensor->data_type, self_tensor->data_type:"
                      << self_tensor->data_type() << ",  src_tensor->data_type:" << src_tensor->data_type();
  }

  return {storage_info};
}

REG_VIEW_STRIDES_CALC_FUN(CopyWithSlice, CopyWithSliceCalc);
}  // namespace mindspore::ops
