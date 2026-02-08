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

#include "view/view_strides_calc.h"
#include <vector>
#include <memory>
#include <utility>
#include "view/reshape_strides_calc.h"

namespace mindspore::ops {
TensorStorageInfoPtrList ViewStridesCalc(const std::vector<int64_t> &cur_shape, const std::vector<int64_t> &cur_strides,
                                         const TensorStorageInfoPtr &cur_storage_info,
                                         const std::vector<int64_t> &shape) {
  TensorStorageInfoPtrList storage_info_list;
  auto new_storage_info = ReshapeStridesCalc(cur_shape, cur_strides, cur_storage_info, shape);
  if (MS_UNLIKELY(new_storage_info == nullptr)) {
    MS_EXCEPTION(ValueError)
      << "view shape " << shape << " is not compatible with input tensor's shape " << cur_shape << " and stride "
      << cur_strides << " (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.";
  }
  return {std::move(new_storage_info)};
}

TensorStorageInfoPtrList ViewBasicTypeCalc(const tensor::TensorPtr &input_tensor, const std::vector<int64_t> &shape) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  return ViewStridesCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(), shape);
}
}  // namespace mindspore::ops
