/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "view/as_strided_strides_calc.h"
#include <vector>
#include <memory>

namespace mindspore::ops {
constexpr size_t kAsStridedInputsNum = 4;
TensorStorageInfoPtrList AsStridedBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                                const std::vector<int64_t> &size, const std::vector<int64_t> &stride,
                                                const int64_t &storage_offset) {
  if (std::any_of(size.begin(), size.end(), [](const int &shape_i) { return shape_i < -1; })) {
    MS_EXCEPTION(ValueError) << "For primitive[AsStrided], the component of shape can't be less than -1, but got "
                             << size;
  }
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  // To do check
  auto new_storage_info = std::make_shared<TensorStorageInfo>(size, stride, storage_offset, old_tensor_info->ori_shape,
                                                              old_tensor_info->ori_strides, IsContiguous(size, stride));
  return {new_storage_info};
}
}  // namespace mindspore::ops
