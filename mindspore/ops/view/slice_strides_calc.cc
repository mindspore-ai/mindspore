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

#include "view/slice_strides_calc.h"
#include <memory>
#include <vector>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace {
constexpr size_t kSliceInputsNum = 3;
}

namespace mindspore::ops {
void SliceInputsCheck(const std::vector<int64_t> &tensor_shape, const std::vector<int64_t> &begin,
                      const std::vector<int64_t> &size) {
  (void)CheckAndConvertUtils::CheckInteger("rank of input_x", SizeToLong(tensor_shape.size()), kGreaterThan, 0,
                                           "Slice");
  if (tensor_shape.size() != begin.size() || tensor_shape.size() != size.size()) {
    MS_EXCEPTION(ValueError) << "For Slice, the shape of input|begin|size must be equal.";
  }

  for (size_t idx = 0; idx < tensor_shape.size(); idx++) {
    if (begin[idx] < 0 || begin[idx] > tensor_shape[idx]) {
      MS_EXCEPTION(ValueError) << "For Slice, the begin is invalid.";
    }
    if (size[idx] == -1) {
      continue;
    }

    if (size[idx] < -1) {
      MS_EXCEPTION(RuntimeError) << "For Slice, the value in size should not be less than -1, but got " << size[idx];
    }

    if (begin[idx] + size[idx] > tensor_shape[idx]) {
      MS_EXCEPTION(ValueError) << "For Slice, the sum of begin_shape[" << idx << "] and size_shape[" << idx
                               << "] must be no greater than input_x_shape[" << idx << "].";
    }
  }
}

TensorStorageInfoPtrList SliceBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                            const std::vector<int64_t> &begin, const std::vector<int64_t> &size) {
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  MS_EXCEPTION_IF_NULL(old_tensor_info);
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto old_storage_offset = old_tensor_info->old_offset;
  SliceInputsCheck(old_shape, begin, size);

  auto new_shape = size;
  auto new_strides = old_strides;
  size_t new_storage_offset = old_storage_offset;

  for (size_t idx = 0; idx < old_shape.size(); idx++) {
    if (new_shape[idx] == -1) {
      new_shape[idx] = old_shape[idx] - begin[idx];
    }
    new_storage_offset += LongToSize(begin[idx] * new_strides[idx]);
  }

  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}
}  // namespace mindspore::ops
