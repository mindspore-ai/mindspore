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
#include <algorithm>
#include <memory>
#include <utility>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "view/split_with_size_strides_calc.h"

namespace mindspore::ops {
namespace {
inline static void SplitSizeInputsCheck(const std::vector<int64_t> &split_size, const int64_t &axis,
                                        const std::vector<int64_t> &tensor_shape) {
  MS_CHECK_VALUE(split_size.size() > 0,
                 CheckAndConvertUtils::FormatCommMsg("For SplitWithSize, the size of split_size should > 0, but got",
                                                     split_size.size()));
  int64_t sum_split_size = std::accumulate(split_size.begin(), split_size.end(), 0);
  MS_CHECK_VALUE(sum_split_size == tensor_shape[axis],
                 CheckAndConvertUtils::FormatCommMsg("For 'SplitWithSize',  the sum of split_size should be equal to ",
                                                     tensor_shape[axis], "(input.shape[", axis,
                                                     "]), but got split_sizes: ", split_size));
}
}  // namespace

TensorStorageInfoPtrList SplitWithSizeStridesCalc(const std::vector<int64_t> &cur_shape,
                                                  const std::vector<int64_t> &cur_strides,
                                                  const TensorStorageInfoPtr &cur_storage_info,
                                                  const std::vector<int64_t> &split_size, const int64_t &dim) {
  MS_LOG(DEBUG) << "SplitWithSize: input shape " << cur_shape << ", input stride " << cur_strides << ", storage_info "
                << (cur_storage_info != nullptr ? cur_storage_info->ToString() : "null") << ", split_size "
                << split_size << ", dim " << dim;
  auto [ori_shape, ori_strides, current_offset] = GetOriShapeStridesAndOffset(cur_shape, cur_strides, cur_storage_info);

  auto rank = SizeToLong(cur_shape.size());
  MS_CHECK_VALUE(rank > 0, CheckAndConvertUtils::FormatCommMsg("For SplitWithSize, rank should > 0, but got", rank));
  const auto ndim = cur_shape.size();
  const auto wrap_dim = DynamicDimWrap(dim, ndim);
  SplitSizeInputsCheck(split_size, wrap_dim, cur_shape);

  std::vector<TensorStorageInfoPtr> storage_info_list;
  storage_info_list.reserve(split_size.size());
  for (size_t i = 0; i < split_size.size(); ++i) {
    auto split_iter = split_size[i];

    std::vector<int64_t> slice_shape(cur_shape);
    slice_shape[wrap_dim] = split_iter;

    // Calculate the storage offset of sub tensors
    size_t new_storage_offset = current_offset;
    // Update current offset
    current_offset += LongToSize(split_iter * cur_strides[wrap_dim]);

    // Creating storage information for sub tensors
    bool is_contiguous = IsContiguous(slice_shape, cur_strides);
    auto new_storage_info = std::make_shared<TensorStorageInfo>(std::move(slice_shape), cur_strides, new_storage_offset,
                                                                ori_shape, ori_strides, is_contiguous);
    MS_LOG(DEBUG) << "SplitWithSize: output[" << i << "] storage_info " << new_storage_info->ToString();
    storage_info_list.push_back(std::move(new_storage_info));
  }

  return storage_info_list;
}

TensorStorageInfoPtrList SplitWithSizeBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                                    const std::vector<int64_t> &split_size, const int64_t &dim) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  return SplitWithSizeStridesCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(),
                                  split_size, dim);
}
}  // namespace mindspore::ops
