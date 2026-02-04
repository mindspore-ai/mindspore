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

#include "view/chunk_strides_calc.h"
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
TensorStorageInfoPtrList ChunkStridesCalc(const std::vector<int64_t> &old_shape,
                                          const std::vector<int64_t> &old_strides,
                                          const TensorStorageInfoPtr &storage_info, const int64_t &chunks,
                                          const int64_t &dim) {
  MS_LOG(DEBUG) << "Chunk: input shape " << old_shape << ", input stride " << old_strides << ", storage_info "
                << (storage_info != nullptr ? storage_info->ToString() : "null") << ", chunks " << chunks << ", dim "
                << dim;
  const auto ndim = old_shape.size();
  MS_CHECK_VALUE(ndim > 0, "For 'Chunk', input's rank should be greater than 0, but got " + std::to_string(ndim));
  MS_CHECK_VALUE(chunks > 0, "For 'Chunk', chunks should be greater than 0, but got " + std::to_string(chunks));

  const auto wrap_dim = DynamicDimWrap(dim, ndim);
  int64_t dim_size = old_shape[wrap_dim];
  int64_t split_size = (dim_size + chunks - 1) / chunks;
  if (MS_UNLIKELY(dim_size == 0)) {
    if (split_size == 0) {
      TensorStorageInfoPtr new_storage_info{storage_info};
      if (storage_info == nullptr) {
        new_storage_info = std::make_shared<TensorStorageInfo>(old_shape, old_strides, 0, old_shape, old_strides,
                                                               IsContiguous(old_shape, old_strides));
      }
      std::vector<TensorStorageInfoPtr> storage_info_list(chunks, new_storage_info);
      MS_LOG(DEBUG) << "Chunk: outputs' storage_info is all the same, which are " << new_storage_info->ToString();
      return storage_info_list;
    }
    MS_EXCEPTION(ValueError) << "For 'Chunk', output_num must be positive, but got 0.";
  }

  auto [ori_shape, ori_strides, old_offset] = GetOriShapeStridesAndOffset(old_shape, old_strides, storage_info);
  // Calculate the number of sub tensors after segmentation
  auto num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  auto last_split_size = split_size - (split_size * num_splits - dim_size);
  // Create a storage information list
  std::vector<TensorStorageInfoPtr> storage_info_list{};

  for (int64_t idx = 0; idx < num_splits; ++idx) {
    // Calculate the shape and length of sub tensors
    std::vector<int64_t> slice_shape = old_shape;

    // Calculate the size of a sub tensor in a specified dimension
    slice_shape[wrap_dim] = (idx == num_splits - 1) ? last_split_size : split_size;
    // Calculate the storage offset of sub tensors
    size_t new_storage_offset = old_offset + LongToSize(idx * split_size * old_strides[wrap_dim]);
    bool is_contiguous = IsContiguous(slice_shape, old_strides);
    auto new_storage_info = std::make_shared<TensorStorageInfo>(std::move(slice_shape), old_strides, new_storage_offset,
                                                                ori_shape, ori_strides, is_contiguous);
    MS_LOG(DEBUG) << "Chunk: output[" << idx << "] storage_info " << new_storage_info->ToString();
    (void)storage_info_list.emplace_back(std::move(new_storage_info));
  }

  return storage_info_list;
}

TensorStorageInfoPtrList ChunkBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor, const int64_t &chunks,
                                            const int64_t &dim) {
  if (MS_UNLIKELY(chunks < 1)) {
    MS_EXCEPTION(ValueError) << "For 'Chunk', chunks should be greater equal than 1, but got " << chunks;
  }
  MS_EXCEPTION_IF_NULL(input_tensor);
  return ChunkStridesCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(), chunks, dim);
}
}  // namespace mindspore::ops
