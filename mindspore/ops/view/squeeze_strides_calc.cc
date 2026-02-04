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
#include "view/squeeze_strides_calc.h"
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "utils/check_convert_utils.h"
#include "ops/op_def_utils.h"

namespace mindspore::ops {
namespace {
constexpr auto kSqueezedDimValue = 1;
inline std::pair<std::vector<int64_t>, std::vector<int64_t>> InferSqueezeShapeAndStrides(
  const std::vector<int64_t> &old_shape, const std::vector<int64_t> &old_strides, const std::vector<int64_t> &axis) {
  const auto ndims = old_shape.size();
  std::vector<int64_t> new_shape;
  new_shape.reserve(ndims);
  std::vector<int64_t> new_strides;
  new_strides.reserve(ndims);
  if (MS_UNLIKELY(axis.empty())) {
    for (size_t i = 0; i < ndims; ++i) {
      if (old_shape[i] != kSqueezedDimValue) {
        new_shape.push_back(old_shape[i]);
        new_strides.push_back(old_strides[i]);
      }
    }
    return {std::move(new_shape), std::move(new_strides)};
  }

  constexpr size_t bit_set_size = 32;
  MS_CHECK_VALUE(ndims < bit_set_size,
                 CheckAndConvertUtils::FormatCommMsg("For 'Squeeze', input's rank should be less than ", bit_set_size,
                                                     ", but got ", ndims));
  std::bitset<bit_set_size> seen_dims{};
  for (int64_t dim : axis) {
    const auto wrap_dim = DynamicDimWrap(dim, ndims);
    seen_dims.set(wrap_dim, true);
  }
  for (size_t i = 0; i < ndims; i++) {
    if (MS_LIKELY(!seen_dims.test(i) || old_shape[i] != kSqueezedDimValue)) {
      new_shape.push_back(old_shape[i]);
      new_strides.push_back(old_strides[i]);
    }
  }
  return {std::move(new_shape), std::move(new_strides)};
}
}  // namespace
TensorStorageInfoPtrList SqueezeBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                              const std::vector<int64_t> &axis) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &old_shape = input_tensor->shape();
  const auto &old_strides = input_tensor->stride();
  auto [ori_shape, ori_strides, storage_offset] =
    GetOriShapeStridesAndOffset(old_shape, old_strides, input_tensor->storage_info());

  TensorStorageInfoPtrList newStorageInfoList{};
  newStorageInfoList.reserve(1);

  const auto ndims = old_shape.size();
  if (MS_UNLIKELY(ndims == 0)) {
    newStorageInfoList.push_back(std::make_shared<TensorStorageInfo>(std::vector<int64_t>{}, std::vector<int64_t>{},
                                                                     storage_offset, std::move(ori_shape),
                                                                     std::move(ori_strides), true));
    return newStorageInfoList;
  }

  auto [new_shape, new_strides] = InferSqueezeShapeAndStrides(old_shape, old_strides, axis);
  bool is_contiguous =
    new_shape.size() == old_shape.size() ? input_tensor->is_contiguous() : IsContiguous(new_shape, new_strides);
  newStorageInfoList.push_back(std::make_shared<TensorStorageInfo>(std::move(new_shape), std::move(new_strides),
                                                                   storage_offset, std::move(ori_shape),
                                                                   std::move(ori_strides), is_contiguous));

  return newStorageInfoList;
}
}  // namespace mindspore::ops
