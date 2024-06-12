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
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/view/split_tensor_strides_calc.h"

namespace {
constexpr size_t kSplitTensorInputsNum = 3;
}

namespace mindspore::ops {
void SplitTensorInputsCheck(const PrimitivePtr &prim, const int64_t &output_num, const int64_t &axis) {
  auto prim_name = prim->name();
  if (output_num <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', output_num must be positive, but got " << output_num << ".";
  }
}

TensorStorageInfoPtrList SplitTensorCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kSplitTensorInputsNum) || !inputs[kInputIndex0]->isa<tensor::BaseTensor>()) {
    MS_LOG(EXCEPTION) << "inputs num is invalid, num:" << inputs.size();
  }

  auto input_tensor = inputs[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto split_size = GetValue<int64_t>(inputs[kInputIndex1]);
  auto dim = GetValue<int64_t>(inputs[kInputIndex2]);
  auto input_type = input_tensor->Dtype();
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, common_valid_types_with_complex_and_bool,
                                             prim->name());
  auto tensor_shape = GetOldTensorInfo(input_tensor);
  MS_EXCEPTION_IF_NULL(tensor_shape);
  auto old_shape = tensor_shape->old_shape;
  auto old_strides = tensor_shape->old_strides;

  auto rank = SizeToLong(old_shape.size());
  MS_CHECK_VALUE(rank > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("rank", rank, kGreaterEqual, 1, prim));
  const auto ndim = old_shape.size();
  const auto wrap_dim = DynamicDimWrap(dim, ndim);

  // Check if the output quantity is positive
  SplitTensorInputsCheck(prim, split_size, wrap_dim);

  // Calculate the number of sub tensors after segmentation
  auto num_splits = (old_shape[wrap_dim] + split_size - 1) / split_size;

  // Create a storage information list
  std::vector<TensorStorageInfoPtr> storage_info_list;

  for (int64_t idx = 0; idx < num_splits; ++idx) {
    // Calculate the shape and length of sub tensors
    std::vector<int64_t> slice_shape = old_shape;

    // Calculate the size of a sub tensor in a specified dimension
    int64_t slice_size = split_size;
    if (idx == num_splits - 1) {
      // For the last sub tensor, ensure that it contains all remaining elements in that dimension
      slice_size = old_shape[wrap_dim] - (idx * split_size);
    }
    slice_shape[wrap_dim] = slice_size;
    // Calculate the storage offset of sub tensors
    size_t new_storage_offset = tensor_shape->old_offset + LongToSize(idx * split_size * old_strides[wrap_dim]);
    auto new_storage_info =
      std::make_shared<TensorStorageInfo>(slice_shape, old_strides, new_storage_offset, tensor_shape->ori_shape,
                                          tensor_shape->ori_strides, IsContiguous(slice_shape, old_strides));
    storage_info_list.emplace_back(new_storage_info);
  }
  return storage_info_list;
}
REG_TUPLE_OUT_VIEW_STRIDES_CALC_FUN(SplitTensor, SplitTensorCalc);
}  // namespace mindspore::ops
