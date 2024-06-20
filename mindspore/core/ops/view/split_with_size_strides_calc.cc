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
#include "utils/convert_utils_base.h"
#include "ops/view/split_with_size_strides_calc.h"

namespace {
constexpr size_t kSplitWithSizeInputsNum = 3;
}

namespace mindspore::ops {
void SplitSizeInputsCheck(const PrimitivePtr &prim, const int64_t &output_num, const int64_t &axis,
                          const std::vector<int64_t> &tensor_shape) {
  auto prim_name = prim->name();
  if (output_num != tensor_shape[axis]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', output_num must be equal with dimIndex, but got "
                             << output_num << ".";
    return;
  }
}

TensorStorageInfoPtrList SplitWithSizeCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kSplitWithSizeInputsNum) || !inputs[kInputIndex0]->isa<tensor::BaseTensor>()) {
    MS_LOG(EXCEPTION) << "inputs num is invalid, num:" << inputs.size();
  }

  auto input_tensor = inputs[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto split_size = GetValue<std::vector<int64_t>>(inputs[kInputIndex1]);
  auto dim = GetValue<int64_t>(inputs[kInputIndex2]);
  auto input_type = input_tensor->Dtype();
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, common_valid_types_with_complex_and_bool,
                                             prim->name());
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  MS_EXCEPTION_IF_NULL(old_tensor_info);
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto old_storage_offset = old_tensor_info->old_offset;

  auto rank = SizeToLong(old_shape.size());
  MS_CHECK_VALUE(rank > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("rank", rank, kGreaterEqual, 1, prim));
  const auto ndim = old_shape.size();
  const auto wrap_dim = DynamicDimWrap(dim, ndim);
  int64_t sum_split_size = std::accumulate(split_size.begin(), split_size.end(), 0);
  SplitSizeInputsCheck(prim, sum_split_size, wrap_dim, old_shape);

  std::vector<TensorStorageInfoPtr> storage_info_list;
  size_t current_offset = old_storage_offset;
  for (const auto &split_iter : split_size) {
    std::vector<int64_t> slice_shape = old_shape;
    slice_shape[wrap_dim] = split_iter;

    // Calculate the storage offset of sub tensors
    size_t new_storage_offset = current_offset;
    // Update current offset
    current_offset += LongToSize(split_iter * old_strides[wrap_dim]);

    // Creating storage information for sub tensors
    auto new_storage_info =
      std::make_shared<TensorStorageInfo>(slice_shape, old_strides, new_storage_offset, old_tensor_info->ori_shape,
                                          old_strides, IsContiguous(slice_shape, old_strides));
    storage_info_list.emplace_back(new_storage_info);
  }
  return storage_info_list;
}
REG_TUPLE_OUT_VIEW_STRIDES_CALC_FUN(SplitWithSize, SplitWithSizeCalc);
}  // namespace mindspore::ops
