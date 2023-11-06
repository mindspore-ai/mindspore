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
#include <algorithm>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/view/split_strides_calc.h"

namespace {
constexpr size_t kSplitInputsNum = 1;
}

namespace mindspore::ops {
void SplitInputsCheck(const int64_t &output_num, const int64_t &axis, const std::vector<int64_t> &tensor_shape) {
  if (output_num <= 0) {
    MS_EXCEPTION(ValueError) << "For Split, the output_num is invalid.";
  }

  if (tensor_shape[axis] % output_num != 0) {
    MS_EXCEPTION(ValueError) << "For Split, the tensor_shape is invalid.";
  }
}

TensorStorageInfoPtrList SplitCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kSplitInputsNum) || !inputs[0]->isa<tensor::Tensor>()) {
    MS_LOG(EXCEPTION) << "inputs num is invalid, num:" << inputs.size();
  }

  auto axis_ptr = prim->GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(axis_ptr);
  auto axis = GetValue<int64_t>(axis_ptr);

  auto output_num_ptr = prim->GetAttr(kOutputNum);
  MS_EXCEPTION_IF_NULL(output_num_ptr);
  auto output_num = GetValue<int64_t>(output_num_ptr);

  auto input_tensor = inputs[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto input_type = input_tensor->Dtype();
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, common_valid_types_with_complex_and_bool,
                                             prim->name());
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  MS_EXCEPTION_IF_NULL(old_tensor_info);
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto old_storage_offset = old_tensor_info->old_offset;

  const auto ndim = old_shape.size();
  const auto wrap_axis = DynamicDimWrap(axis, ndim);
  SplitInputsCheck(output_num, wrap_axis, old_shape);

  size_t splits_section_size = old_shape[wrap_axis] / output_num;

  std::vector<TensorStorageInfoPtr> storage_info_list;

  auto new_shape = old_shape;
  new_shape[wrap_axis] = splits_section_size;
  auto new_strides = old_strides;

  for (int64_t idx = 0; idx < output_num; idx++) {
    size_t new_storage_offset = old_storage_offset + idx * splits_section_size * new_strides[wrap_axis];

    auto new_storage_info =
      std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_storage_offset, old_tensor_info->ori_shape,
                                          old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
    storage_info_list.emplace_back(new_storage_info);
  }

  return storage_info_list;
}

REG_TUPLE_OUT_VIEW_STRIDES_CALC_FUN(Split, SplitCalc);
}  // namespace mindspore::ops
