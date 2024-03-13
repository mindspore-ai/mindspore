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
constexpr size_t kSplitInputsNum = 3;
}

namespace mindspore::ops {
void SplitInputsCheck(const PrimitivePtr &prim, const int64_t &output_num, const int64_t &axis,
                      const std::vector<int64_t> &tensor_shape) {
  auto prim_name = prim->name();
  if (output_num <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', output_num must be positive, but got " << output_num << ".";
    return;
  }

  if ((!IsDynamic(tensor_shape)) && (tensor_shape[axis] % output_num != 0)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', x_shape[" << axis
                             << "] must be divisible by output_num = " << output_num << ", but got "
                             << tensor_shape[axis];
  }
}

TensorStorageInfoPtrList SplitCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kSplitInputsNum) || !inputs[kInputIndex0]->isa<tensor::BaseTensor>()) {
    MS_LOG(EXCEPTION) << "inputs num is invalid, num:" << inputs.size();
  }

  auto input_tensor = inputs[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto axis = GetValue<int64_t>(inputs[kInputIndex1]);
  auto output_num = GetValue<int64_t>(inputs[kInputIndex2]);
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
  const auto wrap_axis = DynamicDimWrap(axis, ndim);
  SplitInputsCheck(prim, output_num, wrap_axis, old_shape);

  int64_t splits_section_size = old_shape[wrap_axis] / output_num;

  std::vector<TensorStorageInfoPtr> storage_info_list;

  auto new_shape = old_shape;
  new_shape[wrap_axis] = splits_section_size;
  auto new_strides = old_strides;

  for (int64_t idx = 0; idx < output_num; idx++) {
    size_t new_storage_offset = old_storage_offset + LongToSize(idx * splits_section_size * new_strides[wrap_axis]);

    auto new_storage_info =
      std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_storage_offset, old_tensor_info->ori_shape,
                                          old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
    storage_info_list.emplace_back(new_storage_info);
  }

  return storage_info_list;
}

REG_TUPLE_OUT_VIEW_STRIDES_CALC_FUN(Split, SplitCalc);
}  // namespace mindspore::ops
