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
#include "ops/view/unstack_strides_calc.h"
#include <vector>
#include <memory>
#include <set>
#include <string>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore::ops {
constexpr size_t kUnstackCalcInputsNum = 1;
TensorStorageInfoPtrList UnstackCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kUnstackCalcInputsNum) || !inputs[0]->isa<tensor::BaseTensor>()) {
    return {};
  }

  auto tensor = inputs[0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  auto type = tensor->Dtype();
  (void)CheckAndConvertUtils::CheckTypeValid("input_x", type, common_valid_types_with_complex_and_bool, "Unstack");
  auto axis_value_ptr = prim->GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(axis_value_ptr);
  auto dim = GetValue<int64_t>(axis_value_ptr);

  auto old_tensor_info = GetOldTensorInfo(tensor);
  auto oldShape = old_tensor_info->old_shape;
  auto oldStrides = old_tensor_info->old_strides;
  auto oldStorageOffset = old_tensor_info->old_offset;
  const auto ndims = oldShape.size();

  (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(ndims), kGreaterEqual, 1, "Unstack");
  CheckAndConvertUtils::CheckInRange("axis value", dim, kIncludeLeft, {-ndims, ndims}, "Unstack");
  dim = DynamicDimWrap(dim, ndims);
  int64_t size = oldShape[dim];
  (void)CheckAndConvertUtils::CheckInteger("output_num", size, kGreaterEqual, 0, "Unstack");

  std::vector<TensorStorageInfoPtr> res_storage_info(size);
  for (int64_t d = 0; d < size; d++) {
    ShapeVector newShape(oldShape.begin(), oldShape.end());
    StridesVecotr newStrides(oldStrides.begin(), oldStrides.end());
    auto newStorageOffset = oldStorageOffset + LongToSize(d * newStrides[dim]);

    newShape.erase(newShape.begin() + dim);
    newStrides.erase(newStrides.begin() + dim);
    bool is_contiguous = IsContiguous(newShape, newStrides);

    auto newStorageInfo = std::make_shared<TensorStorageInfo>(
      newShape, newStrides, newStorageOffset, old_tensor_info->ori_shape, old_tensor_info->ori_strides, is_contiguous);
    res_storage_info[d] = newStorageInfo;
  }
  return res_storage_info;
}
REG_TUPLE_OUT_VIEW_STRIDES_CALC_FUN(Unstack, UnstackCalc);

}  // namespace mindspore::ops
