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

#include <memory>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/view/slice_ext_strides_calc.h"

namespace {
constexpr size_t kSliceExtInputsNum = 5;
}

namespace mindspore::ops {

TensorStorageInfoPtrList SliceExtCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kSliceExtInputsNum) || !inputs[kInputIndex0]->isa<tensor::BaseTensor>()) {
    MS_LOG(EXCEPTION) << "inputs num is invalid, num:" << inputs.size();
  }

  auto input_tensor = inputs[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto input_type = input_tensor->Dtype();
  const std::set<TypePtr> valid_type = {kInt8, kInt32, kInt64, kUInt8, kFloat16, kFloat32, kBool, kBFloat16};
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, valid_type, prim->name());

  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  MS_EXCEPTION_IF_NULL(old_tensor_info);
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;

  auto dim = GetValue<int64_t>(inputs[kInputIndex1]);
  auto start = GetValue<int64_t>(inputs[kInputIndex2]);
  auto end = GetValue<int64_t>(inputs[kInputIndex3]);
  auto step = GetValue<int64_t>(inputs[kInputIndex4]);
  MS_CHECK_VALUE(step == 1, "step value must be 1");

  int dim_size = SizeToLong(old_shape.size());
  MS_CHECK_VALUE(dim_size > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("rank", dim_size, kGreaterEqual, 1, prim));

  dim = DynamicDimWrap(dim, dim_size);
  auto dim_value = old_shape[dim];
  auto length = end - start;
  MS_CHECK_VALUE(length >= 0, "For Primitive [SliceExt] end should less or less equal start");
  start = start < 0 ? start + dim_value : start;
  MS_CHECK_VALUE(start >= 0 && start <= dim_value, "For Primitive [SliceExt] start exceed range");
  end = start + length;
  MS_CHECK_VALUE(end >= 0 && end <= dim_value, "For Primitive [SliceExt] end exceed range");

  auto new_shape = old_shape;
  new_shape[dim] = length;
  auto new_strides = old_strides;
  size_t new_storage_offset = LongToSize(start * new_strides[dim]);

  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}

REG_VIEW_STRIDES_CALC_FUN(SliceExt, SliceExtCalc);
}  // namespace mindspore::ops
