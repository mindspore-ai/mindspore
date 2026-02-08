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

#include "view/unstack_strides_calc.h"

#include <vector>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "utils/check_convert_utils.h"
#include "ops_utils/op_utils.h"

namespace mindspore::ops {
TensorStorageInfoPtrList UnstackStridesCalc(const std::vector<int64_t> &old_shape,
                                            const std::vector<int64_t> &old_strides,
                                            const TensorStorageInfoPtr &old_storage_info, const int64_t &dim) {
  auto [ori_shape, ori_strides, old_storage_offset] =
    GetOriShapeStridesAndOffset(old_shape, old_strides, old_storage_info);

  const auto ndims = old_shape.size();
  MS_CHECK_VALUE(ndims >= 1,
                 "For 'Unstack', input's rank should be greater equal to 1, but got " + std::to_string(ndims));
  auto dim_new = DynamicDimWrap(dim, ndims);
  int64_t output_num = old_shape[dim_new];
  std::vector<TensorStorageInfoPtr> storage_info_list;
  storage_info_list.reserve(output_num);
  for (int64_t i = 0; i < output_num; i++) {
    ShapeVector newShape(old_shape);
    StridesVecotr newStrides(old_strides);
    auto new_storage_offset = old_storage_offset + LongToSize(i * newStrides[dim_new]);

    newShape.erase(newShape.begin() + dim_new);
    newStrides.erase(newStrides.begin() + dim_new);
    bool is_contiguous = IsContiguous(newShape, newStrides);

    auto new_storage_info = std::make_shared<TensorStorageInfo>(
      std::move(newShape), std::move(newStrides), new_storage_offset, ori_shape, ori_strides, is_contiguous);
    storage_info_list.push_back(std::move(new_storage_info));
  }

  return storage_info_list;
}

TensorStorageInfoPtrList UnstackCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (!inputs[kInputIndex0]->isa<tensor::Tensor>()) {
    return {};
  }
  auto tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  auto type = tensor->Dtype();
  (void)CheckAndConvertUtils::CheckTypeValid("input_x", type, common_valid_types_with_complex_and_bool, "Unstack");
  auto axis_value_ptr = prim->GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(axis_value_ptr);
  auto dim = GetValue<int64_t>(axis_value_ptr);
  return UnstackStridesCalc(tensor->shape(), tensor->stride(), tensor->storage_info(), dim);
}

REG_TUPLE_OUT_VIEW_STRIDES_CALC_FUN(Unstack, UnstackCalc);
}  // namespace mindspore::ops
