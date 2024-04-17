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
#include "ops/view/squeeze_strides_calc.h"
#include <vector>
#include <memory>
#include <string>
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
namespace {
constexpr size_t kSqueezeCalcInputsNum = 1;
constexpr auto kSqueezedNum = 1;
}  // namespace
TensorStorageInfoPtrList SqueezeCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kSqueezeCalcInputsNum) || !inputs[0]->isa<tensor::Tensor>()) {
    return {};
  }
  auto tensor = inputs[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  auto value_str = prim->GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(value_str);
  auto axis = GetValue<std::vector<int64_t>>(value_str);
  auto old_tensor_info = GetOldTensorInfo(tensor);
  auto oldShape = old_tensor_info->old_shape;
  auto oldStrides = old_tensor_info->old_strides;
  auto oldStorageOffset = old_tensor_info->old_offset;
  const auto ndims = oldShape.size();

  if (ndims == 0) {
    bool is_contiguous = IsContiguous(oldShape, oldStrides);
    auto newStorageInfo = std::make_shared<TensorStorageInfo>(
      oldShape, oldStrides, oldStorageOffset, old_tensor_info->ori_shape, old_tensor_info->ori_strides, is_contiguous);
    return {newStorageInfo};
  }

  std::vector<bool> seen_dims(ndims, false);

  if (axis.empty()) {
    for (size_t i = 0; i < ndims; i++) {
      seen_dims[i] = true;
    }
  } else {
    for (int64_t dim : axis) {
      CheckAndConvertUtils::CheckInRange<int64_t>("element or value of axis", dim, kIncludeLeft, {-ndims, ndims},
                                                  "Squeeze");
      const auto wrap_dim = DynamicDimWrap(dim, ndims);
      // If shape dims contain unknown dim, ignore it.
      if (oldShape[LongToSize(wrap_dim)] != abstract::Shape::kShapeDimAny) {
        const std::string ith_shape = "input_x.shape[" + std::to_string(wrap_dim) + "]";
        (void)CheckAndConvertUtils::CheckValue<int64_t>(ith_shape, oldShape[wrap_dim], kEqual, kSqueezedNum, "Squeeze");
      }
      seen_dims[wrap_dim] = true;
    }
  }

  // delete shape dim if it equals one in seen dimension.
  ShapeVector newShape;
  StridesVecotr newStrides;
  for (size_t i = 0; i < ndims; i++) {
    if (!seen_dims[i] || oldShape[i] != kSqueezedNum) {
      newShape.push_back(oldShape[i]);
      newStrides.push_back(oldStrides[i]);
    }
  }

  bool is_contiguous = IsContiguous(newShape, newStrides);
  auto newStorageInfo = std::make_shared<TensorStorageInfo>(
    newShape, newStrides, oldStorageOffset, old_tensor_info->ori_shape, old_tensor_info->ori_strides, is_contiguous);
  return {newStorageInfo};
}
REG_VIEW_STRIDES_CALC_FUN(Squeeze, SqueezeCalc);
}  // namespace mindspore::ops
