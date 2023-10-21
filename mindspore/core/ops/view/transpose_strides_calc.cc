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

#include "ops/view/transpose_strides_calc.h"
#include <vector>
#include <memory>
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
constexpr size_t kTransposeCalcInputsNum = 2;

TensorStorageInfoPtrList TransposeCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kTransposeCalcInputsNum) || !inputs[0]->isa<tensor::Tensor>() ||
      !inputs[1]->isa<ValueSequence>()) {
    return {};
  }

  auto tensor = inputs[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);

  const auto &x_shape = tensor->shape();
  (void)CheckAndConvertUtils::CheckInteger("input_x size", SizeToLong(x_shape.size()), kGreaterThan, 0, "Transpose");
  if (x_shape[0] == 0) {
    MS_EXCEPTION(ValueError) << "For 'Transpose', first dim of input_x's shape can not be 0, but got 0.";
  }

  auto old_tensor_info = GetOldTensorInfo(tensor);
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto old_storage_offset = old_tensor_info->old_offset;

  auto dims = CheckAndConvertUtils::CheckTupleInt("perm", inputs[1], "Transpose");
  const auto ndim = old_shape.size();
  if (ndim != dims.size()) {
    return {};
  }

  ShapeVector new_shape(ndim);
  std::vector<int64_t> new_strides(ndim);
  std::vector<bool> seen_dims(ndim, false);

  for (size_t i = 0; i < ndim; i++) {
    const auto wrap_dim = DynamicDimWrap(dims[i], ndim);
    if (seen_dims[wrap_dim]) {
      return {};
    }
    seen_dims[wrap_dim] = true;
    new_shape[i] = old_shape[wrap_dim];
    new_strides[i] = old_strides[wrap_dim];
  }

  bool is_contiguouts = IsContiguous(new_shape, new_strides);
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, old_storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, is_contiguouts);
  return {new_storage_info};
}

REG_VIEW_STRIDES_CALC_FUN(Transpose, TransposeCalc);
}  // namespace mindspore::ops
