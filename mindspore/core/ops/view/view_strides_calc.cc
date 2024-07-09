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

#include "ops/view/view_strides_calc.h"
#include <vector>
#include <memory>

namespace mindspore::ops {
constexpr size_t kViewInputsNum = 2;
ShapeVector update_shape(const ShapeVector &input_shape, ShapeVector shape) {
  int64_t x_num = 1;
  for (int64_t value : input_shape) {
    x_num = LongMulWithOverflowCheck(value, x_num);
  }

  auto it_first = find(shape.begin(), shape.end(), -1);
  if (it_first != shape.end()) {
    auto it_second = find(it_first + 1, shape.end(), -1);
    if (it_second != shape.end()) {
      MS_EXCEPTION(ValueError) << "At most one component of input shape can be -1, but got " << shape;
    }
    auto index = LongToSize(std::distance(shape.begin(), it_first));
    int64_t infer_value = x_num;
    for (size_t i = 0; i < shape.size(); ++i) {
      int64_t value = shape[i];
      if (value != -1 && value != 0) {
        infer_value = infer_value / value;
      }
    }
    shape[index] = infer_value;
  }

  int64_t shape_num = 1;
  for (int64_t value : shape) {
    shape_num = LongMulWithOverflowCheck(value, shape_num);
  }
  if (shape_num != x_num) {
    MS_EXCEPTION(ValueError) << "The accumulate of x_shape must be equal to out_shape, but got x_shape: " << input_shape
                             << ", and out_shape: " << shape;
  }
  return shape;
}

TensorStorageInfoPtrList ViewCalcImpl(const PrimitivePtr &prim, const tensor::BaseTensorPtr &input_tensor,
                                      const std::vector<int64_t> &shape) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  const auto &old_shape = old_tensor_info->old_shape;
  auto storage_offset = old_tensor_info->old_offset;

  const auto &new_shape = update_shape(old_shape, shape);
  const auto &new_strides = GetOriStrides(new_shape);
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}

TensorStorageInfoPtrList ViewCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (inputs.size() != kViewInputsNum) {
    return {};
  }
  auto input_tensor = inputs[0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto ori_storage_info = input_tensor->storage_info();
  if (ori_storage_info != nullptr && !ori_storage_info->is_contiguous) {
    MS_LOG(EXCEPTION) << "input tensor:" << input_tensor->ToString()
                      << " is not contiguous, storage info:" << ori_storage_info->ToString();
  }

  auto shape = GetValue<std::vector<int64_t>>(inputs[1]);
  if (std::any_of(shape.begin(), shape.end(), [](const int &shape_i) { return shape_i < -1; })) {
    MS_EXCEPTION(ValueError) << "For primitive[" << prim->name()
                             << "], the component of shape can't be less than -1, but got " << shape;
  }

  return ViewCalcImpl(prim, input_tensor, shape);
}
}  // namespace mindspore::ops
