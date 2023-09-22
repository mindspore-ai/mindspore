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
#include "ops/view/view_strides_calculator.h"

namespace mindspore::ops {
ViewStridesCalcFactory &ViewStridesCalcFactory::GetInstance() {
  static ViewStridesCalcFactory instance;
  return instance;
}

bool IsDynamic(const std::vector<int64_t> &shape) {
  return std::any_of(shape.begin(), shape.end(), [](int64_t value) { return value < 0; });
}

bool HasZero(const std::vector<int64_t> &value) {
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == 0) {
      return true;
    }
  }
  return false;
}

bool CheckInputsNull(const std::vector<ValuePtr> &inputs, const size_t &input_num) {
  if (inputs.size() != input_num) {
    MS_LOG(DEBUG) << "inputs.size() is not equal to input_num, inputs.size():" << inputs.size()
                  << " input_num:" << input_num;
    return true;
  }

  return std::any_of(inputs.cbegin(), inputs.cend(), [](const ValuePtr &v) { return v == nullptr; });
}

std::vector<int64_t> GetOriStrides(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<int64_t> ret{1};
  int64_t strides = 1;
  for (int64_t i = shape.size() - 1; i > 0; i--) {
    strides *= shape[i];
    (void)ret.emplace(ret.begin(), strides);
  }
  return ret;
}

bool IsContiguous(const ShapeVector &shape, const std::vector<int64_t> &strides) {
  if (shape.size() == 0) {
    return true;
  }
  if (shape.size() != strides.size()) {
    MS_LOG(EXCEPTION) << "shape.size() != strides.size()";
  }
  int64_t expected_strides = 1;
  for (int64_t i = strides.size() - 1; i >= 0; i--) {
    if (expected_strides != strides[i]) {
      return false;
    }
    expected_strides *= shape[i];
  }
  return true;
}

int64_t DynamicDimWrap(int64_t dim, int64_t dim_post_expr) {
  if (dim_post_expr * -1 <= dim && dim < dim_post_expr) {
    if (dim < 0) {
      return dim + dim_post_expr;
    }
    return dim;
  }
  MS_EXCEPTION(ValueError) << "dim:" << dim << " dim_post_expr:" << dim_post_expr;
}

OldTensorInfoPtr GetOldTensorInfo(const tensor::TensorPtr &tensor) {
  OldTensorInfo old_tensor_info;
  if (tensor->storage_info() == nullptr) {
    auto shape = tensor->shape();
    auto strides = GetOriStrides(shape);
    old_tensor_info.old_shape = shape;
    old_tensor_info.ori_shape = shape;
    old_tensor_info.ori_strides = strides;
    old_tensor_info.old_strides = strides;
    old_tensor_info.old_offset = 0;
  } else {
    auto storage_info = tensor->storage_info();
    old_tensor_info.old_shape = storage_info->shape;
    old_tensor_info.old_strides = storage_info->strides;
    old_tensor_info.ori_shape = storage_info->ori_shape;
    old_tensor_info.ori_strides = storage_info->ori_strides;
    old_tensor_info.old_offset = storage_info->storage_offset;
  }
  return std::make_shared<OldTensorInfo>(old_tensor_info);
}
}  // namespace mindspore::ops
