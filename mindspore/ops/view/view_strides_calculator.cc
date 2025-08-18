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
#include "view/view_strides_calculator.h"
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <functional>
#include <numeric>

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

  std::vector<int64_t> ret(shape.size(), 1);
  int64_t strides = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides *= shape[i];
    ret[i - 1] = strides;
  }
  return ret;
}

bool IsContiguous(const ShapeVector &shape, const std::vector<int64_t> &strides) {
  if (shape.size() != strides.size()) {
    MS_LOG(EXCEPTION) << "shape.size() != strides.size()";
  }

  auto numel = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (numel == 0) {
    return true;
  }

  int64_t z = 1;
  for (int64_t i = SizeToLong(shape.size()) - 1; i >= 0; --i) {
    const auto &shape_i = shape[i];
    if (shape_i != 1) {
      if (strides[i] == z) {
        z *= shape_i;
      } else {
        return false;
      }
    }
  }

  return true;
}

int64_t ComputeStorageNelements(int64_t storage_offset, const std::vector<int64_t> &shape,
                                const std::vector<int64_t> &stride) {
  if (shape.size() != stride.size()) {
    MS_LOG(EXCEPTION) << "unequal shape length (" << shape.size() << ") and stride length (" << stride.size() << ")";
  }
  int64_t size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) {
      return 0;
    }
    if (shape[i] < 0 || stride[i] < 0) {
      MS_LOG(EXCEPTION) << "Storage size calculation overflowed with sizes=" << shape << " and strides=" << stride;
    }
    size += stride[i] * (shape[i] - 1);
  }
  return size + storage_offset;
}

TensorStorageInfoPtr CheckSetStorageInfo(const tensor::TensorPtr &origin_tensor, int64_t storage_offset,
                                         const std::vector<int64_t> &shape, const std::vector<int64_t> &stride,
                                         const std::string &source_device_type_name, int64_t source_storage_size,
                                         const TypeId &source_storage_dtype) {
  MS_EXCEPTION_IF_NULL(origin_tensor);
  MS_EXCEPTION_IF_NULL(origin_tensor->device_address());
  const auto &origin_device_address = origin_tensor->device_address();
  if (device::GetDeviceTypeByName(source_device_type_name) != origin_device_address->GetDeviceType()) {
    MS_LOG(EXCEPTION) << "Attempted to set the storage of a tensor on device \""
                      << device::GetDeviceNameByType(origin_device_address->GetDeviceType())
                      << "\" to a storage on different device \"" << source_device_type_name
                      << "\".  This is no longer allowed; the devices must match.";
  }
  constexpr size_t max_shape_dim = 8;
  if (shape.size() > max_shape_dim) {
    MS_LOG(EXCEPTION) << "The input shape's dim must in the range of [0, " << max_shape_dim << "], but got '"
                      << shape.size();
  }
  if (storage_offset < 0) {
    MS_LOG(EXCEPTION) << "Tensor: invalid storage offset " << storage_offset;
  }

  bool isContiguous = IsContiguous(shape, stride);

  const TensorStorageInfoPtr &origin_storage_info = origin_tensor->storage_info();
  std::vector<int64_t> ori_shape;
  std::vector<int64_t> ori_stride;
  if (origin_storage_info == nullptr) {
    ori_shape = origin_tensor->shape();
    ori_stride = GetOriStrides(ori_shape);
  } else {
    ori_shape = origin_storage_info->shape;
    ori_stride = origin_storage_info->strides;
  }

  bool shape_unchanged = shape == ori_shape ? true : false;
  bool stride_unchanged = stride == ori_stride ? true : false;
  int64_t storage_nelements = ComputeStorageNelements(storage_offset, shape, stride);
  int64_t origin_item_size = static_cast<int64_t>(GetTypeByte(TypeIdToType(origin_tensor->data_type())));
  int64_t storage_nelements_bytes = origin_item_size * storage_nelements;
  if (shape_unchanged && stride_unchanged) {
    if (storage_nelements_bytes > source_storage_size) {
      MS_LOG(EXCEPTION) << "setStorage: shape " << shape << ", strides " << stride << ", storage offset "
                        << storage_offset << ", and itemsize " << origin_item_size << " requiring a storage size of "
                        << storage_nelements_bytes << " are out of bounds for storage of size " << source_storage_size;
    }
  }

  int64_t source_item_size = static_cast<int64_t>(GetTypeByte(TypeIdToType(source_storage_dtype)));
  int64_t source_storage_nelements = static_cast<int64_t>(source_storage_size / source_item_size);
  if (storage_nelements_bytes <= source_storage_size) {
    ori_shape = {source_storage_nelements};
  } else {
    ori_shape = {storage_nelements};
  }
  ori_stride = {1};
  auto new_storage_info = std::make_shared<TensorStorageInfo>(
    std::move(shape), std::move(stride), storage_offset, std::move(ori_shape), std::move(ori_stride), isContiguous);
  return new_storage_info;
}

int64_t DynamicDimWrap(int64_t dim, int64_t dim_post_expr, bool wrap_scalar) {
  if (MS_LIKELY(dim_post_expr * -1 <= dim && dim < dim_post_expr)) {
    if (dim < 0) {
      return dim + dim_post_expr;
    }
    return dim;
  }
  if (dim_post_expr == 0) {
    if ((!wrap_scalar)) {
      MS_EXCEPTION(ValueError) << "dim value specified as " << dim << ", but tensor has no dimensions";
    }
    return DynamicDimWrap(dim, 1, false);
  }
  MS_EXCEPTION(ValueError) << "Dimension out of range (expected to be in range of [" << -dim_post_expr << ", "
                           << dim_post_expr << "), but got " << dim << ")";
}

OldTensorInfoPtr GetOldTensorInfo(const tensor::TensorPtr &tensor) {
  if (tensor->storage_info() == nullptr) {
    auto old_strides = GetOriStrides(tensor->shape());
    return std::make_shared<OldTensorInfo>(tensor->shape(), old_strides, tensor->shape(), old_strides, 0);
  } else {
    auto storage_info = tensor->storage_info();
    return std::make_shared<OldTensorInfo>(storage_info->shape, storage_info->strides, storage_info->ori_shape,
                                           storage_info->ori_strides, storage_info->storage_offset);
  }
}
}  // namespace mindspore::ops
