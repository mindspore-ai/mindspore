/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_ACLNN_TENSOR_WRAPPER_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_ACLNN_TENSOR_WRAPPER_H_

#include <functional>
#include <numeric>
#include <optional>
#include <vector>
#include "kernel/ascend/acl_ir/op_api_cache.h"

namespace mindspore::device::ascend {
struct AclTensorWrapper {
  tensor::TensorPtr tensor;
  aclDataType dtype;
};

namespace acl_tensor_wrapper {
inline bool IsFourBitType(aclDataType dtype) { return dtype == ACL_FLOAT4_E2M1 || dtype == ACL_FLOAT4_E1M2; }

inline void ExpandFourBitShapeAndStrides(ShapeVector *shape, std::vector<int64_t> *strides) {
  MS_EXCEPTION_IF_NULL(shape);
  MS_EXCEPTION_IF_NULL(strides);
  const auto rank = shape->size();
  if (rank == 1) {
    (*shape)[0] *= 2;
    return;
  }
  if (rank < 2) {
    MS_LOG(EXCEPTION) << "A logical 4-bit ACL tensor must have at least one dimension.";
  }

  const auto last = rank - 1;
  const auto penultimate = rank - 2;
  if ((*strides)[last] == 1 && (*strides)[penultimate] == 1) {
    if ((*shape)[last] == 1) {
      (*strides)[penultimate] *= 2;
      (*shape)[last] *= 2;
    } else if ((*shape)[penultimate] == 1) {
      (*strides)[last] *= 2;
      (*shape)[penultimate] *= 2;
    }
  } else if ((*strides)[last] == 1) {
    (*strides)[penultimate] *= 2;
    (*shape)[last] *= 2;
  } else if ((*strides)[penultimate] == 1) {
    (*strides)[last] *= 2;
    (*shape)[penultimate] *= 2;
  }
  for (size_t i = 0; i < penultimate; ++i) {
    (*strides)[i] *= 2;
  }
}

inline int64_t TensorElementCount(const ShapeVector &shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
}
}  // namespace acl_tensor_wrapper

inline aclTensor *ConvertType(const AclTensorWrapper &wrapper) {
  MS_EXCEPTION_IF_NULL(wrapper.tensor);
  static const auto acl_create_tensor = GET_OP_API_FUNC(aclCreateTensor);
  if (acl_create_tensor == nullptr) {
    return nullptr;
  }

  auto shape = wrapper.tensor->shape();
  auto strides = wrapper.tensor->stride();
  auto format = ACL_FORMAT_ND;
  switch (shape.size()) {
    case 3:
      format = ACL_FORMAT_NCL;
      break;
    case 4:
      format = ACL_FORMAT_NCHW;
      break;
    case 5:
      format = ACL_FORMAT_NCDHW;
      break;
    default:
      break;
  }

  ShapeVector storage_shape;
  const auto &storage_info = wrapper.tensor->storage_info();
  if (storage_info != nullptr) {
    if (wrapper.tensor->format() == Format::FRACTAL_NZ) {
      format = ACL_FORMAT_FRACTAL_NZ;
      storage_shape = storage_info->ori_shape;
    } else {
      storage_shape = {acl_tensor_wrapper::TensorElementCount(storage_info->ori_shape)};
    }
  } else {
    storage_shape = {acl_tensor_wrapper::TensorElementCount(shape)};
  }

  if (acl_tensor_wrapper::IsFourBitType(wrapper.dtype)) {
    if (format == ACL_FORMAT_FRACTAL_NZ) {
      storage_shape.back() *= 2;
    } else {
      storage_shape[0] *= 2;
    }
    acl_tensor_wrapper::ExpandFourBitShapeAndStrides(&shape, &strides);
  }

  const auto &device_address = wrapper.tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  if (device_address->size() != 0 && device_address->GetMutablePtr() == nullptr) {
    MS_LOG(EXCEPTION) << "The device memory is null, please allocate the device memory for tensor "
                      << wrapper.tensor->ToString();
  }
  return acl_create_tensor(shape.data(), shape.size(), wrapper.dtype, strides.data(), wrapper.tensor->storage_offset(),
                           format, storage_shape.data(), storage_shape.size(), device_address->GetMutablePtr());
}

inline aclTensor *ConvertType(const std::optional<AclTensorWrapper> &wrapper) {
  if (!wrapper.has_value()) {
    return nullptr;
  }
  return ConvertType(wrapper.value());
}

inline std::vector<void *> GetAddr(const AclTensorWrapper &wrapper) { return GetAddr(wrapper.tensor); }

inline std::vector<void *> GetAddr(const std::optional<AclTensorWrapper> &wrapper) {
  if (!wrapper.has_value()) {
    return {nullptr};
  }
  return GetAddr(wrapper.value());
}

inline void GatherInfo(const AclTensorWrapper &wrapper) {
  GatherInfo(wrapper.tensor);
  MemcpyToBuf(&wrapper.dtype, sizeof(wrapper.dtype));
}

inline void GatherHash(const AclTensorWrapper &wrapper) {
  GatherHash(wrapper.tensor);
  GatherHash(wrapper.dtype);
}

inline void RefreshAddr(const AclTensorWrapper &wrapper) { RefreshAddr(wrapper.tensor); }

inline void RefreshAddr(const std::optional<AclTensorWrapper> &wrapper) {
  if (wrapper.has_value()) {
    RefreshAddr(wrapper.value());
  }
}
}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_ACLNN_TENSOR_WRAPPER_H_
