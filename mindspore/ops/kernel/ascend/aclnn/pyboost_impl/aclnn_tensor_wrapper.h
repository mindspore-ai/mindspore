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

#include <algorithm>
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

struct AclTensorListWrapper {
  std::vector<tensor::TensorPtr> tensors;
  aclDataType dtype;
};

namespace acl_tensor_wrapper {
constexpr size_t kFirstDimension = 0;
constexpr size_t kOneDimensionalRank = 1;
constexpr size_t kLastDimensionOffset = 1;
constexpr size_t kMinimumFourBitTensorRank = 2;
constexpr size_t kPenultimateDimensionOffset = 2;
constexpr size_t kNclRank = 3;
constexpr size_t kNchwRank = 4;
constexpr size_t kNcdhwRank = 5;
constexpr int64_t kUnitStride = 1;
constexpr int64_t kFourBitStorageExpansionFactor = 2;

inline bool IsFourBitType(aclDataType dtype) { return dtype == ACL_FLOAT4_E2M1 || dtype == ACL_FLOAT4_E1M2; }

inline void ExpandFourBitShapeAndStrides(ShapeVector *shape, std::vector<int64_t> *strides) {
  MS_EXCEPTION_IF_NULL(shape);
  MS_EXCEPTION_IF_NULL(strides);
  const auto rank = shape->size();
  if (rank == kOneDimensionalRank) {
    (*shape)[kFirstDimension] *= kFourBitStorageExpansionFactor;
    return;
  }
  if (rank < kMinimumFourBitTensorRank) {
    MS_LOG(EXCEPTION) << "A logical 4-bit ACL tensor must have at least one dimension.";
  }

  const auto last = rank - kLastDimensionOffset;
  const auto penultimate = rank - kPenultimateDimensionOffset;
  if ((*strides)[last] == kUnitStride && (*strides)[penultimate] == kUnitStride) {
    if ((*shape)[last] == kUnitStride) {
      (*strides)[penultimate] *= kFourBitStorageExpansionFactor;
      (*shape)[last] *= kFourBitStorageExpansionFactor;
    } else if ((*shape)[penultimate] == kUnitStride) {
      (*strides)[last] *= kFourBitStorageExpansionFactor;
      (*shape)[penultimate] *= kFourBitStorageExpansionFactor;
    }
  } else if ((*strides)[last] == kUnitStride) {
    (*strides)[penultimate] *= kFourBitStorageExpansionFactor;
    (*shape)[last] *= kFourBitStorageExpansionFactor;
  } else if ((*strides)[penultimate] == kUnitStride) {
    (*strides)[last] *= kFourBitStorageExpansionFactor;
    (*shape)[penultimate] *= kFourBitStorageExpansionFactor;
  }
  for (size_t i = 0; i < penultimate; ++i) {
    (*strides)[i] *= kFourBitStorageExpansionFactor;
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
    case acl_tensor_wrapper::kNclRank:
      format = ACL_FORMAT_NCL;
      break;
    case acl_tensor_wrapper::kNchwRank:
      format = ACL_FORMAT_NCHW;
      break;
    case acl_tensor_wrapper::kNcdhwRank:
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
      storage_shape.back() *= acl_tensor_wrapper::kFourBitStorageExpansionFactor;
    } else {
      storage_shape[acl_tensor_wrapper::kFirstDimension] *= acl_tensor_wrapper::kFourBitStorageExpansionFactor;
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

inline aclTensorList *ConvertType(const AclTensorListWrapper &wrapper) {
  static const auto acl_create_tensor_list = GET_OP_API_FUNC(aclCreateTensorList);
  if (acl_create_tensor_list == nullptr) {
    return nullptr;
  }
  std::vector<aclTensor *> acl_tensors;
  std::transform(
    wrapper.tensors.begin(), wrapper.tensors.end(), std::back_inserter(acl_tensors),
    [&wrapper](const tensor::TensorPtr &tensor) { return ConvertType(AclTensorWrapper{tensor, wrapper.dtype}); });
  return acl_create_tensor_list(acl_tensors.data(), acl_tensors.size());
}

inline std::vector<void *> GetAddr(const AclTensorWrapper &wrapper) { return GetAddr(wrapper.tensor); }

inline std::vector<void *> GetAddr(const std::optional<AclTensorWrapper> &wrapper) {
  if (!wrapper.has_value()) {
    return {nullptr};
  }
  return GetAddr(wrapper.value());
}

inline std::vector<void *> GetAddr(const AclTensorListWrapper &wrapper) { return GetAddr(wrapper.tensors); }

inline void GatherInfo(const AclTensorWrapper &wrapper) {
  GatherInfo(wrapper.tensor);
  MemcpyToBuf(&wrapper.dtype, sizeof(wrapper.dtype));
}

inline void GatherInfo(const AclTensorListWrapper &wrapper) {
  GatherInfo(wrapper.tensors);
  MemcpyToBuf(&wrapper.dtype, sizeof(wrapper.dtype));
}

inline void GatherHash(const AclTensorWrapper &wrapper) {
  GatherHash(wrapper.tensor);
  GatherHash(wrapper.dtype);
}

inline void GatherHash(const AclTensorListWrapper &wrapper) {
  GatherHash(wrapper.tensors);
  GatherHash(wrapper.dtype);
}

inline void RefreshAddr(const AclTensorWrapper &wrapper) { RefreshAddr(wrapper.tensor); }

inline void RefreshAddr(const std::optional<AclTensorWrapper> &wrapper) {
  if (wrapper.has_value()) {
    RefreshAddr(wrapper.value());
  }
}

inline void RefreshAddr(const AclTensorListWrapper &wrapper) {
  for (const auto &tensor : wrapper.tensors) {
    RefreshAddr(tensor);
  }
}
}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_ACLNN_TENSOR_WRAPPER_H_
