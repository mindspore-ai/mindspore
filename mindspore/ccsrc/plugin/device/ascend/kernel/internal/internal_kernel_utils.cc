
/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include <string>

#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"

namespace mindspore {
namespace kernel {
internal::TensorFormat InternalKernelUtils::ToInternalFormat(Format format) {
  switch (format) {
    case FRACTAL_NZ:
      return internal::TensorFormat::TENSOR_FORMAT_FRACTAL_NZ;
    default:
      // some op not support NCHW, NHWC, ... format, current return ND format
      return internal::TensorFormat::TENSOR_FORMAT_ND;
  }
}

int InternalKernelUtils::ToInternalOpId(std::string name) {
  if (ms_op_key_to_internel_op_id.find(name) != ms_op_key_to_internel_op_id.end()) {
    return ms_op_key_to_internel_op_id[name];
  }
  return -1;
}

internal::TensorDType InternalKernelUtils::ToInternalDType(TypeId type) {
  switch (type) {
    // float data type
    case kNumberTypeFloat16:
      return internal::TensorDType::TENSOR_DTYPE_FLOAT16;
    case kNumberTypeBFloat16:
      return internal::TensorDType::TENSOR_DTYPE_BF16;
    case kNumberTypeFloat32:
      return internal::TensorDType::TENSOR_DTYPE_FLOAT;
    case kNumberTypeDouble:
      return internal::TensorDType::TENSOR_DTYPE_DOUBLE;

    // int data type
    case kNumberTypeInt32:
      return internal::TensorDType::TENSOR_DTYPE_INT32;
    case kNumberTypeUInt32:
      return internal::TensorDType::TENSOR_DTYPE_UINT32;
    case kNumberTypeInt16:
      return internal::TensorDType::TENSOR_DTYPE_INT16;
    case kNumberTypeUInt16:
      return internal::TensorDType::TENSOR_DTYPE_UINT16;
    case kNumberTypeInt8:
      return internal::TensorDType::TENSOR_DTYPE_INT8;
    case kNumberTypeUInt8:
      return internal::TensorDType::TENSOR_DTYPE_INT8;
    case kNumberTypeInt64:
      return internal::TensorDType::TENSOR_DTYPE_INT64;
    case kNumberTypeUInt64:
      return internal::TensorDType::TENSOR_DTYPE_UINT64;

    // complex data type
    case kNumberTypeComplex64:
      return internal::TensorDType::TENSOR_DTYPE_COMPLEX64;
    case kNumberTypeComplex128:
      return internal::TensorDType::TENSOR_DTYPE_COMPLEX128;

    // other data type
    case kNumberTypeBool:
      return internal::TensorDType::TENSOR_DTYPE_BOOL;
    case kObjectTypeString:
      return internal::TensorDType::TENSOR_DTYPE_STRING;
    default:
      return internal::TensorDType::TENSOR_DTYPE_UNDEFINED;
  }
}

void InternalKernelUtils::ToInternalTensor(internal::Tensor *internal_tensor, const KernelTensor *kernel_tensor) {
  internal_tensor->desc.format = ToInternalFormat(kernel_tensor->format());
  internal_tensor->desc.dtype = ToInternalDType(kernel_tensor->dtype_id());
  if (kernel_tensor->GetShapeVector().size() == kDim0) {
    internal_tensor->desc.dims = {kDim1};
  } else {
    internal_tensor->desc.dims = internal::VecToSVec<int64_t>(kernel_tensor->GetShapeVector());
  }

  internal_tensor->data = kernel_tensor->device_ptr();
}

internal::DeviceRawBuf InternalKernelUtils::ToDeviceRawBuf(const KernelTensor *kernel_tensor) {
  return internal::DeviceRawBuf{kernel_tensor->size(), kernel_tensor->device_ptr()};
}
}  // namespace kernel
}  // namespace mindspore
