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
#include "src/litert/kernel/cpu/fp32/tril_fp32.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32/triu_tril_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/nnacl_common.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Tril;

namespace mindspore::kernel {
int TrilCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_NOT_EQUAL_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  return ReSize();
}

int TrilCPUKernel::ReSize() { return RET_OK; }

int TrilCPUKernel::GetKValue() {
  if (in_tensors_.size() <= 1) {
    k_ = 0;
    return lite::RET_OK;
  }
  auto k_tensor = in_tensors_.at(1);
  if (k_tensor == nullptr || k_tensor->data() == nullptr) {
    MS_LOG(ERROR) << "Failed to get value of k, input 1 cannot be nullptr";
    return RET_ERROR;
  }
  switch (k_tensor->data_type()) {
    case kNumberTypeInt:
    case kNumberTypeInt32:
      k_ = *(reinterpret_cast<int *>(k_tensor->data()));
      break;
    case kNumberTypeInt64:
      k_ = *(reinterpret_cast<int64_t *>(k_tensor->data()));
      break;
    default:
      MS_LOG(ERROR) << "Failed to get value of k, unsupported data type: " << k_tensor->data();
      return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int TrilCPUKernel::Run() {
  if (GetKValue() != RET_OK) {
    MS_LOG(ERROR) << "Failed to get k value from input 1, node " << op_parameter_->name_;
    return RET_ERROR;
  }
  auto input0 = in_tensors_[0];
  auto input_shape = input0->shape();
  if (std::any_of(input_shape.begin(), input_shape.end(), [](auto dim) { return dim <= 0; })) {
    MS_LOG(ERROR) << "Input shape cannot be dynamic, input shape: " << input_shape << ", node " << op_parameter_->name_;
    return RET_ERROR;
  }
  constexpr size_t input_hw_dims = 2;
  if (input_shape.size() < input_hw_dims) {
    MS_LOG(ERROR) << "Dims of input shape cannot < 2, input shape: " << input_shape << ", node "
                  << op_parameter_->name_;
    return RET_ERROR;
  }
  int64_t mul = 1;
  for (size_t i = 0; i < input_shape.size() - input_hw_dims; i++) {
    mul *= input_shape[i];
  }
  int64_t height = input_shape[input_shape.size() - 2];
  int64_t width = input_shape[input_shape.size() - 1];

  auto src_data = input0->data();
  auto dst_data = out_tensors_[0]->data();
  auto type_size = lite::DataTypeSize(input0->data_type());
  if (type_size == 0) {
    MS_LOG(ERROR) << "Unsupported data type: " << input0->data_type();
    return RET_ERROR;
  }
  switch (type_size) {
    case sizeof(int64_t): {
      TrilByte8(src_data, dst_data, k_, height, width, mul);
      break;
    }
    case sizeof(int32_t): {
      TrilByte4(src_data, dst_data, k_, height, width, mul);
      break;
    }
    case sizeof(int16_t): {
      TrilByte2(src_data, dst_data, k_, height, width, mul);
      break;
    }
    case sizeof(int8_t): {
      TrilByte1(src_data, dst_data, k_, height, width, mul);
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported data type: " << input0->data_type();
      return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat64, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt64, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt16, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeUInt64, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeUInt32, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeUInt16, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeUInt8, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Tril, LiteKernelCreator<TrilCPUKernel>)
}  // namespace mindspore::kernel
