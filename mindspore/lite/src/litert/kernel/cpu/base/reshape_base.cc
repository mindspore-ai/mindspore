/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include <cstring>
#include "src/litert/kernel/cpu/base/reshape_base.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ExpandDims;
using mindspore::schema::PrimitiveType_Flatten;
using mindspore::schema::PrimitiveType_FlattenGrad;
using mindspore::schema::PrimitiveType_Reshape;
using mindspore::schema::PrimitiveType_Squeeze;
using mindspore::schema::PrimitiveType_Unsqueeze;

namespace mindspore::kernel {
constexpr int kMinCostPerThread = 16384;
int ReshapeBaseCPUKernel::Run() {
  /*
   * in_tensor : CPU-allocator ;  out_tensor : GPU-allocator
   * out_tensor data_c can not change
   * */
  auto in_tensor = in_tensors().front();
  CHECK_NULL_RETURN(in_tensor);
  auto out_tensor = out_tensors().front();
  CHECK_NULL_RETURN(out_tensor);
  auto in_shape = in_tensor->shape();
  // element number is 0, no need to copy data
  if (std::any_of(in_shape.begin(), in_shape.end(), [](auto dim) { return dim == 0; })) {
    return RET_OK;
  }

  if (in_tensor->data_type() != out_tensor->data_type() || in_tensor->data() == nullptr ||
      in_tensor->Size() != out_tensor->Size()) {
    MS_LOG(ERROR) << "Invalid param in reshape";
    return RET_ERROR;
  }

  if (in_tensor->allocator() == nullptr || in_tensor->allocator() != out_tensor->allocator() ||
      in_tensor->allocator() != ms_context_->allocator || /* runtime allocator */
      op_parameter_->is_train_session_ || out_tensor->IsGraphOutput()) {
    CHECK_NULL_RETURN(out_tensor->data());
    CHECK_NULL_RETURN(in_tensor->data());
    MS_CHECK_FALSE(in_tensor->Size() == 0, RET_ERROR);
    auto size = in_tensor->Size();
    thread_num_ = MSMIN(static_cast<size_t>(op_parameter_->thread_num_), UP_DIV(size, kMinCostPerThread));
    if (thread_num_ < 1) {
      thread_num_ = 1;
    }
    auto block_size = UP_DIV(size, thread_num_);
    thread_num_ = UP_DIV(size, block_size);
    auto in_data = static_cast<const uint8_t *>(in_tensor->data());
    auto out_data = static_cast<uint8_t *>(out_tensor->data());
    auto Copy = [in_data, out_data, size, block_size, this](void *, int task_id, float, float) {
      auto in_start = in_data + task_id * block_size;
      auto out_start = out_data + task_id * block_size;
      auto copy_size = block_size;
      if (task_id == (thread_num_ - 1)) {
        copy_size = size - task_id * block_size;
      }
      (void)memcpy(out_start, in_start, copy_size);
      return RET_OK;
    };
    if (in_data != out_data) {
      if (thread_num_ == 1) {
        (void)memcpy(out_data, in_data, size);
        return RET_OK;
      }
      return lite::ParallelLaunch(this->ms_context_, Copy, nullptr, thread_num_);
    }
    return RET_OK;
  }

  out_tensor->FreeData();
  out_tensor->ResetRefCount();
  out_tensor->set_data(in_tensor->data());
  if (in_tensor->IsConst()) {
    out_tensor->set_own_data(false);
  } else {
    out_tensor->set_own_data(in_tensor->own_data());
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Reshape, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Reshape, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Reshape, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Reshape, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Flatten, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Flatten, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Flatten, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_FlattenGrad, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FlattenGrad, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_ExpandDims, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ExpandDims, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ExpandDims, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_ExpandDims, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_ExpandDims, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Squeeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Squeeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Squeeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Squeeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Unsqueeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Unsqueeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Unsqueeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt64, PrimitiveType_Unsqueeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Unsqueeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
}  // namespace mindspore::kernel
