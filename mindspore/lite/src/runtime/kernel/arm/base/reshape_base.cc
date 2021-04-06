/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/arm/base/reshape_base.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ExpandDims;
using mindspore::schema::PrimitiveType_Flatten;
using mindspore::schema::PrimitiveType_FlattenGrad;
using mindspore::schema::PrimitiveType_Reshape;
using mindspore::schema::PrimitiveType_Squeeze;
using mindspore::schema::PrimitiveType_Unsqueeze;

namespace mindspore::kernel {
int ReshapeBaseCPUKernel::Init() { return ReSize(); }

int ReshapeBaseCPUKernel::ReSize() {
  int in_data_size = in_tensors_.front()->Size();
  int thread_num = context_->thread_num_;
  cal_max_num_per_thread_ = UP_DIV(in_data_size, thread_num);
  return RET_OK;
}

int ReshapeBaseCPUKernel::RunImpl(int task_id) {
  size_t start_index = task_id * cal_max_num_per_thread_;
  if (start_index >= in_tensors_.front()->Size()) {
    return RET_OK;
  }
  auto cur_in_ptr = input_ptr_ + start_index;
  auto cur_out_ptr = output_ptr_ + start_index;

  size_t data_size = in_tensors_.front()->Size() - start_index;
  data_size = data_size > cal_max_num_per_thread_ ? cal_max_num_per_thread_ : data_size;
  memcpy(cur_out_ptr, cur_in_ptr, data_size);
  return RET_OK;
}

int ReshapeRun(void *cdata, int task_id) {
  auto reshape = reinterpret_cast<ReshapeBaseCPUKernel *>(cdata);
  auto ret = reshape->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ReshapeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ReshapeBaseCPUKernel::Run() {
  input_ptr_ = reinterpret_cast<uint8_t *>(in_tensors_.at(kInputIndex)->data_c());
  output_ptr_ = reinterpret_cast<uint8_t *>(out_tensors_.at(kOutputIndex)->data_c());
  auto ret = ParallelLaunch(this->context_->thread_pool_, ReshapeRun, this, context_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Reshape run error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Reshape, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Reshape, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Reshape, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Reshape, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Flatten, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Flatten, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FlattenGrad, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_ExpandDims, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ExpandDims, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ExpandDims, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_ExpandDims, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Squeeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Squeeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Squeeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Squeeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Unsqueeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Unsqueeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Unsqueeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt64, PrimitiveType_Unsqueeze, LiteKernelCreator<ReshapeBaseCPUKernel>)
}  // namespace mindspore::kernel
