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

#include "src/runtime/kernel/arm/fp32/elu_fp32.h"
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Elu;

namespace mindspore::kernel {
int EluCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int EluCPUKernel::ReSize() {
  elu_parameter_->in_size_ = in_tensors_.front()->ElementsNum();
  return RET_OK;
}

int EluCPUKernel::DoExcute(int task_id) {
  auto input_addr = reinterpret_cast<float *>(in_tensors_.front()->MutableData());
  auto output_addr = reinterpret_cast<float *>(out_tensors_.front()->MutableData());
  Elu(input_addr, output_addr, elu_parameter_, task_id);
  return RET_OK;
}

int EluRun(void *cdata, int task_id) {
  auto EluData = reinterpret_cast<EluCPUKernel *>(cdata);
  auto ret = EluData->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "EluRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int EluCPUKernel::Run() {
  auto ret = ParallelLaunch(this->context_->thread_pool_, EluRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Elu error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Elu, LiteKernelCreator<EluCPUKernel>)
}  // namespace mindspore::kernel
