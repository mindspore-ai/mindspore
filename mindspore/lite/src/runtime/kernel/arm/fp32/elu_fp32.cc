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

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Elu;

namespace mindspore::kernel {
int EluCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
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
  auto error_code = Elu(input_addr, output_addr, elu_parameter_, task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "EluCPUKernel DoExcute error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int EluRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto EluData = reinterpret_cast<EluCPUKernel *>(cdata);
  auto ret = EluData->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "EluRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int EluCPUKernel::Run() {
  auto ret = ParallelLaunch(this->ms_context_, EluRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Elu error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Elu, LiteKernelCreator<EluCPUKernel>)
}  // namespace mindspore::kernel
