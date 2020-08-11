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

#include "src/runtime/kernel/arm/fp32/elu.h"
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Elu;

namespace mindspore::kernel {
int EluCPUKernel::Init() {
  elu_parameter_ = reinterpret_cast<EluParameter *>(op_parameter_);
  elu_parameter_->thread_num_ = thread_count_;

  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int EluCPUKernel::ReSize() {
  elu_parameter_->in_size_ = in_tensors_.front()->ElementsNum();
  return RET_OK;
}

int EluCPUKernel::DoExcute(int task_id) { Elu(input_addr, output_addr, elu_parameter_, task_id); }

int EluRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto EluData = reinterpret_cast<EluCPUKernel *>(cdata);
  auto ret = EluData->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "EluRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int EluCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  input_addr = reinterpret_cast<float *>(in_tensors_.front()->Data());
  output_addr = reinterpret_cast<float *>(out_tensors_.front()->Data());

  auto ret = LiteBackendParallelLaunch(EluRun, this, elu_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Elu error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuEluFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                            const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *parameter,
                                            const lite::Context *ctx, const KernelKey &desc,
                                            const lite::Primitive *primitive) {
  if (parameter == nullptr || ctx == nullptr) {
    MS_LOG(ERROR) << "parameter or ctx is nullptr";
    return nullptr;
  }
  MS_ASSERT(desc.type == PrimitiveType_Elu);
  auto *kernel = new (std::nothrow) EluCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create Kernel failed, name: " << parameter->name_;
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init Kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Elu, CpuEluFp32KernelCreator)
}  // namespace mindspore::kernel
