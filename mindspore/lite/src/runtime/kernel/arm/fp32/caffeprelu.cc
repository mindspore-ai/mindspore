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
#include "src/runtime/kernel/arm/fp32/caffeprelu.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/runtime/kernel/arm/nnacl/caffeprelu.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_CaffePReLU;

namespace mindspore::kernel {
int CaffePReluCPUKernel::Init() { return RET_OK; }

int CaffePReluCPUKernel::DoExcute(int task_id) {
  CaffePRelu(input_data, output_data, prelu_param_, task_id);
  return RET_OK;
}

int CaffePReluRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto PReludata = reinterpret_cast<CaffePReluCPUKernel *>(cdata);
  auto ret = PReludata->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PReluRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int CaffePReluCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto input = in_tensors_[0];
  auto input1 = in_tensors_[1];

  prelu_param_->input_num_ = input->ElementsNum();
  input_data = reinterpret_cast<float *>(input->Data());
  output_data = reinterpret_cast<float *>(out_tensors_[0]->Data());
  auto channels = input->shape();
  prelu_param_->negtive_slope_ = reinterpret_cast<float *>(input1->Data());
  prelu_param_->channel_num_ = channels.at(1);

  auto ret = LiteBackendParallelLaunch(CaffePReluRun, this, prelu_param_->op_parameter_.thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PReluDwRun error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuCaffePReluFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *opParameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Prelu);
  auto *kernel = new (std::nothrow) CaffePReluCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PReluCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_CaffePReLU, CpuCaffePReluFp32KernelCreator)
}  // namespace mindspore::kernel
