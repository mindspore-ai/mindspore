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
#include "src/runtime/kernel/arm/fp32/leaky_relu.h"
#include <vector>
#include "schema/model_generated.h"
#include "nnacl/fp32/leaky_relu.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LeakyReLU;

namespace mindspore::kernel {
namespace {
int LeakyReluRun(void *cdata, int task_id) {
  auto kernel_relu = reinterpret_cast<LeakyReluCPUKernel *>(cdata);
  auto ret = kernel_relu->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LeakyReluRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

LeakyReluCPUKernel::~LeakyReluCPUKernel() {
  if (prelu_param_->slope_ != nullptr) {
    free(prelu_param_->slope_);
    prelu_param_->slope_ = nullptr;
  }
}

int LeakyReluCPUKernel::Init() { return RET_OK; }

int LeakyReluCPUKernel::DoExcute(int task_id) {
  DoLeakyRelu(input_data, output_data, prelu_param_, task_id);
  return RET_OK;
}

int LeakyReluCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto input = in_tensors_.at(0);
  prelu_param_->input_num_ = input->ElementsNum();
  input_data = reinterpret_cast<float *>(input->Data());
  output_data = reinterpret_cast<float *>(out_tensors_.at(0)->Data());

  auto ret = ParallelLaunch(THREAD_POOL_DEFAULT, LeakyReluRun, this, context_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PReluDwRun error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuLeakyReluFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                  const std::vector<lite::tensor::Tensor *> &outputs,
                                                  OpParameter *param, const lite::Context *ctx,
                                                  const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "input param is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_LeakyRelu);
  auto *kernel = new (std::nothrow) LeakyReluCPUKernel(param, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new LeakyReluCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << param->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(param->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LeakyReLU, CpuLeakyReluFp32KernelCreator)
}  // namespace mindspore::kernel
