/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32_grad/bn_grad.h"
#include <math.h>
#include <algorithm>
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32_grad/batch_norm.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BNGrad;

namespace mindspore::kernel {
int BNGradCPUKernel::ReSize() {
  auto *input_x = in_tensors_.at(1);
  int channels = input_x->shape().at(kNHWC_C);
  set_workspace_size(2 * channels * sizeof(float));
  return RET_OK;
}

int BNGradCPUKernel::Init() { return ReSize(); }

int BNGradCPUKernel::Execute(int task_id) {
  auto *input_yt = in_tensors_.at(0);
  auto *input_x = in_tensors_.at(1);
  auto *input_scale = in_tensors_.at(2);
  auto *input_mean = in_tensors_.at(3);
  auto *input_var = in_tensors_.at(4);

  float *save_mean = reinterpret_cast<float *>(input_mean->MutableData());
  float *save_var = reinterpret_cast<float *>(input_var->MutableData());

  auto *output_dx = out_tensors_.at(0);
  auto *output_scale = out_tensors_.at(1);
  auto *output_bias = out_tensors_.at(2);
  int32_t batch = input_x->Batch();
  int32_t channels = input_x->Channel();
  int32_t spatial = input_x->Height() * input_x->Width();

  float *workspace_temp = static_cast<float *>(workspace());
  std::fill(workspace_temp, workspace_temp + workspace_size() / sizeof(*workspace_temp), 0.f);
  float *dxhat_sum = workspace_temp;
  float *dxhathat_sum = dxhat_sum + channels;

  float *x = reinterpret_cast<float *>(input_x->MutableData());
  float *yt = reinterpret_cast<float *>(input_yt->MutableData());
  float *scale = reinterpret_cast<float *>(input_scale->MutableData());
  float *dx = reinterpret_cast<float *>(output_dx->MutableData());
  float *dbias = reinterpret_cast<float *>(output_bias->MutableData());
  float *dscale = reinterpret_cast<float *>(output_scale->MutableData());
  std::fill(dbias, dbias + channels, 0.f);
  std::fill(dscale, dscale + channels, 0.f);
  backwardAll(x, yt, save_mean, save_var, scale, batch * spatial, channels, dxhat_sum, dxhathat_sum, dbias, dscale, dx);
  return RET_OK;
}

int BNGradRun(void *cdata, int task_id) {
  MS_ASSERT(cdata != nullptr);
  auto bn_kernel = reinterpret_cast<BNGradCPUKernel *>(cdata);

  auto error_code = bn_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BNGradRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int BNGradCPUKernel::Run() {
  auto *input_var = in_tensors_.at(4);
  float *save_var = reinterpret_cast<float *>(input_var->MutableData());
  auto bn_param = reinterpret_cast<BNGradParameter *>(op_parameter_);
  float eps = bn_param->epsilon_;
  var2Invar(save_var, input_var->ElementsNum(), eps);
  int error_code = ParallelLaunch(this->context_->thread_pool_, BNGradRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BN function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuBNGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_BNGrad);
  auto *kernel = new (std::nothrow) BNGradCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BNGradCPUKernel fail!";
    free(opParameter);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BNGrad, CpuBNGradFp32KernelCreator)
}  // namespace mindspore::kernel
