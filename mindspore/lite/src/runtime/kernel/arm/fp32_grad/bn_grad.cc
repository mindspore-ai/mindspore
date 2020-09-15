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
#include <algorithm>
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32_grad/batch_norm.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
// using mindspore::lite::REG_OP;
using mindspore::schema::PrimitiveType_BNGrad;


namespace mindspore::kernel {
int BNGradCPUKernel::Init() {
  auto *input_x = in_tensors_.at(1);
  int channels = input_x->shape().at(kNHWC_C);
  workspace_size = 4 * channels;
  workspace = new (std::nothrow) float[workspace_size];
  if (workspace == nullptr) {
    MS_LOG(ERROR) << "new workspace fail!";
    return RET_ERROR;
  }
  return RET_OK;
}

int BNGradCPUKernel::ReSize() { return RET_OK; }

int BNGradCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto bn_param = reinterpret_cast<BNGradParameter *>(op_parameter_);
  auto *input_yt = in_tensors_.at(0);
  auto *input_x = in_tensors_.at(1);
  auto *input_scale = in_tensors_.at(2);
  auto *output_dx = out_tensors_.at(0);
  auto *output_scale = out_tensors_.at(1);
  auto *output_bias = out_tensors_.at(2);
  // Tensor *bias = input[5];
  int batch = input_x->Batch();
  int channels = input_x->Channel();
  int spatial = input_x->Height() * input_x->Width();
  float eps = bn_param->epsilon_;
  std::fill(workspace, workspace + workspace_size, 0.f);
  float *mean = workspace;
  float *invar = mean + channels;
  float *dxhat_sum = invar + channels;
  float *dxhathat_sum = dxhat_sum + channels;

  float *x = reinterpret_cast<float *>(input_x->MutableData());
  float *yt = reinterpret_cast<float *>(input_yt->MutableData());
  float *scale = reinterpret_cast<float *>(input_scale->MutableData());
  float *dx = reinterpret_cast<float *>(output_dx->MutableData());
  float *dscale = reinterpret_cast<float *>(output_scale->MutableData());
  float *dbias = reinterpret_cast<float *>(output_bias->MutableData());

  backwardX(x, yt, scale, batch * spatial, channels, eps, mean, invar, dxhat_sum, dxhathat_sum, dx);
  // dbias
  sumSpatialBatch(yt, batch * spatial, channels, dbias);
  // dscale
  backwardScale(x, mean, invar, yt, batch, channels, spatial, dscale);
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
    return nullptr;
  }
  auto ret = kernel->Init();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BNGrad, CpuBNGradFp32KernelCreator)
}  // namespace mindspore::kernel
