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

#include <algorithm>
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_factory.h"
#include "src/runtime/kernel/arm/fp32_grad/bn_grad.h"
#include "src/runtime/kernel/arm/nnacl/fp32_grad/batch_norm.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
// using mindspore::lite::REG_OP;
using mindspore::schema::PrimitiveType_BNGradInput;

namespace mindspore::kernel {
int BNGradInputCPUKernel::Init() {
  auto bn_param = reinterpret_cast<bnParameter *>(opParameter);
  workspace_size = 5 * bn_param->channels;
  workspace = new float[workspace_size];

  if (2 != this->inputs_.size()) {
    MS_LOG(ERROR) << "Conv2d Grad should has 2 inputs";
    return RET_ERROR;
  }
  if (1 != this->outputs_.size()) {
    MS_LOG(ERROR) << "Conv2d Grad should has one output";
    return RET_ERROR;
  }
  auto *input_tensor = inputs_.at(0);
  // auto *weight_tensor = inputs_.at(1);
  auto *out_tensor = outputs_.at(0);
  auto in_shape = input_tensor->shape();
  out_tensor->set_shape(in_shape);
  out_tensor->set_data_type(input_tensor->data_type());
  return RET_OK;
}

int BNGradInputCPUKernel::ReSize() { return RET_OK; }

int BNGradInputCPUKernel::Run() {
  // std::cout << "run succ" << std::endl;
  auto *input_x = inputs_.at(0);
  auto *input_yt = inputs_.at(1);
  auto *input_scale = inputs_.at(2);
  auto *output_grad = outputs_.at(0);
  // Tensor *bias = input[5];
  auto bn_param = reinterpret_cast<bnParameter *>(opParameter);
  int batch = bn_param->batch;
  int channels = bn_param->channels;
  int spatial = bn_param->spatial;
  float eps = bn_param->eps;
  std::fill(workspace, workspace + workspace_size, 0.f);

  float *mean = workspace;
  float *variance = mean + channels;
  float *mean_delta = variance + channels;
  float *variance_delta = mean_delta + channels;
  float *mean_add_delta = variance_delta + channels;

  float *x = reinterpret_cast<float *>(input_x->Data());
  float *yt = reinterpret_cast<float *>(input_yt->Data());
  float *scale = reinterpret_cast<float *>(input_scale->Data());
  float *out = reinterpret_cast<float *>(output_grad->Data());

  std::copy(yt, yt + batch * channels * spatial, out);
  meanVar(x, batch, spatial, channels, mean, variance);
  scaleBias(scale, batch, channels, spatial, out);
  meanDelta(out, spatial, channels, eps, variance, mean_delta);
  varianceDelta(x, out, mean, variance, batch, channels, spatial, eps, variance_delta);
  meanAdd(x, mean, variance_delta, batch, channels, spatial, mean_add_delta, mean_delta);
  NormalizeDelta(x, mean, variance, mean_delta, variance_delta, batch, channels, eps, spatial, out);
  return RET_OK;
}

kernel::LiteKernel *CpuBNGradInputFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                    const std::vector<lite::tensor::Tensor *> &outputs,
                                                    OpParameter *opParameter, const lite::Context *ctx,
                                                    const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_BNGradInput);
  //  parameter->name = opDef.name()->str().data();
  //  parameter->type = opDef.attr_type();
  auto *kernel = new (std::nothrow) BNGradInputCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  MS_ASSERT(kernel != nullptr);
  auto ret = kernel->Init();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BNGradInput, CpuBNGradInputFp32KernelCreator)
}  // namespace mindspore::kernel
