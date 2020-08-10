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

#include "src/runtime/kernel/arm/fp32_grad/convolution_grad_input.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/nnacl/pack.h"
#include "src/runtime/kernel/arm/nnacl/fp32_grad/pack_ext.h"
#include "src/runtime/kernel/arm/nnacl/fp32_grad/gemm.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DGradInput;

namespace mindspore::kernel {
int ConvolutionGradInputCPUKernel::Init() {
  if (2 != this->inputs_.size()) {
    MS_LOG(ERROR) << "Conv2d Grad should has 2 inputs";
    return RET_ERROR;
  }
  if (1 != this->outputs_.size()) {
    MS_LOG(ERROR) << "Conv2d Grad should has one output";
    return RET_ERROR;
  }

  auto *dy_tensor = inputs_.at(kInputIndex);
  MS_ASSERT(dy_tensor != nullptr);
  auto *weight_tensor = inputs_.at(kWeightIndex);
  MS_ASSERT(weight_tensor != nullptr);
  auto *dx_tensor = outputs_.at(kOutputIndex);
  MS_ASSERT(dx_tensor != nullptr);

  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  conv_param->output_batch_ = dx_tensor->shape()[(kNHWC_N)];
  conv_param->input_batch_ = dy_tensor->shape()[(kNHWC_N)];

  conv_param->input_h_ = dx_tensor->shape()[(kNHWC_H)];
  conv_param->input_w_ = dx_tensor->shape()[(kNHWC_W)];

  // assume OutCh|kh|kw|In
  conv_param->input_channel_ = dx_tensor->shape()[(kNHWC_C)];
  conv_param->output_channel_ = weight_tensor->shape()[(kNHWC_N)];

  // TBD
  conv_param->output_h_ = dy_tensor->shape()[kNHWC_H];
  conv_param->output_w_ = dy_tensor->shape()[kNHWC_W];

  int ws_size = conv_param->output_h_ * conv_param->output_w_ * conv_param->kernel_h_ * conv_param->kernel_w_ *
                conv_param->input_channel_ / conv_param->group_;

  workspace = new float[ws_size];
  return 0;
}

int ConvolutionGradInputCPUKernel::ReSize() { return 0; }

int ConvolutionGradInputCPUKernel::Run() {
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  auto *input_dy = inputs_.at(0);
  auto *input_w = inputs_.at(1);
  auto *out_dx = outputs_.at(0);

  auto dy_addr = reinterpret_cast<float *>(input_dy->Data());
  auto w_addr = reinterpret_cast<float *>(input_w->Data());
  auto dx_addr = reinterpret_cast<float *>(out_dx->Data());

  int i, j;
  int nweights = input_w->ElementsNum();
  int in_ch = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int k_h = conv_param->kernel_h_;  // out_dw->shape()[1];
  int k_w = conv_param->kernel_w_;  // out_dw->shape()[2];
  int batch = conv_param->output_batch_;
  int out_ch = conv_param->output_channel_;
  int groups = conv_param->group_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;

  int m = out_h * out_w;
  int n = k_w * k_h * in_ch / groups;
  int k = out_ch / groups;

  memset(dx_addr, 0, sizeof(float) * batch * in_ch * in_h * in_w);

  for (i = 0; i < batch; ++i) {
    for (j = 0; j < groups; ++j) {
      float *mat_a = dy_addr + (i * groups) * m * k + j * (out_ch / groups);
      float *mat_b = w_addr + j * nweights / groups;
      float *mat_c = workspace;
      gemm(0, 0, m, n, k, 1, mat_a, out_ch, mat_b, n, 0, mat_c, n);
      col2im_hwc(mat_c, dx_addr + (i * groups) * (in_ch / groups) * in_h * in_w + j * (in_ch / groups), conv_param);
    }
  }

  // std::cout << "run succ" << std::endl;
  return 0;
}

kernel::LiteKernel *CpuConvGradInputFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                      const std::vector<lite::tensor::Tensor *> &outputs,
                                                      OpParameter *opParameter, const lite::Context *ctx,
                                                      const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DGradInput);

  auto *kernel = new (std::nothrow) ConvolutionGradInputCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  MS_ASSERT(kernel != nullptr);

  auto ret = kernel->Init();
  if (0 != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2DGradInput, CpuConvGradInputFp32KernelCreator)
}  // namespace mindspore::kernel
