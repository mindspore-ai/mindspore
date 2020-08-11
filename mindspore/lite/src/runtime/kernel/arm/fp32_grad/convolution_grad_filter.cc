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

#include "src/runtime/kernel/arm/fp32_grad/convolution_grad_filter.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/nnacl/pack.h"
#include "src/runtime/kernel/arm/nnacl/fp32_grad/pack_ext.h"
#include "src/runtime/kernel/arm/nnacl/fp32_grad/gemm.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DGradFilter;

namespace mindspore::kernel {
int ConvolutionGradFilterCPUKernel::Init() {
  // dy is in input 0
  // x is in input 1
  // dw is output 0

  if (2 != this->inputs_.size()) {
    MS_LOG(ERROR) << "Conv2d Grad should has 2 inputs";
    return RET_ERROR;
  }
  if (1 != this->outputs_.size()) {
    MS_LOG(ERROR) << "Conv2d Grad should has one output";
    return RET_ERROR;
  }

  auto *input_tensor = inputs_.at(1);
  MS_ASSERT(input_tensor != nullptr);
  auto *dy = inputs_.at(0);
  MS_ASSERT(dy != nullptr);
  auto *weight_tensor = outputs_.at(0);
  MS_ASSERT(weight_tensor != nullptr);

  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  conv_param->output_batch_ = this->inputs_.at(0)->shape().at(kNHWC_N);
  conv_param->input_batch_ = this->inputs_.at(1)->shape().at(kNHWC_N);
  conv_param->input_h_ = this->inputs_.at(1)->shape().at(kNHWC_H);
  conv_param->input_w_ = this->inputs_.at(1)->shape().at(kNHWC_W);
  // assume OutCh|kh|kw|In
  conv_param->input_channel_ = this->inputs_.at(1)->shape().at(kNHWC_C);
  conv_param->output_channel_ = this->outputs_.at(0)->shape().at(kNHWC_N);

  int ws_size = conv_param->output_h_ * conv_param->output_w_ * conv_param->kernel_h_ * conv_param->kernel_w_ *
                conv_param->input_channel_ / conv_param->group_;

  workspace = new float[ws_size];

  int output_w = 0;
  int output_h = 0;
  output_h = dy->shape()[kNHWC_H];
  output_w = dy->shape()[kNHWC_W];

  std::vector<int> out_shape(4);
  out_shape.at(0) = conv_param->output_channel_;
  out_shape.at(1) = conv_param->kernel_h_;
  out_shape.at(2) = conv_param->kernel_w_;
  out_shape.at(3) = conv_param->input_channel_ / conv_param->group_;

  // weight is output
  weight_tensor->set_shape(out_shape);
  weight_tensor->set_data_type(input_tensor->data_type());

  conv_param->output_h_ = output_h;
  conv_param->output_w_ = output_w;

  return RET_OK;
}

int ConvolutionGradFilterCPUKernel::ReSize() { return 0; }

int ConvolutionGradFilterCPUKernel::Run() {
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  auto *input_dy = inputs_.at(0);
  auto *input_x = inputs_.at(1);
  auto *out_dw = outputs_.at(0);

  auto x_addr = reinterpret_cast<float *>(input_x->Data());
  auto dy_addr = reinterpret_cast<float *>(input_dy->Data());
  auto dw_addr = reinterpret_cast<float *>(out_dw->Data());

  int i, j;
  int nweights = out_dw->ElementsNum();
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
  int n = k_h * k_w * in_ch / groups;
  int k = out_ch / groups;

  // zero out pointer
  memset(dw_addr, 0, out_dw->Size());

  for (i = 0; i < batch; ++i) {
    for (j = 0; j < groups; ++j) {
      float *mat_a = dy_addr + (i * groups) * m * k + j * (out_ch / groups);
      float *mat_b = workspace;
      float *mat_c = dw_addr + j * nweights / groups;
      float *im = x_addr + (i * groups) * (in_ch / groups) * in_h * in_w + j * (in_ch / groups);

      im2row_hwc(im, mat_b, conv_param);
      gemm(1, 1, k, n, m, 1, mat_a, out_ch, mat_b, m, 1, mat_c, n);
    }
  }

  // std::cout << "run succ" << std::endl;
  return RET_OK;
}

kernel::LiteKernel *CpuConvGradFilterFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                       const std::vector<lite::tensor::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::Context *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DGradFilter);

  auto *kernel = new (std::nothrow) ConvolutionGradFilterCPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2DGradFilter, CpuConvGradFilterFp32KernelCreator)
}  // namespace mindspore::kernel
