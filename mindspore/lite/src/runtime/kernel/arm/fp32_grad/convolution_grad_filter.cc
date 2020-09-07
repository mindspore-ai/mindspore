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
#include "nnacl/pack.h"
#include "nnacl/fp32_grad/pack_ext.h"
#include "nnacl/fp32_grad/gemm.h"
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

  auto *x_tensor = in_tensors_.at(1);
  MS_ASSERT(x_tensor != nullptr);
  auto *dy_tensor = in_tensors_.at(0);
  MS_ASSERT(dy_tensor != nullptr);
#if 0
  auto *weight_tensor = out_tensors_.at(0);
  MS_ASSERT(weight_tensor != nullptr);
#endif
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  conv_param->output_batch_ = dy_tensor->shape().at(kNHWC_N);
  conv_param->input_batch_ = x_tensor->shape().at(kNHWC_N);
  conv_param->input_h_ = x_tensor->shape().at(kNHWC_H);
  conv_param->input_w_ = x_tensor->shape().at(kNHWC_W);
  // assume OutCh|kh|kw|InCh
  conv_param->input_channel_ = x_tensor->shape().at(kNHWC_C);
  conv_param->output_channel_ = dy_tensor->shape().at(kNHWC_C);
  // TBD
  conv_param->output_h_ = dy_tensor->shape()[kNHWC_H];
  conv_param->output_w_ = dy_tensor->shape()[kNHWC_W];

  int ws_size = conv_param->output_h_ * conv_param->output_w_ * conv_param->kernel_h_ * conv_param->kernel_w_ *
                conv_param->input_channel_ / conv_param->group_;

  workspace = new (std::nothrow) float[ws_size];
  if (workspace == nullptr) {
    MS_LOG(ERROR) << "new workspace fail!";
    return RET_ERROR;
  }

  return RET_OK;
}

int ConvolutionGradFilterCPUKernel::ReSize() { return RET_OK; }

int ConvolutionGradFilterCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto *input_dy = in_tensors_.at(0);
  auto *input_x = in_tensors_.at(1);
  auto *out_dw = out_tensors_.at(0);

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
#if 0
OpParameter *PopulateConvolutionGradFilterParameter(const lite::Primitive *primitive) {
  ConvParameter *param = new (std::nothrow) ConvParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "new Param for conv grad filter failed.";
    return nullptr;
  }
  param->op_parameter_.type_ = primitive->Type();

  auto convg_primitive = primitive->Value()->value_as_Conv2DGradFilter();
  param->kernel_h_ = convg_primitive->kernelH();
  param->kernel_w_ = convg_primitive->kernelW();
  param->stride_h_ = convg_primitive->strideH();
  param->stride_w_ = convg_primitive->strideW();
  param->dilation_h_ = convg_primitive->dilateH();
  param->dilation_w_ = convg_primitive->dilateW();
  param->pad_h_ = convg_primitive->padUp();
  param->pad_w_ = convg_primitive->padLeft();
  param->pad_u_ = convg_primitive->padUp();
  param->pad_d_ = convg_primitive->padDown();
  param->pad_l_ = convg_primitive->padLeft();
  param->pad_r_ = convg_primitive->padRight();
  param->group_ = convg_primitive->group();
  auto act_type = convg_primitive->activationType();
  switch (act_type) {
    case schema::ActivationType_RELU:
      param->is_relu_ = true;
      param->is_relu6_ = false;
      break;
    case schema::ActivationType_RELU6:
      param->is_relu_ = false;
      param->is_relu6_ = true;
      break;
    default:
      param->is_relu_ = false;
      param->is_relu6_ = false;
      break;
  }

  return reinterpret_cast<OpParameter *>(param);
}
#endif
kernel::LiteKernel *CpuConvGradFilterFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                       const std::vector<lite::tensor::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::Context *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DGradFilter);

  auto *kernel = new (std::nothrow) ConvolutionGradFilterCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new kernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2DGradFilter, CpuConvGradFilterFp32KernelCreator)
}  // namespace mindspore::kernel
