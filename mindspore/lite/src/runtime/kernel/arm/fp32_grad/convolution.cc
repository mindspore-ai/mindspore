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

#include "src/runtime/kernel/arm/fp32_grad/convolution.h"
#include "nnacl/fp32_grad/pack_ext.h"
#include "nnacl/fp32_grad/gemm.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ConvolutionTrainCPUKernel::Init() {
  auto conv_param_ = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto *input_x = in_tensors_.at(kInputIndex);
  auto *input_weight = in_tensors_.at(kWeightIndex);
  auto *out_y = out_tensors_.at(kOutputIndex);

  conv_param_->output_batch_ = out_y->shape().at(kNHWC_N);
  conv_param_->input_batch_ = input_x->shape().at(kNHWC_N);
  conv_param_->input_h_ = input_x->shape().at(kNHWC_H);
  conv_param_->input_w_ = input_x->shape().at(kNHWC_W);
  conv_param_->output_h_ = out_y->shape().at(kNHWC_H);
  conv_param_->output_w_ = out_y->shape().at(kNHWC_W);
  conv_param_->input_channel_ = input_x->shape().at(kNHWC_C);
  conv_param_->output_channel_ = input_weight->shape().at(kNHWC_N);
  conv_param_->kernel_h_ = input_weight->shape().at(kNHWC_H);
  conv_param_->kernel_w_ = input_weight->shape().at(kNHWC_W);

  int ws_size = conv_param_->output_h_ * conv_param_->output_w_ * conv_param_->kernel_h_ * conv_param_->kernel_w_ *
                conv_param_->input_channel_ / conv_param_->group_;

  workspace = new float[ws_size];
  return RET_OK;
}

int ConvolutionTrainCPUKernel::ReSize() { return RET_OK; }

int ConvolutionTrainCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto conv_param_ = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto *input_x = in_tensors_.at(kInputIndex);
  auto *input_w = in_tensors_.at(kWeightIndex);
  auto *out_y = out_tensors_.at(kOutputIndex);

  auto x_addr = reinterpret_cast<float *>(input_x->MutableData());
  auto y_addr = reinterpret_cast<float *>(out_y->MutableData());
  auto w_addr = reinterpret_cast<float *>(input_w->MutableData());

  int i, j;
  int nweights = input_w->ElementsNum();
  int in_ch = conv_param_->input_channel_;
  int in_h = conv_param_->input_h_;
  int in_w = conv_param_->input_w_;
  int k_h = conv_param_->kernel_h_;
  int k_w = conv_param_->kernel_w_;
  int batch = conv_param_->output_batch_;
  int out_ch = conv_param_->output_channel_;  // out_y->shape()[3];
  int groups = conv_param_->group_;
  int out_h = conv_param_->output_h_;
  int out_w = conv_param_->output_w_;
  int m = out_h * out_w;
  int n = out_ch / groups;
  int k = k_h * k_w * in_ch / groups;

  memset(y_addr, 0, out_y->Size());

  for (i = 0; i < batch; ++i) {
    for (j = 0; j < groups; ++j) {
      float *mat_a = workspace;
      float *mat_b = w_addr + j * nweights / groups;
      float *mat_c = y_addr + (i * groups) * n * m + j * (out_ch / groups);
      float *im = x_addr + (i * groups) * (in_ch / groups) * in_h * in_w + j * (in_ch / groups);
      im2col_hwc(im, mat_a, conv_param_);
      gemm(0, 1, m, n, k, 1, mat_a, k, mat_b, k, 1, mat_c, out_ch);
    }
  }

  // std::cout << "run succ" << std::endl;
  return RET_OK;
}

kernel::LiteKernel *CpuConvTrainFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                  const lite::Context *ctx, const kernel::KernelKey &desc,
                                                  const lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2D);

  auto *kernel = new (std::nothrow) ConvolutionTrainCPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

}  // namespace mindspore::kernel
