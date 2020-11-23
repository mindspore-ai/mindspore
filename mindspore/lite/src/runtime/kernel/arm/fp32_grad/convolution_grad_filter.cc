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
#include "src/runtime/runtime_api.h"

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

  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  conv_param->output_batch_ = dy_tensor->shape().at(kNHWC_N);
  conv_param->input_batch_ = x_tensor->shape().at(kNHWC_N);
  conv_param->input_h_ = x_tensor->shape().at(kNHWC_H);
  conv_param->input_w_ = x_tensor->shape().at(kNHWC_W);
  // assume OutCh|kh|kw|InCh
  conv_param->input_channel_ = x_tensor->shape().at(kNHWC_C);
  conv_param->output_channel_ = dy_tensor->shape().at(kNHWC_C);

  conv_param->output_h_ = dy_tensor->shape()[kNHWC_H];
  conv_param->output_w_ = dy_tensor->shape()[kNHWC_W];

  ws_size = chunk * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_ / conv_param->group_;

  int n = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_ / conv_param->group_;
  int k = conv_param->output_channel_ / conv_param->group_;
  size_t mat_alloc = MatSizeTotal(k, n, chunk, n);
  set_workspace_size((ws_size + mat_alloc) * sizeof(float));
  return RET_OK;
}

int ConvolutionGradFilterCPUKernel::ReSize() { return RET_OK; }

int ConvolutionGradFilterCPUKernel::Execute(int task_id) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto *input_dy = in_tensors_.at(0);
  auto *input_x = in_tensors_.at(1);
  auto *out_dw = out_tensors_.at(0);

  auto x_addr = reinterpret_cast<float *>(input_x->MutableData());
  auto dy_addr = reinterpret_cast<float *>(input_dy->MutableData());
  auto dw_addr = reinterpret_cast<float *>(out_dw->MutableData());

  int i, j;
  int nweights = out_dw->ElementsNum();
  int in_ch = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int batch = conv_param->output_batch_;
  int out_ch = conv_param->output_channel_;
  int groups = conv_param->group_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;

  int m = out_h * out_w;
  int n = k_h * k_w * in_ch / groups;
  int k = out_ch / groups;

  float *workspace_temp = reinterpret_cast<float *>(workspace());
  float *mat_workspace = workspace_temp + ws_size;
  // zero out pointer
  memset(dw_addr, 0, out_dw->Size());
  for (i = 0; i < batch; ++i) {
    for (j = 0; j < groups; ++j) {
      for (int ci = 0; ci < m; ci += chunk) {
        int real_chunk = MSMIN(m - ci, chunk);
        float *mat_a = dy_addr + (i * groups) * m * k + j * (out_ch / groups) + ci * out_ch;
        float *mat_b = workspace_temp;
        float *mat_c = dw_addr + j * nweights / groups;
        float *im = x_addr + (i * in_ch * in_h * in_w) + j * (in_ch / groups);
        memset(mat_b, 0, n * real_chunk * sizeof(float));
        RollingIm2ColPackUnitFp32(im, conv_param, mat_b, real_chunk, ci);
        GemmMatmul(1, 0, k, n, real_chunk, 1, mat_a, out_ch, mat_b, n, 1, mat_c, n, mat_workspace);
      }
    }
  }
  return RET_OK;
}

int ConvolutionGradFilterRun(void *cdata, int task_id) {
  MS_ASSERT(cdata != nullptr);
  auto convfilter_kernel = reinterpret_cast<ConvolutionGradFilterCPUKernel *>(cdata);
  auto error_code = convfilter_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionGradFilterRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionGradFilterCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, ConvolutionGradFilterRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv filter function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuConvGradFilterFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                       const std::vector<lite::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::InnerContext *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DGradFilter);

  auto *kernel = new (std::nothrow) ConvolutionGradFilterCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new kernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2DGradFilter, CpuConvGradFilterFp32KernelCreator)
}  // namespace mindspore::kernel
