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

#include "src/litert/kernel/cpu/fp32_grad/deconvolution_grad_filter.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/pack.h"
#include "nnacl/fp32_grad/pack_ext.h"
#include "nnacl/fp32_grad/gemm.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DeConv2DGradFilter;

namespace mindspore::kernel {
int DeConvolutionGradFilterCPUKernel::Prepare() {
  // dy is in input 0
  // x is in input 1
  // dw is output 0
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_2D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  auto *x_tensor = in_tensors_.at(1);
  auto *dy_tensor = in_tensors_.at(0);

  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  CHECK_NULL_RETURN(conv_param);
  MS_CHECK_TRUE_RET(dy_tensor->shape().size() == COMM_SHAPE_SIZE, RET_ERROR);
  MS_CHECK_TRUE_RET(x_tensor->shape().size() == COMM_SHAPE_SIZE, RET_ERROR);
  MS_CHECK_GT(conv_param->group_, 0, RET_ERROR);
  conv_param->output_batch_ = dy_tensor->shape().at(kNHWC_N);
  conv_param->input_batch_ = x_tensor->shape().at(kNHWC_N);
  conv_param->input_h_ = x_tensor->shape().at(kNHWC_H);
  conv_param->input_w_ = x_tensor->shape().at(kNHWC_W);
  // assume OutCh|kh|kw|InCh
  conv_param->input_channel_ = x_tensor->shape().at(kNHWC_C);
  conv_param->output_channel_ = dy_tensor->shape().at(kNHWC_C);

  conv_param->output_h_ = dy_tensor->shape()[kNHWC_H];
  conv_param->output_w_ = dy_tensor->shape()[kNHWC_W];

  ws_size = chunk * conv_param->input_w_ * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->output_channel_ /
            conv_param->group_;

  int m = conv_param->input_channel_ / conv_param->group_;
  int n = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->output_channel_ / conv_param->group_;
  size_t mat_alloc = MatSizeTotal(n, m, chunk * conv_param->input_w_, conv_param->input_channel_);

  set_workspace_size((ws_size + mat_alloc) * sizeof(float));

  return RET_OK;
}

int DeConvolutionGradFilterCPUKernel::ReSize() { return RET_OK; }

int DeConvolutionGradFilterCPUKernel::DoExecute(int task_id) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto *input_dy = in_tensors_.at(0);
  auto *input_x = in_tensors_.at(1);
  auto *out_dw = out_tensors_.at(0);

  auto x_addr = reinterpret_cast<float *>(input_x->data());
  CHECK_NULL_RETURN(x_addr);
  auto dy_addr = reinterpret_cast<float *>(input_dy->data());
  CHECK_NULL_RETURN(dy_addr);
  auto dw_addr = reinterpret_cast<float *>(out_dw->data());
  CHECK_NULL_RETURN(dw_addr);

  int i;
  int j;
  int out_w = conv_param->output_w_;
  int out_h = conv_param->output_h_;
  int out_ch = conv_param->output_channel_;
  int k_w = conv_param->kernel_w_;
  int k_h = conv_param->kernel_h_;
  int in_w = conv_param->input_w_;
  int in_h = conv_param->input_h_;
  int in_ch = conv_param->input_channel_;
  int groups = conv_param->group_;
  int batch = conv_param->output_batch_;

  MS_CHECK_GT(groups, 0, RET_ERROR);
  const int m = in_ch / groups;
  const int n = k_h * k_w * out_ch / groups;

  float *workspace_temp = reinterpret_cast<float *>(workspace());
  CHECK_NULL_RETURN(workspace_temp);
  float *mat_workspace = workspace_temp + ws_size;
  // zero out pointer
  memset(dw_addr, 0, out_dw->Size());
  for (i = 0; i < batch; ++i) {
    for (j = 0; j < groups; ++j) {
      for (int ci = 0; ci < in_h; ci += chunk) {
        int real_chunk = MSMIN(in_h - ci, chunk);
        float *mat_a = x_addr + (i * (in_ch * in_h * in_w) + j * (in_ch / groups)) + ci * in_w * in_ch;
        float *mat_b = workspace_temp;
        float *mat_c = dw_addr + j * m;
        float *im = dy_addr + (i * (out_h * out_w * out_ch) + j * (out_ch / groups));
        rolling_im2row_hwc(im, mat_b, conv_param, real_chunk, ci);
        GemmMatmul(0, 0, n, m, real_chunk * in_w, 1, mat_b, real_chunk * in_w, mat_a, in_ch, 1, mat_c, in_ch,
                   mat_workspace);
      }
    }
  }
  return RET_OK;
}

int DeConvolutionGradFilterRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto convfilter_kernel = reinterpret_cast<DeConvolutionGradFilterCPUKernel *>(cdata);
  auto error_code = convfilter_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvolutionGradFilterRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvolutionGradFilterCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, DeConvolutionGradFilterRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv filter function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuDeConvGradFilterFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                         const std::vector<lite::Tensor *> &outputs,
                                                         OpParameter *opParameter, const lite::InnerContext *ctx,
                                                         const kernel::KernelKey &desc) {
  MS_ASSERT(desc.type == schema::PrimitiveType_DeConv2DGradFilter);

  auto *kernel = new (std::nothrow) DeConvolutionGradFilterCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new kernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DeConv2DGradFilter, CpuDeConvGradFilterFp32KernelCreator)
}  // namespace mindspore::kernel
