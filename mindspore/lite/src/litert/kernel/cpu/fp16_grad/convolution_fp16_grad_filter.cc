/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp16_grad/convolution_fp16_grad_filter.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/pack.h"
#include "nnacl/fp16_grad/convolution_grad_filter.h"
#include "nnacl/fp16_grad/pack_fp16_ext.h"
#include "nnacl/fp16_grad/gemm_fp16.h"
#include "nnacl/errorcode.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DBackpropFilterFusion;

namespace mindspore::kernel {
int ConvolutionGradFilterCPUKernelFp16::ReSize() {
  // dy is in input 0
  // x is in input 1
  // dw is output 0
  CHECK_LESS_RETURN(in_tensors_.size(), THIRD_INPUT);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  auto *x_tensor = in_tensors_.at(SECOND_INPUT);
  CHECK_NULL_RETURN(x_tensor);
  auto *dy_tensor = in_tensors_.at(FIRST_INPUT);
  CHECK_NULL_RETURN(dy_tensor);
  CHECK_NULL_RETURN(op_parameter_);
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

  NNACL_CHECK_ZERO_RETURN_ERR(conv_param->group_);
  do_img2col_ = (conv_param->kernel_h_ == 1) && (conv_param->kernel_w_ == 1) && (conv_param->pad_d_ == 0) &&
                    (conv_param->pad_u_ == 0) && (conv_param->pad_l_ == 0) && (conv_param->pad_r_ == 0) &&
                    (conv_param->dilation_h_ == 1) && (conv_param->dilation_w_ == 1) && (conv_param->stride_h_ == 1) &&
                    (conv_param->stride_w_ == 1) && (conv_param->group_ == 1)
                  ? false
                  : true;
  do_dw_ = (conv_param->output_channel_ == conv_param->group_) &&
               (conv_param->input_channel_ == conv_param->output_channel_) && (conv_param->dilation_h_ == 1) &&
               (conv_param->dilation_w_ == 1)
             ? true
             : false;

  ws_size_ = chunk_ * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  ws_size_ = do_dw_ ? ws_size_ : ws_size_ / conv_param->group_;
  int n = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_ / conv_param->group_;
  int k = conv_param->output_channel_ / conv_param->group_;
  int thread_num = ms_context_->thread_num_;
  mat_alloc_ = MatSizeTotalFp16(k, n, chunk_, 0);
  set_workspace_size((ws_size_ + mat_alloc_ + (k * n)) * thread_num * sizeof(float16_t));

  return RET_OK;
}

int ConvolutionGradFilterCPUKernelFp16::Prepare() { return ReSize(); }

int ConvolutionGradFilterCPUKernelFp16::DoExecute(int task_id) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  CHECK_NULL_RETURN(conv_param);
  auto *input_dy = in_tensors_.at(FIRST_INPUT);
  CHECK_NULL_RETURN(input_dy);
  auto *input_x = in_tensors_.at(SECOND_INPUT);
  CHECK_NULL_RETURN(input_x);
  auto *out_dw = out_tensors_.at(FIRST_INPUT);
  CHECK_NULL_RETURN(out_dw);
  auto x_addr = reinterpret_cast<float16_t *>(input_x->data());
  CHECK_NULL_RETURN(x_addr);
  auto dy_addr = reinterpret_cast<float16_t *>(input_dy->data());
  CHECK_NULL_RETURN(dy_addr);
  auto dw_addr = reinterpret_cast<float16_t *>(out_dw->data());
  CHECK_NULL_RETURN(dw_addr);

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
  int thread_num = ms_context_->thread_num_;
  float16_t *workspace_temp = reinterpret_cast<float16_t *>(workspace());
  float16_t *mat_workspace = workspace_temp + ws_size_ * thread_num + task_id * (mat_alloc_ + k * n);
  float16_t *mat_tmp = mat_workspace + mat_alloc_;
  int stride = UP_DIV(batch, thread_num);
  int count = MSMIN(stride, batch - stride * task_id);
  count = (count < 0) ? 0 : count;
  int start = stride * task_id;
  int end = start + count;

  if (do_dw_) {
#ifdef ENABLE_ARM
    stride = UP_DIV(k_h * k_w, thread_num);
    count = MSMIN(stride, k_h * k_w - stride * task_id);
    count = (count < 0) ? 0 : count;
    start = stride * task_id;
    ConvDwFilterFp16Grad(x_addr, dy_addr, dw_addr, start, count, conv_param);
#else
    stride = UP_DIV(groups, thread_num);
    count = MSMIN(stride, groups - stride * task_id);
    start = stride * task_id;
    end = start + count;

    const int kernel_spatial = k_h * k_w;
    for (int i = 0; i < batch; ++i) {
      for (int ci = 0; ci < m; ci += chunk_) {
        int real_chunk = MSMIN(m - ci, chunk_);
        float16_t *mat_b = workspace_temp + task_id * ws_size_;
        float16_t *im = x_addr + (i * in_ch * in_h * in_w);
        RollingIm2ColPackDwUnitFp16(im, conv_param, mat_b, real_chunk, ci);
        for (int j = start; j < end; ++j) {
          float16_t *mat_a = dy_addr + (i * groups) * m * k + j * (out_ch / groups) + ci * out_ch;
          float16_t *mat_c = dw_addr + j * nweights / groups;
          GemmMatmulFp16(1, 0, k, n, real_chunk, 1, mat_a, out_ch, mat_b + (j * kernel_spatial), n * groups, 1, mat_c,
                         n, mat_workspace);
        }
      }
    }
#endif
  } else if (do_img2col_) {
    for (int i = start; i < end; ++i) {
      for (int ci = 0; ci < m; ci += chunk_) {
        for (int j = 0; j < groups; ++j) {
          int real_chunk = MSMIN(m - ci, chunk_);
          float16_t *mat_a = dy_addr + (i * groups) * m * k + j * (out_ch / groups) + ci * out_ch;
          float16_t *mat_b = workspace_temp + task_id * ws_size_;
          float16_t *mat_c = dw_addr + j * nweights / groups;
          float16_t *im = x_addr + (i * in_ch * in_h * in_w) + j * (in_ch / groups);
          RollingIm2ColPackUnitFp16(im, conv_param, mat_b, real_chunk, ci);
          GemmMatmulFp16(1, 0, k, n, real_chunk, 1, mat_a, out_ch, mat_b, n, 0, mat_tmp, n, mat_workspace);
          std::unique_lock<std::mutex> merge_lock(lock_);
          AddMatrixFp16(mat_tmp, mat_c, 1, k, n, n);
        }
      }
    }
  } else {
    NNACL_CHECK_ZERO_RETURN_ERR(out_w * conv_param->stride_h_);
    NNACL_CHECK_ZERO_RETURN_ERR(out_w * conv_param->stride_w_);
    float16_t *mat_c = dw_addr;
    const size_t in_plane_size = in_ch * in_h * in_w;
    for (int i = start; i < end; ++i) {
      for (int ci = 0; ci < m; ci += chunk_) {
        int real_chunk = MSMIN(m - ci, chunk_);
        float16_t *mat_a = dy_addr + i * m * k + ci * out_ch;
        float16_t *im = x_addr + i * in_plane_size;
        int input_h = ci / out_w * conv_param->stride_h_;
        int input_w = ci % out_w * conv_param->stride_w_;
        int offset = (input_h * in_w + input_w) * in_ch;
        GemmMatmulFp16(1, 0, k, n, real_chunk, 1, mat_a, out_ch, im + offset, n, 0, mat_tmp, n, mat_workspace);
        std::unique_lock<std::mutex> merge_lock(lock_);
        AddMatrixFp16(mat_tmp, mat_c, 1, k, n, n);
      }
    }
  }
  return RET_OK;
}

int ConvolutionGradFilterFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto convfilter_kernel = reinterpret_cast<ConvolutionGradFilterCPUKernelFp16 *>(cdata);
  CHECK_NULL_RETURN(convfilter_kernel);
  auto error_code = convfilter_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionGradFilterRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionGradFilterCPUKernelFp16::Run() {
  auto *out_dw = out_tensors_.at(0);
  auto dw_addr = reinterpret_cast<float16_t *>(out_dw->data());
  CHECK_NULL_RETURN(out_dw);
  CHECK_NULL_RETURN(dw_addr);
  memset(dw_addr, 0, out_dw->Size());
  int error_code = ParallelLaunch(this->ms_context_, ConvolutionGradFilterFp16Run, this, ms_context_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv filter function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Conv2DBackpropFilterFusion,
           LiteKernelCreator<ConvolutionGradFilterCPUKernelFp16>)
}  // namespace mindspore::kernel
