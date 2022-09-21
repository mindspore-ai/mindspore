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

#include "src/litert/kernel/cpu/fp32_grad/convolution_grad_filter.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/pack.h"
#include "nnacl/fp32_grad/convolution_grad_filter.h"
#include "nnacl/fp32_grad/pack_ext.h"
#include "nnacl/fp32_grad/gemm.h"
#include "nnacl/errorcode.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DBackpropFilterFusion;

namespace mindspore::kernel {
constexpr static int kDyIdx = 0;
constexpr static int kXIdx = 1;
constexpr static int kDwIdx = 0;

int ConvolutionGradFilterCPUKernel::ReSize() {
  // dy is in input 0
  // x is in input 1
  // dw is output 0

  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_2D);
  CHECK_LESS_RETURN(out_tensors_.size(), DIMENSION_1D);
  CHECK_NULL_RETURN(out_tensors_.at(kDwIdx));
  auto *x_tensor = in_tensors_.at(kXIdx);
  CHECK_NULL_RETURN(x_tensor);
  auto *dy_tensor = in_tensors_.at(kDyIdx);
  CHECK_NULL_RETURN(dy_tensor);
  CHECK_NULL_RETURN(op_parameter_);
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  CHECK_NULL_RETURN(conv_param);
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
  do_img2col_ = !((conv_param->kernel_h_ == 1) && (conv_param->kernel_w_ == 1) && (conv_param->pad_d_ == 0) &&
                  (conv_param->pad_u_ == 0) && (conv_param->pad_l_ == 0) && (conv_param->pad_r_ == 0) &&
                  (conv_param->dilation_h_ == 1) && (conv_param->dilation_w_ == 1) && (conv_param->stride_h_ == 1) &&
                  (conv_param->stride_w_ == 1) && (conv_param->group_ == 1));
  do_dw_ = (conv_param->output_channel_ == conv_param->group_) &&
           (conv_param->input_channel_ == conv_param->output_channel_) && (conv_param->dilation_h_ == 1) &&
           (conv_param->dilation_w_ == 1);

  ws_size_ = chunk_ * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  ws_size_ = do_dw_ ? ws_size_ : ws_size_ / static_cast<size_t>(conv_param->group_);
  int n = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_ / conv_param->group_;
  int k = conv_param->output_channel_ / conv_param->group_;
  auto thread_num = static_cast<size_t>(op_parameter_->thread_num_);
  mat_alloc_ = MatSizeTotal(k, n, chunk_, 0);
  set_workspace_size((ws_size_ + mat_alloc_ + static_cast<size_t>(k * n)) * thread_num * sizeof(float));
  return RET_OK;
}

int ConvolutionGradFilterCPUKernel::Prepare() { return ReSize(); }

int ConvolutionGradFilterCPUKernel::DoExecute(int task_id) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  CHECK_NULL_RETURN(conv_param);
  auto *input_dy = in_tensors_.at(kDyIdx);
  auto *input_x = in_tensors_.at(kXIdx);
  auto *out_dw = out_tensors_.at(kDwIdx);
  CHECK_NULL_RETURN(out_dw);
  auto x_addr = reinterpret_cast<float *>(input_x->MutableData());
  auto dy_addr = reinterpret_cast<float *>(input_dy->MutableData());
  auto dw_addr = reinterpret_cast<float *>(out_dw->MutableData());

  int in_ch = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int out_ch = conv_param->output_channel_;
  int groups = conv_param->group_;

  int m = conv_param->output_h_ * conv_param->output_w_;
  int n = k_h * k_w * in_ch / groups;
  int k = out_ch / groups;
  int thread_num = op_parameter_->thread_num_;
  auto *workspace_temp = reinterpret_cast<float *>(workspace());
  float *mat_workspace =
    workspace_temp + static_cast<int>(ws_size_) * thread_num + task_id * (static_cast<int>(mat_alloc_) + k * n);
  float *mat_tmp = mat_workspace + mat_alloc_;
  int stride = UP_DIV(conv_param->output_batch_, thread_num);
  int count = MSMIN(stride, conv_param->output_batch_ - stride * task_id);
  count = (count < 0) ? 0 : count;
  int start = stride * task_id;
  int end = start + count;
  int real_chunk;
  float *mat_a = nullptr;
  float *mat_b = nullptr;
  float *mat_c = nullptr;
  float *im = nullptr;
  if (do_dw_) {
#ifdef ENABLE_ARM
    stride = UP_DIV(k_h * k_w, thread_num);
    count = MSMIN(stride, k_h * k_w - stride * task_id);
    count = (count < 0) ? 0 : count;
    start = stride * task_id;
    ConvDwFilterGrad(x_addr, dy_addr, dw_addr, start, count, conv_param);
#else
    stride = UP_DIV(groups, thread_num);
    count = MSMIN(stride, groups - stride * task_id);
    start = stride * task_id;
    end = start + count;

    const int kernel_spatial = k_h * k_w;
    for (int i = 0; i < conv_param->output_batch_; ++i) {
      for (int ci = 0; ci < m; ci += chunk_) {
        real_chunk = MSMIN(m - ci, chunk_);
        mat_b = workspace_temp + task_id * ws_size_;
        im = x_addr + (i * in_ch * in_h * in_w);
        RollingIm2ColPackDwUnitFp32(im, conv_param, mat_b, real_chunk, ci);
        for (int j = start; j < end; ++j) {
          mat_a = dy_addr + (i * groups) * m * k + j * (out_ch / groups) + ci * out_ch;
          mat_c = dw_addr + j * out_dw->ElementsNum() / groups;
          GemmMatmul(1, 0, k, n, real_chunk, 1, mat_a, out_ch, mat_b + (j * kernel_spatial), n * groups, 1, mat_c, n,
                     mat_workspace);
        }
      }
    }
#endif
  } else if (do_img2col_) {
    for (int i = start; i < end; ++i) {
      for (int ci = 0; ci < m; ci += chunk_) {
        for (int j = 0; j < groups; ++j) {
          real_chunk = MSMIN(m - ci, chunk_);
          mat_a = dy_addr + (i * groups) * m * k + j * (out_ch / groups) + ci * out_ch;
          mat_b = workspace_temp + task_id * static_cast<int>(ws_size_);
          mat_c = dw_addr + j * out_dw->ElementsNum() / groups;
          im = x_addr + (i * in_ch * in_h * in_w) + j * (in_ch / groups);
          RollingIm2ColPackUnitFp32(im, conv_param, mat_b, real_chunk, ci);
          GemmMatmul(1, 0, k, n, real_chunk, 1, mat_a, out_ch, mat_b, n, 0, mat_tmp, n, mat_workspace);
          std::unique_lock<std::mutex> merge_lock(lock_);
          AddMatrix(mat_tmp, mat_c, 1, k, n, n);
        }
      }
    }
  } else {
    NNACL_CHECK_ZERO_RETURN_ERR(conv_param->output_w_);
    mat_c = dw_addr;
    auto in_plane_size = in_ch * in_h * in_w;
    for (int i = start; i < end; ++i) {
      for (int ci = 0; ci < m; ci += chunk_) {
        real_chunk = MSMIN(m - ci, chunk_);
        mat_a = dy_addr + i * m * k + ci * out_ch;
        im = x_addr + i * in_plane_size;
        int input_h = ci / conv_param->output_w_ * conv_param->stride_h_;
        int input_w = ci % conv_param->output_w_ * conv_param->stride_w_;
        int offset = (input_h * in_w + input_w) * in_ch;
        GemmMatmul(1, 0, k, n, real_chunk, 1, mat_a, out_ch, im + offset, n, 0, mat_tmp, n, mat_workspace);
        std::unique_lock<std::mutex> merge_lock(lock_);
        AddMatrix(mat_tmp, mat_c, 1, k, n, n);
      }
    }
  }
  return RET_OK;
}

int ConvolutionGradFilterRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto convfilter_kernel = reinterpret_cast<ConvolutionGradFilterCPUKernel *>(cdata);
  CHECK_NULL_RETURN(convfilter_kernel);
  auto error_code = convfilter_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionGradFilterRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionGradFilterCPUKernel::Run() {
  auto *out_dw = out_tensors_.at(kDwIdx);
  CHECK_NULL_RETURN(out_dw);
  auto dw_addr = reinterpret_cast<float *>(out_dw->MutableData());
  CHECK_NULL_RETURN(dw_addr);
  memset(dw_addr, 0, out_dw->Size());
  int error_code = ParallelLaunch(this->ms_context_, ConvolutionGradFilterRun, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv filter function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuConvGradFilterFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                       const std::vector<lite::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::InnerContext *ctx,
                                                       const kernel::KernelKey &desc) {
  MS_CHECK_TRUE_MSG(opParameter != nullptr, nullptr, "Op parameter is nullptr.");
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DBackpropFilterFusion);

  auto *kernel = new (std::nothrow) ConvolutionGradFilterCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new kernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2DBackpropFilterFusion, CpuConvGradFilterFp32KernelCreator)
}  // namespace mindspore::kernel
