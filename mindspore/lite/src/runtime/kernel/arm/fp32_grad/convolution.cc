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
#include "src/runtime/runtime_api.h"
#include "nnacl/pack.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ConvolutionTrainCPUKernel::ReSize() {
  if (in_tensors_.size() < 2) {
    MS_LOG(ERROR) << "Convolution should have at least two inputs";
    return RET_ERROR;
  }
  if (out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "Convolution should have one output";
    return RET_ERROR;
  }
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

  conv_param_->group_ = (conv_param_->group_ == 0) ? conv_param_->input_channel_ : conv_param_->group_;
  const int n = conv_param_->output_channel_ * conv_param_->group_;
  const int k = conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ / conv_param_->group_;

  do_img2col_ = (conv_param_->kernel_h_ == 1) && (conv_param_->kernel_w_ == 1) && (conv_param_->pad_d_ == 0) &&
                    (conv_param_->pad_u_ == 0) && (conv_param_->pad_l_ == 0) && (conv_param_->pad_r_ == 0) &&
                    (conv_param_->dilation_h_ == 1) && (conv_param_->dilation_w_ == 1) &&
                    (conv_param_->stride_h_ == 1) && (conv_param_->stride_w_ == 1) && (conv_param_->group_ == 1)
                  ? false
                  : true;
  do_dw_ = (conv_param_->output_channel_ == conv_param_->group_) &&
               (conv_param_->input_channel_ == conv_param_->output_channel_) && (conv_param_->dilation_h_ == 1) &&
               (conv_param_->dilation_w_ == 1)
             ? true
             : false;

  ws_size_ = chunk_ * conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_;
  ws_size_ = do_dw_ ? ws_size_ : ws_size_ / conv_param_->group_;
  int mat_alloc = MatSizeTotal(chunk_, n, k, 0);
  set_workspace_size((ws_size_ + mat_alloc) * sizeof(float));

  return RET_OK;
}

int ConvolutionTrainCPUKernel::Init() { return ReSize(); }

int ConvolutionTrainCPUKernel::Execute(int task_id) {
  auto conv_param_ = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto *input_x = in_tensors_.at(kInputIndex);
  auto *input_w = in_tensors_.at(kWeightIndex);
  auto *out_y = out_tensors_.at(kOutputIndex);

  auto x_addr = reinterpret_cast<float *>(input_x->MutableData());
  auto y_addr = reinterpret_cast<float *>(out_y->MutableData());
  auto w_addr = reinterpret_cast<float *>(input_w->MutableData());

  const int nweights = input_w->ElementsNum();
  const int in_ch = conv_param_->input_channel_;
  const int in_h = conv_param_->input_h_;
  const int in_w = conv_param_->input_w_;
  const int k_h = conv_param_->kernel_h_;
  const int k_w = conv_param_->kernel_w_;
  const int batch = conv_param_->output_batch_;
  const int out_ch = conv_param_->output_channel_;  // out_y->shape()[3];
  const int groups = conv_param_->group_;
  const int out_h = conv_param_->output_h_;
  const int out_w = conv_param_->output_w_;
  const int m = out_h * out_w;
  const int n = out_ch / groups;
  const int k = k_h * k_w * in_ch / groups;
  float *workspace_temp = static_cast<float *>(workspace());
  float *mat_workspace = workspace_temp + ws_size_;

  if (do_dw_) {
    const int kernel_spatial = k_h * k_w;
    for (int i = 0; i < batch; ++i) {
      for (int ci = 0; ci < m; ci += chunk_) {
        int real_chunk = MSMIN(m - ci, chunk_);
        float *mat_a = workspace_temp;
        float *im = x_addr + (i * in_ch * in_h * in_w);
        RollingIm2ColPackDwUnitFp32(im, conv_param_, mat_a, real_chunk, ci);
        for (int j = 0; j < groups; ++j) {
          const float *mat_b = w_addr + j * nweights / groups;
          float *mat_c = y_addr + (i * groups) * n * m + j * (out_ch / groups) + ci * out_ch;
          // float *im = x_addr + i * in_ch * in_h * in_w + j * (in_ch / groups);
          // RollingIm2ColPackUnitFp32(im, conv_param_, mat_a, real_chunk, ci);
          GemmMatmul(0, 1, real_chunk, n, k, 1, mat_a + (j * kernel_spatial), k * groups, mat_b, k, 0, mat_c, out_ch,
                     mat_workspace);
        }
      }
    }
  } else if (do_img2col_) {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < groups; ++j) {
        for (int ci = 0; ci < m; ci += chunk_) {
          int real_chunk = MSMIN(m - ci, chunk_);
          float *mat_a = workspace_temp;
          const float *mat_b = w_addr + j * nweights / groups;
          float *mat_c = y_addr + (i * groups) * n * m + j * (out_ch / groups) + ci * out_ch;
          float *im = x_addr + i * in_ch * in_h * in_w + j * (in_ch / groups);
          RollingIm2ColPackUnitFp32(im, conv_param_, mat_a, real_chunk, ci);
          GemmMatmul(0, 1, real_chunk, n, k, 1, mat_a, k, mat_b, k, 0, mat_c, out_ch, mat_workspace);
        }
      }
    }
  } else {
    const float *mat_b = w_addr;
    const size_t in_plane_size = in_ch * in_h * in_w;
    for (int i = 0; i < batch; ++i) {
      float *im = x_addr + i * in_plane_size;
      for (int ci = 0; ci < m; ci += chunk_) {
        int real_chunk = MSMIN(m - ci, chunk_);
        float *mat_c = y_addr + i * n * m + ci * out_ch;
        int input_height = ci / out_w * conv_param_->stride_h_;
        int input_width = ci % out_w * conv_param_->stride_w_;
        int offset = (input_height * in_w + input_width) * in_ch;
        GemmMatmul(0, 1, real_chunk, n, k, 1, im + offset, k, mat_b, k, 0, mat_c, out_ch, mat_workspace);
      }
    }
  }
  return RET_OK;
}

int ConvolutionTrainRun(void *cdata, int task_id) {
  MS_ASSERT(cdata != nullptr);
  auto conv_kernel = reinterpret_cast<ConvolutionTrainCPUKernel *>(cdata);
  auto error_code = conv_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionTrainRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionTrainCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, ConvolutionTrainRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv train function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuConvTrainFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                  const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DFusion);

  auto *kernel = new (std::nothrow) ConvolutionTrainCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ConvolutionTrainCPUKernel failed!";
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
}  // namespace mindspore::kernel
