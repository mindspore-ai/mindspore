/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32/deconvolution.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DeConv2D;

namespace mindspore::kernel {
DeConvolutionCPUKernel::~DeConvolutionCPUKernel() {
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
  if (tmp_output_ != nullptr) {
    free(tmp_output_);
    tmp_output_ = nullptr;
  }

  if (tmp_buffer_ != nullptr) {
    free(tmp_buffer_);
    tmp_buffer_ = nullptr;
  }
  if (c4_input_ != nullptr) {
    free(c4_input_);
    c4_input_ = nullptr;
  }
  if (c4_output_ != nullptr) {
    free(c4_output_);
    c4_output_ = nullptr;
  }
  return;
}

int DeConvolutionCPUKernel::ReSize() { return 0; }

int DeConvolutionCPUKernel::InitWeightBias() {
  if (inputs_.size() == 3) {
    bias_data_ = malloc(UP_ROUND(conv_param_->output_channel_, C4NUM) * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "deconv malloc bias_data_ error!";
      return RET_ERROR;
    }
    memset(bias_data_, 0, UP_ROUND(conv_param_->output_channel_, C4NUM) * sizeof(float));
    memcpy(bias_data_, inputs_[2]->Data(), conv_param_->output_channel_ * sizeof(float));
  } else {
    bias_data_ = nullptr;
  }

  size_t weight_pack_size = conv_param_->kernel_w_ * conv_param_->kernel_h_ *
                            UP_ROUND(conv_param_->output_channel_, C4NUM) *
                            UP_ROUND(conv_param_->input_channel_, C4NUM) * sizeof(float);
  weight_ptr_ = reinterpret_cast<float *>(malloc(weight_pack_size));
  if (weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "deconv malloc weight_ptr_ error!";
    return RET_ERROR;
  }
  memset(weight_ptr_, 0, weight_pack_size);
  PackDeConvWeightFp32(reinterpret_cast<float *>(inputs_[1]->Data()), weight_ptr_, conv_param_->input_channel_,
                       conv_param_->output_channel_, conv_param_->kernel_w_ * conv_param_->kernel_h_);
  return RET_OK;
}

int DeConvolutionCPUKernel::InitParam() {
  matmul_param_ = new StrassenMatMulParameter();
  matmul_param_->row_ = conv_param_->input_h_ * conv_param_->input_w_;
  matmul_param_->deep_ = UP_DIV(conv_param_->input_channel_, C4NUM);
  matmul_param_->col_ = UP_DIV(conv_param_->output_channel_, 4) * conv_param_->kernel_w_ * conv_param_->kernel_h_;
  matmul_param_->a_stride_ = matmul_param_->row_ * C4NUM;
  matmul_param_->b_stride_ = matmul_param_->deep_ * C4NUM * C4NUM;
  matmul_param_->c_stride_ = matmul_param_->row_ * C4NUM;

  thread_hw_count_ = MSMIN(opParameter->thread_num_, matmul_param_->row_);
  thread_hw_stride_ = UP_DIV(matmul_param_->row_, thread_hw_count_);

  thread_co4_count_ = MSMIN(opParameter->thread_num_, UP_DIV(conv_param_->output_channel_, C4NUM));
  thread_co_stride_ = UP_DIV(UP_DIV(conv_param_->output_channel_, C4NUM), thread_co4_count_) * C4NUM;

  tmp_buffer_ =
    reinterpret_cast<float *>(malloc(matmul_param_->a_stride_ * matmul_param_->deep_ * C4NUM * sizeof(float)));
  if (tmp_buffer_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc tmp_buffer_ error!";
    return RET_ERROR;
  }

  tmp_output_ = reinterpret_cast<float *>(malloc(matmul_param_->row_ * matmul_param_->col_ * C4NUM * sizeof(float)));
  if (tmp_output_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc tmp_output_ error!";
    return RET_ERROR;
  }

  c4_input_ =
    reinterpret_cast<float *>(malloc(inputs_[0]->ElementsC4Num() / conv_param_->input_batch_ * sizeof(float)));
  if (c4_input_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc c4_input_ error!";
    return RET_NULL_PTR;
  }

  c4_output_ =
    reinterpret_cast<float *>(malloc(outputs_[0]->ElementsC4Num() / conv_param_->output_batch_ * sizeof(float)));
  if (c4_output_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc c4_output_ error!";
    return RET_NULL_PTR;
  }
  return RET_OK;
}

int DeConvFp32Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto deconv = reinterpret_cast<DeConvolutionCPUKernel *>(cdata);
  auto error_code = deconv->DoDeconv(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvFp32Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
int DeConvFp32PostRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto deconv = reinterpret_cast<DeConvolutionCPUKernel *>(cdata);
  auto error_code = deconv->DoPostFunc(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvFp32PostRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
int DeConvolutionCPUKernel::DoDeconv(int task_id) {
  matmul_param_->row_ = MSMIN(thread_hw_stride_, matmul_param_->row_ - task_id * thread_hw_stride_);
  if (matmul_param_->row_ <= 0) {
    return RET_OK;
  }

  int error_code = DeConvFp32(c4_input_ + task_id * thread_hw_stride_ * C4NUM, weight_ptr_,
                              tmp_output_ + task_id * thread_hw_stride_ * C4NUM,
                              tmp_buffer_ + task_id * thread_hw_stride_ * matmul_param_->deep_ * C4NUM, *matmul_param_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvFp32 error! error code: " << error_code;
    return error_code;
  }

  matmul_param_->row_ = conv_param_->input_h_ * conv_param_->input_w_;
  return RET_OK;
}

int DeConvolutionCPUKernel::DoPostFunc(int task_id) {
  int input_plane = conv_param_->input_h_ * conv_param_->input_w_;
  int kernel_plane = conv_param_->kernel_w_ * conv_param_->kernel_h_;
  int output_plane = conv_param_->output_h_ * conv_param_->output_w_;

  int cur_oc = MSMIN(thread_co_stride_, conv_param_->output_channel_ - task_id * thread_co_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  float *cur_bias =
    (bias_data_ == nullptr) ? nullptr : reinterpret_cast<float *>(bias_data_) + thread_co_stride_ * task_id;

  DeConvPostFp32(tmp_output_ + thread_co_stride_ * task_id * input_plane * kernel_plane,
                 c4_output_ + thread_co_stride_ * task_id * output_plane, output_ptr_ + thread_co_stride_ * task_id,
                 cur_bias, cur_oc, input_plane, kernel_plane, output_plane, conv_param_);
  return RET_OK;
}

int DeConvolutionCPUKernel::Init() {
  int error_code = ConvolutionBaseCPUKernel::Init();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Conv base init error!";
    return error_code;
  }

  error_code = InitParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv InitParam error!";
    return error_code;
  }

  error_code = InitWeightBias();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv InitWeightBias error!";
    return error_code;
  }
  return RET_OK;
}

int DeConvolutionCPUKernel::Run() {
  float *src_in = reinterpret_cast<float *>(inputs_[0]->Data());
  float *src_out = reinterpret_cast<float *>(outputs_[0]->Data());

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    input_ptr_ = src_in + batch_index * conv_param_->input_w_ * conv_param_->input_h_ * conv_param_->input_channel_;
    output_ptr_ =
      src_out + batch_index * conv_param_->output_h_ * conv_param_->output_w_ * conv_param_->output_channel_;

    PackNHWCToNC4HW4Fp32(input_ptr_, c4_input_, 1, conv_param_->input_h_ * conv_param_->input_w_,
                         conv_param_->input_channel_);

    int error_code = LiteBackendParallelLaunch(DeConvFp32Run, this, thread_hw_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv fp32 run error! error_code[" << error_code << "]";
      return RET_ERROR;
    }

    error_code = LiteBackendParallelLaunch(DeConvFp32PostRun, this, thread_co4_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv fp32 postrun error! error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
