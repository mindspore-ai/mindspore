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

#include "src/runtime/kernel/arm/fp32/convolution_1x1.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
Convolution1x1CPUKernel::~Convolution1x1CPUKernel() {
  if (c4_output_ != nullptr) {
    free(c4_output_);
    c4_output_ = nullptr;
  }
  if (c4_input_ != nullptr) {
    free(c4_input_);
    c4_input_ = nullptr;
  }
  if (pre_trans_input_) {
    free(input_ptr_);
    input_ptr_ = nullptr;
  }
  if (tmp_ptr_ != nullptr) {
    free(tmp_ptr_);
    tmp_ptr_ = nullptr;
  }
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
  delete matmul_param_;
}

int Convolution1x1CPUKernel::ReSize() { return RET_OK; }

void Convolution1x1CPUKernel::InitConv1x1MatmulParam() {
  matmul_param_ = new StrassenMatMulParameter();
  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->col_ = UP_DIV(conv_param_->output_channel_, FP32_STRASSEN_UINT);
  matmul_param_->deep_ = UP_DIV(conv_param_->input_channel_, FP32_STRASSEN_UINT);
  matmul_param_->a_stride_ = matmul_param_->row_ * FP32_STRASSEN_UINT;
  matmul_param_->b_stride_ = matmul_param_->deep_ * FP32_STRASSEN_WEIGHT_UINT;
  matmul_param_->c_stride_ = matmul_param_->row_ * FP32_STRASSEN_UINT;
}

int Convolution1x1CPUKernel::InitConv1x1BiasWeight() {
  if (inputs_.size() == 3) {
    bias_data_ = malloc(matmul_param_->col_ * C4NUM * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc bias_ptr_ error!";
      return RET_ERROR;
    }
    memset(bias_data_, 0, matmul_param_->col_ * C4NUM * sizeof(float));
    memcpy(bias_data_, inputs_[2]->Data(), conv_param_->output_channel_ * sizeof(float));
  } else {
    bias_data_ = nullptr;
  }

  weight_ptr_ = reinterpret_cast<float *>(
    malloc(matmul_param_->col_ * matmul_param_->deep_ * FP32_STRASSEN_WEIGHT_UINT * sizeof(float)));
  if (weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc weight_ptr_ error!";
    return RET_ERROR;
  }
  memset(weight_ptr_, 0, matmul_param_->col_ * matmul_param_->deep_ * FP32_STRASSEN_WEIGHT_UINT * sizeof(float));
  Pack1x1WeightFp32(reinterpret_cast<float *>(inputs_[1]->Data()), weight_ptr_, conv_param_);
  return RET_OK;
}

int Convolution1x1CPUKernel::InitConv1x1Param() {
  pre_trans_input_ = (conv_param_->pad_h_ != 0 || conv_param_->pad_w_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);
  if (pre_trans_input_) {
    input_ptr_ = reinterpret_cast<float *>(malloc(matmul_param_->a_stride_ * matmul_param_->deep_ * sizeof(float)));
    if (input_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc input_ptr_ error!";
      return RET_MEMORY_FAILED;
    }
    memset(input_ptr_, 0, matmul_param_->a_stride_ * matmul_param_->deep_ * sizeof(float));
  }

  thread_hw_count_ = MSMIN(opParameter->thread_num_, matmul_param_->row_);
  thread_hw_stride_ = UP_DIV(matmul_param_->row_, thread_hw_count_);

  thread_oc4_count_ = MSMIN(opParameter->thread_num_, matmul_param_->col_);
  thread_oc_stride_ = UP_DIV(matmul_param_->col_, thread_oc4_count_) * C4NUM;

  tmp_ptr_ = reinterpret_cast<float *>(malloc(matmul_param_->a_stride_ * matmul_param_->deep_ * sizeof(float)));
  if (tmp_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc tmp_ptr_ error!";
    return RET_MEMORY_FAILED;
  }
  c4_output_ =
    reinterpret_cast<float *>(malloc(outputs_[0]->ElementsC4Num() / conv_param_->output_batch_ * sizeof(float)));
  if (c4_output_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc c4_output_ error!";
    return RET_MEMORY_FAILED;
  }

  c4_input_ =
    reinterpret_cast<float *>(malloc(inputs_[0]->ElementsC4Num() / conv_param_->input_batch_ * sizeof(float)));
  if (c4_input_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc c4_input_ error!";
    return RET_MEMORY_FAILED;
  }
  return RET_OK;
}

void Convolution1x1CPUKernel::Pre1x1Trans(float *src_input, float *src_output) {
  output_ptr_ = src_output;
  PackNHWCToNC4HW4Fp32(src_input, c4_input_, 1, conv_param_->input_h_ * conv_param_->input_w_,
                       conv_param_->input_channel_);

  if (!pre_trans_input_) {
    input_ptr_ = c4_input_;
    return;
  }

  Conv1x1InputPackFp32(c4_input_, input_ptr_, conv_param_);
  return;
}

int Convolution1x1CPUKernel::Init() {
  ConvolutionBaseCPUKernel::Init();
  InitConv1x1MatmulParam();

  int error_code = InitConv1x1BiasWeight();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution base init failed.";
    return error_code;
  }
  error_code = InitConv1x1Param();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution base init failed.";
    return error_code;
  }
  return RET_OK;
}

int Convolution1x1CPUKernel::DoStrassen(int task_id) {
  matmul_param_->row_ = MSMIN(thread_hw_stride_, matmul_param_->row_ - task_id * thread_hw_stride_);
  if (matmul_param_->row_ <= 0) {
    return RET_OK;
  }

  auto error_code = Conv1x1Fp32(input_ptr_ + task_id * thread_hw_stride_ * C4NUM, weight_ptr_,
                                c4_output_ + task_id * thread_hw_stride_ * C4NUM,
                                tmp_ptr_ + task_id * thread_hw_stride_ * matmul_param_->deep_ * C4NUM, *matmul_param_);
  if (error_code != 0) {
    MS_LOG(ERROR) << "DoStrassen error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  return RET_OK;
}

int Convolution1x1StrassenRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv1x1 = reinterpret_cast<Convolution1x1CPUKernel *>(cdata);
  auto error_code = conv1x1->DoStrassen(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1StrassenRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1CPUKernel::DoPostFunc(int task_id) {
  int cur_oc = MSMIN(thread_oc_stride_, conv_param_->output_channel_ - task_id * thread_oc_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  float *cur_bias =
    (bias_data_ == nullptr) ? nullptr : reinterpret_cast<float *>(bias_data_) + task_id * thread_oc_stride_;

  PostConvFuncFp32(c4_output_ + matmul_param_->row_ * thread_oc_stride_ * task_id,
                   output_ptr_ + task_id * thread_oc_stride_, cur_bias, cur_oc, matmul_param_->row_,
                   conv_param_->output_channel_, conv_param_->is_relu_, conv_param_->is_relu6_);
  return RET_OK;
}

int Convolution1x1PostFuncRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv1x1 = reinterpret_cast<Convolution1x1CPUKernel *>(cdata);
  auto error_code = conv1x1->DoPostFunc(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1PostFuncRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1CPUKernel::Run() {
  auto src_in = reinterpret_cast<float *>(inputs_[0]->Data());
  auto src_out = reinterpret_cast<float *>(outputs_[0]->Data());

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    Pre1x1Trans(src_in + batch_index * matmul_param_->deep_ * matmul_param_->a_stride_,
                src_out + batch_index * matmul_param_->col_ * matmul_param_->c_stride_);

    int error_code = LiteBackendParallelLaunch(Convolution1x1StrassenRun, this, thread_hw_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "conv1x1 strassen error error_code[" << error_code << "]";
      return RET_ERROR;
    }

    error_code = LiteBackendParallelLaunch(Convolution1x1PostFuncRun, this, thread_oc4_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "conv1x1 post function error error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
