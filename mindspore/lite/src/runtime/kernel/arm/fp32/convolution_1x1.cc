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
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
  if (pack_input_ != nullptr) {
    free(pack_input_);
    pack_input_ = nullptr;
  }
  if (pre_trans_input_ && input_ptr_ != nullptr) {
    free(input_ptr_);
    input_ptr_ = nullptr;
  }
  delete matmul_param_;
}

int Convolution1x1CPUKernel::ReSize() {
  if (pack_input_ != nullptr) {
    free(pack_input_);
    pack_input_ = nullptr;
  }
  if (pre_trans_input_ && input_ptr_ != nullptr) {
    free(input_ptr_);
    input_ptr_ = nullptr;
  }
  InitConv1x1MatmulParam();
  InitConv1x1Param();
  return RET_OK;
}

void Convolution1x1CPUKernel::InitConv1x1MatmulParam() {
  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->col_ = conv_param_->output_channel_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->row_8_ = UP_ROUND(matmul_param_->row_, C8NUM);
  matmul_param_->col_8_ = UP_ROUND(matmul_param_->col_, C8NUM);
  matmul_param_->act_type_ = (conv_param_->is_relu6_) ? ActType_Relu6 : ActType_No;
  matmul_param_->act_type_ = (conv_param_->is_relu_) ? ActType_Relu : matmul_param_->act_type_;
  return;
}

int Convolution1x1CPUKernel::InitConv1x1BiasWeight() {
  if (in_tensors_.size() == 3) {
    bias_data_ = malloc(matmul_param_->col_8_ * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc bias_ptr_ error!";
      return RET_ERROR;
    }
    memset(bias_data_, 0, matmul_param_->col_8_ * sizeof(float));
    memcpy(bias_data_, in_tensors_[2]->Data(), conv_param_->output_channel_ * sizeof(float));
  } else {
    bias_data_ = nullptr;
  }

  weight_ptr_ = reinterpret_cast<float *>(malloc(matmul_param_->deep_ * matmul_param_->col_8_ * sizeof(float)));
  if (weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc weight_ptr_ error!";
    return RET_ERROR;
  }
  memset(weight_ptr_, 0, matmul_param_->deep_ * matmul_param_->col_8_ * sizeof(float));
  RowMajor2Col8Major(reinterpret_cast<float *>(in_tensors_[1]->Data()), weight_ptr_, matmul_param_->col_,
                     matmul_param_->deep_);
  return RET_OK;
}

int Convolution1x1CPUKernel::InitConv1x1Param() {
  pre_trans_input_ = (conv_param_->pad_h_ != 0 || conv_param_->pad_w_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);
  if (pre_trans_input_) {
    input_ptr_ = reinterpret_cast<float *>(malloc(matmul_param_->row_ * matmul_param_->deep_ * sizeof(float)));
    if (input_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc input_ptr_ error!";
      return RET_MEMORY_FAILED;
    }
    memset(input_ptr_, 0, matmul_param_->row_ * matmul_param_->deep_ * sizeof(float));
  }

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(matmul_param_->col_, C8NUM));
  thread_stride_ = UP_DIV(UP_DIV(matmul_param_->col_, C8NUM), thread_count_) * C8NUM;

  pack_input_ = reinterpret_cast<float *>(malloc(matmul_param_->row_8_ * matmul_param_->deep_ * sizeof(float)));
  if (pack_input_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc pack_input_ error!";
    return RET_MEMORY_FAILED;
  }
  memset(pack_input_, 0, matmul_param_->row_8_ * matmul_param_->deep_ * sizeof(float));
  return RET_OK;
}

void Convolution1x1CPUKernel::Pre1x1Trans(float *src_input, float *src_output) {
  output_ptr_ = src_output;

  if (pre_trans_input_) {
    Conv1x1InputPackFp32(src_input, input_ptr_, conv_param_);
  } else {
    input_ptr_ = src_input;
  }

  RowMajor2Col8Major(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
  return;
}

int Convolution1x1CPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
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

int Convolution1x1CPUKernel::DoConv1x1(int task_id) {
  int cur_oc = MSMIN(thread_stride_, matmul_param_->col_ - task_id * thread_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  auto bias = (bias_data_ == nullptr) ? nullptr : reinterpret_cast<float *>(bias_data_) + thread_stride_ * task_id;

  MatMul(pack_input_, weight_ptr_ + task_id * thread_stride_ * matmul_param_->deep_,
         output_ptr_ + task_id * thread_stride_, bias, matmul_param_->act_type_, matmul_param_->deep_,
         matmul_param_->row_, cur_oc, matmul_param_->col_, true);

  return RET_OK;
}

int Convolution1x1Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv1x1 = reinterpret_cast<Convolution1x1CPUKernel *>(cdata);
  auto error_code = conv1x1->DoConv1x1(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1CPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto src_in = reinterpret_cast<float *>(in_tensors_[0]->Data());
  auto src_out = reinterpret_cast<float *>(out_tensors_[0]->Data());

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    Pre1x1Trans(src_in + batch_index * conv_param_->input_h_ * conv_param_->input_w_ * conv_param_->input_channel_,
                src_out + batch_index * matmul_param_->row_ * matmul_param_->col_);

    int error_code = LiteBackendParallelLaunch(Convolution1x1Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "conv1x1 strassen error error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
