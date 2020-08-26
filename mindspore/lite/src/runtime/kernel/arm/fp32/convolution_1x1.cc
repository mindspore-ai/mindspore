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
  FreeTmpBuffer();
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
}

void Convolution1x1CPUKernel::FreeTmpBuffer() {
  if (pre_trans_input_ && input_ptr_ != nullptr) {
    free(input_ptr_);
    input_ptr_ = nullptr;
  }
  return;
}

int Convolution1x1CPUKernel::ReSize() {
  FreeTmpBuffer();
  ConvolutionBaseCPUKernel::Init();
  InitConv1x1MatmulParam();

  int error_code = InitConv1x1Param();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution base init failed.";
    return error_code;
  }
  return RET_OK;
}

void Convolution1x1CPUKernel::InitConv1x1MatmulParam() {
  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->col_ = conv_param_->output_channel_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->row_12_ = UP_ROUND(matmul_param_->row_, C12NUM);
  matmul_param_->col_8_ = UP_ROUND(matmul_param_->col_, C8NUM);
  matmul_param_->act_type_ = conv_param_->act_type_;
  return;
}

int Convolution1x1CPUKernel::InitConv1x1BiasWeight() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();

  int size = UP_ROUND(output_channel, C8NUM) * sizeof(float);
  bias_data_ = malloc(size);
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc bias_ptr_ error!";
    return RET_ERROR;
  }
  memset(bias_data_, 0, size);
  if (in_tensors_.size() == 3) {
    memcpy(bias_data_, in_tensors_[kBiasIndex]->Data(), output_channel * sizeof(float));
  }

  size = input_channel * UP_ROUND(output_channel, C8NUM) * sizeof(float);
  weight_ptr_ = reinterpret_cast<float *>(malloc(size));
  if (weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc weight_ptr_ error!";
    return RET_ERROR;
  }
  memset(weight_ptr_, 0, size);
  RowMajor2Col8Major(reinterpret_cast<float *>(filter_tensor->Data()), weight_ptr_, output_channel, input_channel);
  return RET_OK;
}

int Convolution1x1CPUKernel::InitConv1x1Param() {
  pre_trans_input_ = (conv_param_->pad_u_ != 0 || conv_param_->pad_l_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(matmul_param_->col_, C8NUM));
  thread_stride_ = UP_DIV(UP_DIV(matmul_param_->col_, C8NUM), thread_count_) * C8NUM;

  if (pre_trans_input_) {
    input_ptr_ = reinterpret_cast<float *>(malloc(matmul_param_->row_ * matmul_param_->deep_ * sizeof(float)));
    if (input_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc input_ptr_ error!";
      return RET_MEMORY_FAILED;
    }
    memset(input_ptr_, 0, matmul_param_->row_ * matmul_param_->deep_ * sizeof(float));
  }

  return RET_OK;
}

void Convolution1x1CPUKernel::Pre1x1Trans(float *src_input, float *src_output) {
  output_ptr_ = src_output;

  if (pre_trans_input_) {
    Conv1x1InputPack(src_input, input_ptr_, conv_param_, sizeof(float));
  } else {
    input_ptr_ = src_input;
  }

  RowMajor2Col12Major(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
  return;
}

int Convolution1x1CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }

  int error_code = InitConv1x1BiasWeight();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution base init failed.";
    return error_code;
  }
  return ReSize();
}

int Convolution1x1CPUKernel::DoConv1x1(int task_id) {
  int cur_oc = MSMIN(thread_stride_, matmul_param_->col_ - task_id * thread_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  MatMulOpt(pack_input_, weight_ptr_ + task_id * thread_stride_ * matmul_param_->deep_,
            output_ptr_ + task_id * thread_stride_, reinterpret_cast<float *>(bias_data_) + thread_stride_ * task_id,
            matmul_param_->act_type_, matmul_param_->deep_, matmul_param_->row_, cur_oc, matmul_param_->col_,
            OutType_Nhwc);
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

  pack_input_ =
    reinterpret_cast<float *>(ctx_->allocator->Malloc(matmul_param_->row_12_ * matmul_param_->deep_ * sizeof(float)));
  if (pack_input_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc pack_input_ error!";
    return RET_MEMORY_FAILED;
  }

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    Pre1x1Trans(src_in + batch_index * conv_param_->input_h_ * conv_param_->input_w_ * conv_param_->input_channel_,
                src_out + batch_index * matmul_param_->row_ * matmul_param_->col_);

    int error_code = LiteBackendParallelLaunch(Convolution1x1Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "conv1x1 strassen error error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }

  if (pack_input_ != nullptr) {
    ctx_->allocator->Free(pack_input_);
    pack_input_ = nullptr;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
