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

#include "src/runtime/kernel/arm/fp16/convolution_1x1_fp16.h"
#include "nnacl/base/conv1x1_base.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/pack_fp16.h"
#include "src/runtime/kernel/arm/fp16/layout_transform_fp16.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int Convolution1x1FP16CPUKernel::InitMatmulParam() {
  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->col_ = conv_param_->output_channel_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->row_16_ = UP_ROUND(matmul_param_->row_, C16NUM);
  matmul_param_->col_8_ = UP_ROUND(matmul_param_->col_, C8NUM);
  matmul_param_->act_type_ = conv_param_->act_type_;
  return RET_OK;
}

Convolution1x1FP16CPUKernel::~Convolution1x1FP16CPUKernel() {
  FreeTmpBuffer();
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
  return;
}

int Convolution1x1FP16CPUKernel::InitConv1x1Param() {
  pre_trans_input_ = (conv_param_->pad_u_ != 0 || conv_param_->pad_l_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);

  if ((matmul_param_->row_ > (C16NUM * op_parameter_->thread_num_)) && (matmul_param_->row_ > matmul_param_->col_)) {
    multi_thread_by_hw_ = true;
    thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(matmul_param_->row_, C16NUM));
    thread_stride_ = UP_DIV(UP_DIV(matmul_param_->row_, C16NUM), thread_count_) * C16NUM;
  } else {
    multi_thread_by_hw_ = false;
    thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(matmul_param_->col_, C8NUM));
    thread_stride_ = UP_DIV(UP_DIV(matmul_param_->col_, C8NUM), thread_count_) * C8NUM;
  }

  if (pre_trans_input_) {
    input_ptr_ = reinterpret_cast<float16_t *>(malloc(matmul_param_->row_ * matmul_param_->deep_ * sizeof(float16_t)));
    if (input_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc input_ptr_ error!";
      return RET_MEMORY_FAILED;
    }
    memset(input_ptr_, 0, matmul_param_->row_ * matmul_param_->deep_ * sizeof(float16_t));
  }
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::InitWeightBias() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = weight_tensor->Channel();
  auto output_channel = weight_tensor->Batch();

  if (in_tensors_.size() == 3) {
    size_t size = UP_ROUND(output_channel, C8NUM) * sizeof(float16_t);
    size_t weight_size = output_channel * sizeof(float16_t);
    bias_data_ = malloc(size);
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc bias_ptr_ error!";
      return RET_ERROR;
    }
    if (origin_bias_data_type_ == kNumberTypeFloat16) {
      memcpy(bias_data_, origin_bias_, output_channel * sizeof(float16_t));
    } else {
      Float32ToFloat16(reinterpret_cast<float *>(origin_bias_), reinterpret_cast<float16_t *>(bias_data_),
                       output_channel);
    }
    memset(reinterpret_cast<char *>(bias_data_) + weight_size, 0, size - weight_size);
  }

  size_t size = input_channel * UP_ROUND(output_channel, C8NUM) * sizeof(float16_t);
  size_t down_size = input_channel * DOWN_DIV(output_channel, C8NUM) * C8NUM * sizeof(float16_t);
  weight_ptr_ = reinterpret_cast<float16_t *>(malloc(size));
  if (weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc weight_ptr_ error!";
    return RET_ERROR;
  }
  memset(reinterpret_cast<char *>(weight_ptr_) + down_size, 0, size - down_size);
  ColMajor2Row8MajorFp16(origin_weight_, weight_ptr_, input_channel, output_channel,
                         origin_weight_data_type_ == kNumberTypeFloat16);
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::Init() {
  matmul_param_ = new (std::nothrow) MatMulParameter();
  if (matmul_param_ == nullptr) {
    MS_LOG(ERROR) << "Init matmul_param_ failed.";
    return RET_ERROR;
  }
  int ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return ret;
  }
  return RET_OK;
}

void Convolution1x1FP16CPUKernel::FreeTmpBuffer() {
  if (pre_trans_input_ && input_ptr_ != nullptr) {
    free(input_ptr_);
    input_ptr_ = nullptr;
  }
  return;
}

int Convolution1x1FP16CPUKernel::ReSize() {
  FreeTmpBuffer();
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return ret;
  }
  ret = InitMatmulParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init matmul param failed.";
    return ret;
  }
  ret = InitConv1x1Param();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init conv1x1 param failed.";
    return ret;
  }
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::RunOc(int task_id) {
  int cur_stride = matmul_param_->col_ - task_id * thread_stride_;
  int cur_oc = MSMIN(thread_stride_, cur_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  auto bias = (bias_data_ == nullptr) ? nullptr : reinterpret_cast<float16_t *>(bias_data_) + thread_stride_ * task_id;

  MatMulFp16(pack_input_, weight_ptr_ + task_id * thread_stride_ * matmul_param_->deep_,
             output_ptr_ + task_id * thread_stride_, bias, matmul_param_->act_type_, matmul_param_->deep_,
             matmul_param_->row_, cur_oc, matmul_param_->col_, OutType_Nhwc);

  return RET_OK;
}

int Convolution1x1FP16CPUKernel::RunHw(int task_id) {
  int res_stride = matmul_param_->row_ - task_id * thread_stride_;
  int cur_hw_ = MSMIN(thread_stride_, res_stride);
  if (cur_hw_ <= 0) {
    return RET_OK;
  }

  float16_t *thread_input_ptr = input_ptr_ + task_id * thread_stride_ * matmul_param_->deep_;
  float16_t *thread_pack_input = pack_input_ + task_id * thread_stride_ * matmul_param_->deep_;
  RowMajor2Col16MajorFp16Opt(thread_input_ptr, thread_pack_input, cur_hw_, matmul_param_->deep_);

  float16_t *thread_output_ptr = output_ptr_ + task_id * thread_stride_ * matmul_param_->col_;
  MatMulFp16(thread_pack_input, weight_ptr_, thread_output_ptr, reinterpret_cast<float16_t *>(bias_data_),
             matmul_param_->act_type_, matmul_param_->deep_, cur_hw_, matmul_param_->col_, matmul_param_->col_,
             OutType_Nhwc);

  return RET_OK;
}

static int Convolution1x1Fp16RunOc(void *cdata, int task_id) {
  auto conv = reinterpret_cast<Convolution1x1FP16CPUKernel *>(cdata);
  auto error_code = conv->RunOc(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1 Fp16 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

static int Convolution1x1Fp16RunHw(void *cdata, int task_id) {
  auto conv = reinterpret_cast<Convolution1x1FP16CPUKernel *>(cdata);
  auto error_code = conv->RunHw(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1 Fp16 Run hw error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::Run() {
  ConvolutionBaseFP16CPUKernel::GetExecuteTensor();

  pack_input_ = reinterpret_cast<float16_t *>(
    ctx_->allocator->Malloc(matmul_param_->row_16_ * matmul_param_->deep_ * sizeof(float16_t)));
  if (pack_input_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc pack_input_ error!";
    return RET_MEMORY_FAILED;
  }

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    output_ptr_ = execute_output_ + batch_index * matmul_param_->row_ * matmul_param_->col_;
    float16_t *batch_in =
      execute_input_ + batch_index * conv_param_->input_h_ * conv_param_->input_w_ * conv_param_->input_channel_;
    if (pre_trans_input_) {
      Conv1x1InputPack(batch_in, input_ptr_, conv_param_, sizeof(float16_t));
    } else {
      input_ptr_ = batch_in;
    }

    int ret = RET_ERROR;
    if (multi_thread_by_hw_) {
      ret = ParallelLaunch(this->context_->thread_pool_, Convolution1x1Fp16RunHw, this, thread_count_);
    } else {
      RowMajor2Col16MajorFp16Opt(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
      ret = ParallelLaunch(this->context_->thread_pool_, Convolution1x1Fp16RunOc, this, thread_count_);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ParallelLaunch failed.";
      ctx_->allocator->Free(pack_input_);
      pack_input_ = nullptr;
      return ret;
    }
  }

  ctx_->allocator->Free(pack_input_);
  pack_input_ = nullptr;
  return RET_OK;
}
}  // namespace mindspore::kernel
