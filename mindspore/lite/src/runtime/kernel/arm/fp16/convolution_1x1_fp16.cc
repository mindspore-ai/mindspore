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
#include "src/runtime/kernel/arm/nnacl/fp16/conv_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/cast_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/pack_fp16.h"
#include "src/runtime/kernel/arm/fp16/layout_transform_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;

namespace mindspore::kernel {
int Convolution1x1FP16CPUKernel::InitMatmulParam() {
  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->col_ = conv_param_->output_channel_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->row_16_ = UP_ROUND(matmul_param_->row_, C16NUM);
  matmul_param_->col_8_ = UP_ROUND(matmul_param_->col_, C8NUM);
  matmul_param_->act_type_ = (conv_param_->is_relu6_) ? ActType_Relu6 : ActType_No;
  matmul_param_->act_type_ = (conv_param_->is_relu_) ? ActType_Relu : matmul_param_->act_type_;
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::InitConv1x1Param() {
  pre_trans_input_ = (conv_param_->pad_h_ != 0 || conv_param_->pad_w_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);
  if (pre_trans_input_) {
    input_ptr_ = reinterpret_cast<float16_t *>(malloc(matmul_param_->row_ * matmul_param_->deep_ * sizeof(float16_t)));
    if (input_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc input_ptr_ error!";
      return RET_MEMORY_FAILED;
    }
    memset(input_ptr_, 0, matmul_param_->row_ * matmul_param_->deep_ * sizeof(float16_t));
  }

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(matmul_param_->col_, C8NUM));
  thread_stride_ = UP_DIV(UP_DIV(matmul_param_->col_, C8NUM), thread_count_) * C8NUM;

  pack_input_ =
    reinterpret_cast<float16_t *>(malloc(matmul_param_->row_16_ * matmul_param_->deep_ * sizeof(float16_t)));
  if (pack_input_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc pack_input_ error!";
    return RET_MEMORY_FAILED;
  }
  memset(pack_input_, 0, matmul_param_->row_16_ * matmul_param_->deep_ * sizeof(float16_t));
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::InitWeightBias() {
  auto ret = ConvolutionBaseFP16CPUKernel::GetExecuteFilter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Execute filter failed.";
    return ret;
  }
  if (in_tensors_.size() == 3) {
    bias_data_ = malloc(matmul_param_->col_8_ * sizeof(float16_t));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc bias_ptr_ error!";
      return RET_ERROR;
    }
    memset(bias_data_, 0, matmul_param_->col_8_ * sizeof(float16_t));
    memcpy(bias_data_, in_tensors_[2]->Data(), conv_param_->output_channel_ * sizeof(float16_t));
  } else {
    bias_data_ = nullptr;
  }

  weight_ptr_ = reinterpret_cast<float16_t *>(malloc(matmul_param_->deep_ * matmul_param_->col_8_ * sizeof(float16_t)));
  if (weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc weight_ptr_ error!";
    return RET_ERROR;
  }
  memset(weight_ptr_, 0, matmul_param_->deep_ * matmul_param_->col_8_ * sizeof(float16_t));
  RowMajor2Col8MajorFp16(reinterpret_cast<float16_t *>(execute_weight_), weight_ptr_, matmul_param_->col_,
                         matmul_param_->deep_);

  return RET_OK;
}

int Convolution1x1FP16CPUKernel::InitBuffer() {
  /*=============================fp16_input_============================*/
  size_t fp16_input_size = conv_param_->input_channel_ * conv_param_->input_batch_ * conv_param_->input_h_ *
                           conv_param_->input_w_ * sizeof(float16_t);
  fp16_input_ = reinterpret_cast<float16_t *>(malloc(fp16_input_size));
  if (fp16_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc fp16_input_ failed.";
    return RET_ERROR;
  }
  memset(fp16_input_, 0, fp16_input_size);

  /*=============================fp16_out_============================*/
  size_t fp16_output_size = conv_param_->output_channel_ * conv_param_->output_batch_ * conv_param_->output_h_ *
                            conv_param_->output_w_ * sizeof(float16_t);
  fp16_out_ = reinterpret_cast<float16_t *>(malloc(fp16_output_size));
  if (fp16_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc fp16_out_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::Init() {
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
  ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init buffer failed.";
    return ret;
  }
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return ret;
  }
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::ReSize() {
  if (fp16_out_ != nullptr) {
    free(fp16_out_);
  }
  if (fp16_input_ != nullptr) {
    free(fp16_input_);
  }
  if (fp16_weight_ != nullptr) {
    free(fp16_weight_);
  }
  if (input_ptr_ != nullptr) {
    free(input_ptr_);
  }
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
  }

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
  ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init buffer failed.";
    return ret;
  }

  return RET_OK;
}

void Convolution1x1FP16CPUKernel::Pre1x1Trans(float16_t *src_input, float16_t *src_output) {
  output_ptr_ = src_output;
  if (pre_trans_input_) {
    Conv1x1InputPackFp16(src_input, input_ptr_, conv_param_);
  } else {
    input_ptr_ = src_input;
  }

  RowMajor2Col8MajorFp16(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
  return;
}

int Convolution1x1FP16CPUKernel::RunImpl(int task_id) {
  int cur_oc = MSMIN(thread_stride_, matmul_param_->col_ - task_id * thread_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  auto bias = (bias_data_ == nullptr) ? nullptr : reinterpret_cast<float16_t *>(bias_data_) + thread_stride_ * task_id;

  MatMulFp16(pack_input_, weight_ptr_ + task_id * thread_stride_ * matmul_param_->deep_,
             output_ptr_ + task_id * thread_stride_, bias, matmul_param_->act_type_, matmul_param_->deep_,
             matmul_param_->row_, cur_oc, matmul_param_->col_, true);

  return RET_OK;
}

int Convolution1x1Fp16Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv = reinterpret_cast<Convolution1x1FP16CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1 Fp16 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }

  ret = ConvolutionBaseFP16CPUKernel::GetExecuteTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get executor tensor failed.";
    return ret;
  }

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    Pre1x1Trans(
      execute_input_ + batch_index * conv_param_->input_h_ * conv_param_->input_w_ * conv_param_->input_channel_,
      execute_output_ + batch_index * matmul_param_->row_ * matmul_param_->col_);

    int error_code = LiteBackendParallelLaunch(Convolution1x1Fp16Impl, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "conv1x1 fp16 error error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }

  ConvolutionBaseFP16CPUKernel::IfCastOutput();
  return RET_OK;
}
}  // namespace mindspore::kernel
