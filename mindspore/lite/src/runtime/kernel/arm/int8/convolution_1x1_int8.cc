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

#include "src/runtime/kernel/arm/int8/convolution_1x1_int8.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

Convolution1x1Int8CPUKernel::~Convolution1x1Int8CPUKernel() {
  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
  if (packed_weight_ != nullptr) {
    delete packed_weight_;
    packed_weight_ = nullptr;
  }
  FreeResizeBuf();
  FreeQuantParam();
}

void Convolution1x1Int8CPUKernel::FreeResizeBuf() {
  if (packed_input_ != nullptr) {
    free(packed_input_);
    packed_input_ = nullptr;
  }
  if (input_sum_ != nullptr) {
    free(input_sum_);
    input_sum_ = nullptr;
  }
  return;
}

void Convolution1x1Int8CPUKernel::CheckSupportOptimize() {
  support_optimize_ = false;
  matmul_func_ = MatMulInt8_16x4_r;
#ifdef ENABLE_ARM64
  void *optimize_op_handler = OptimizeModule::GetInstance()->optimized_op_handler_;
  if (optimize_op_handler != nullptr) {
    dlerror();
    *(reinterpret_cast<void **>(&matmul_func_)) = dlsym(optimize_op_handler, "MatMulRInt8_optimize_handler");
    auto dlopen_error = dlerror();
    if (dlopen_error != nullptr) {
      MS_LOG(ERROR) << "load matmul func failed! " << dlopen_error << ".";
      support_optimize_ = false;
      matmul_func_ = nullptr;
    } else {
      support_optimize_ = true;
    }
  } else {
    support_optimize_ = false;
    matmul_func_ = nullptr;
  }
#endif

  matmul_func_ = MatMulInt8_16x4_r;
  return;
}

int Convolution1x1Int8CPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();

  /* weight */
  size_t size = UP_ROUND(input_channel, C16NUM) * UP_ROUND(output_channel, C4NUM) * sizeof(int8_t);
  packed_weight_ = reinterpret_cast<int8_t *>(malloc(size));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 int8 Malloc weight error!";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, size);
  RowMajor2Row4x16MajorInt8(reinterpret_cast<int8_t *>(filter_tensor->Data()), packed_weight_, output_channel,
                            input_channel);

  /* bias = bias - v2 x zp1 + zp1 x zp2  */
  int col4 = UP_ROUND(output_channel, C4NUM);
  bias_data_ = malloc(col4 * sizeof(int32_t));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 int8 Malloc bias_ptr_ error!";
    return RET_ERROR;
  }
  memset(bias_data_, 0, col4 * sizeof(int32_t));
  if (in_tensors_.size() == 3) {
    memcpy(bias_data_, in_tensors_[kBiasIndex]->Data(), output_channel * sizeof(int32_t));
  }

  int32_t *bias_data = reinterpret_cast<int32_t *>(bias_data_);
  int8_t *weight = reinterpret_cast<int8_t *>(filter_tensor->Data());
  int32_t input_zp = conv_param_->conv_quant_arg_.input_quant_args_[0].zp_;
  for (int oc = 0; oc < output_channel; oc++) {
    int32_t weight_sum_value = 0;
    int32_t filter_zp = (conv_param_->conv_quant_arg_.filter_arg_num_ == 1)
                          ? conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_
                          : conv_param_->conv_quant_arg_.filter_quant_args_[oc].zp_;
    for (int ic = 0; ic < input_channel; ic++) {
      weight_sum_value += weight[oc * input_channel + ic];
    }
    bias_data[oc] += filter_zp * input_zp * input_channel - weight_sum_value * input_zp;
  }
  return RET_OK;
}

int Convolution1x1Int8CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  matmul_param_ = new (std::nothrow) MatMulParameter();
  if (matmul_param_ == nullptr) {
    MS_LOG(ERROR) << "Init matmul_param_ failed.";
    return RET_ERROR;
  }

  CheckSupportOptimize();

  auto ret = SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
  }

  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return ret;
  }

  return ReSize();
}

int Convolution1x1Int8CPUKernel::InitParam() {
  pre_trans_input_ = (conv_param_->pad_h_ != 0 || conv_param_->pad_w_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);

  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->col_ = conv_param_->output_channel_;

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(matmul_param_->col_, C4NUM));
  thread_stride_ = UP_DIV(UP_DIV(matmul_param_->col_, C4NUM), thread_count_);

  size_t size = UP_ROUND(matmul_param_->row_, C4NUM) * UP_ROUND(matmul_param_->deep_, C16NUM);
  packed_input_ = reinterpret_cast<int8_t *>(malloc(size * sizeof(int8_t)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "conv1x1 int8 Malloc packed_input_ error!";
    return RET_ERROR;
  }
  memset(packed_input_, 0, size * sizeof(int8_t));

  if (conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL) {
    size = UP_ROUND(conv_param_->output_channel_, C4NUM) * UP_ROUND(matmul_param_->row_, C4NUM);
  } else {
    size = UP_ROUND(matmul_param_->row_, C4NUM);
  }
  input_sum_ = reinterpret_cast<int32_t *>(malloc(size * sizeof(int32_t)));
  if (input_sum_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_sum_ failed.";
    return RET_ERROR;
  }
  memset(input_sum_, 0, size * sizeof(int32_t));

  return RET_OK;
}

int Convolution1x1Int8CPUKernel::ReSize() {
  FreeResizeBuf();

  ConvolutionBaseCPUKernel::Init();

  int error_code = InitParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution base init failed.";
    return error_code;
  }
  return RET_OK;
}

void Convolution1x1Int8CPUKernel::Pre1x1Trans(int8_t *src_input, int8_t *src_output) {
  output_ptr_ = src_output;
  if (pre_trans_input_) {
    Conv1x1InputPack(src_input, input_ptr_, conv_param_, sizeof(int8_t));
  } else {
    input_ptr_ = src_input;
  }
  RowMajor2Row16x4MajorInt8(input_ptr_, packed_input_, matmul_param_->row_, matmul_param_->deep_);
  return;
}

int Convolution1x1Int8CPUKernel::RunImpl(int task_id) {
  int cur_oc = MSMIN(thread_stride_ * C4NUM, matmul_param_->col_ - task_id * thread_stride_ * C4NUM);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  int32_t *bias = reinterpret_cast<int32_t *>(bias_data_) + thread_stride_ * C4NUM * task_id;

  Conv1x1Int8(packed_input_, packed_weight_ + task_id * thread_stride_ * C4NUM * matmul_param_->deep_,
              output_ptr_ + task_id * thread_stride_ * C4NUM, input_sum_, bias + task_id * thread_stride_ * C4NUM,
              matmul_param_->row_, cur_oc, UP_ROUND(matmul_param_->deep_, C16NUM), conv_param_, matmul_func_);
  return RET_OK;
}

int Convolution1x1Int8Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv = reinterpret_cast<Convolution1x1Int8CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv1x1 Int8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1Int8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }

  if (pre_trans_input_) {
    input_ptr_ =
      reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(matmul_param_->row_ * matmul_param_->deep_ * sizeof(int8_t)));
    if (input_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 int8 Malloc input_ptr_ error!";
      return RET_MEMORY_FAILED;
    }
  }

  int8_t *src_in = reinterpret_cast<int8_t *>(in_tensors_[0]->Data());
  int8_t *src_out = reinterpret_cast<int8_t *>(out_tensors_[0]->Data());

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    Pre1x1Trans(src_in + batch_index * conv_param_->input_h_ * conv_param_->input_w_ * conv_param_->input_channel_,
                src_out + batch_index * matmul_param_->row_ * matmul_param_->col_);

    PackInputSum16x4Int8(packed_input_, input_sum_, matmul_param_->deep_, matmul_param_->col_, matmul_param_->row_,
                         conv_param_);

    int error_code = LiteBackendParallelLaunch(Convolution1x1Int8Impl, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "conv1x1 fp16 error error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }

  if (pre_trans_input_ && input_ptr_ != nullptr) {
    ctx_->allocator->Free(input_ptr_);
    input_ptr_ = nullptr;
  }

  return RET_OK;
}
}  // namespace mindspore::kernel
