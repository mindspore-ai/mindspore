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
#include "src/common/file_utils.h"
#ifdef ENABLE_ARM64
#include "src/runtime/kernel/arm/int8/opt_op_handler.h"
#endif

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int Convolution1x1Int8Pre(void *cdata, int task_id) {
  auto conv = reinterpret_cast<Convolution1x1Int8CPUKernel *>(cdata);
  auto error_code = conv->RunPre(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv1x1 Int8 RunPre error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

Convolution1x1Int8CPUKernel::~Convolution1x1Int8CPUKernel() {
  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
  if (packed_weight_ != nullptr) {
    free(packed_weight_);
    packed_weight_ = nullptr;
  }
  if (filter_peroc_ && filter_zp_ptr_ != nullptr) {
    free(filter_zp_ptr_);
    filter_zp_ptr_ = nullptr;
  }
  if (filter_peroc_ && left_shift_ != nullptr) {
    free(left_shift_);
    left_shift_ = nullptr;
  }
  if (filter_peroc_ && right_shift_ != nullptr) {
    free(right_shift_);
    right_shift_ = nullptr;
  }
  if (filter_peroc_ && multiplier_ != nullptr) {
    free(multiplier_);
    multiplier_ = nullptr;
  }
  FreeResizeBuf();
  FreeQuantParam();
}

void Convolution1x1Int8CPUKernel::FreeResizeBuf() {
  if (pre_trans_input_ && input_ptr_ != nullptr) {
    free(input_ptr_);
    input_ptr_ = nullptr;
  }
  return;
}

void Convolution1x1Int8CPUKernel::CheckSupportOptimize() {
  support_optimize_ = false;
  matmul_func_ = MatMulInt8_8x8_r;
#ifdef ENABLE_ARM64
  if (mindspore::lite::IsSupportSDot()) {
    support_optimize_ = true;
    matmul_func_ = MatMulRInt8_optimize_handler;
  } else {
    support_optimize_ = false;
    matmul_func_ = nullptr;
  }
#endif
  return;
}

int Convolution1x1Int8CPUKernel::InitBiasByzp(void *src_weight, int input_channel, int output_channel, int round_oc) {
  /* bias = bias - v2 x zp1 + zp1 x zp2  */
  int32_t *bias_data = reinterpret_cast<int32_t *>(bias_data_);
  int8_t *weight = reinterpret_cast<int8_t *>(src_weight);
  int32_t input_zp = conv_param_->conv_quant_arg_.input_quant_args_[0].zp_;
  for (int oc = 0; oc < output_channel; oc++) {
    int32_t weight_sum_value = 0;
    int32_t filter_zp = (filter_peroc_) ? conv_param_->conv_quant_arg_.filter_quant_args_[oc].zp_
                                        : conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_;
    for (int ic = 0; ic < input_channel; ic++) {
      weight_sum_value += weight[oc * input_channel + ic];
    }
    bias_data[oc] += filter_zp * input_zp * input_channel - weight_sum_value * input_zp;
  }

  if (filter_peroc_) {
    filter_zp_ptr_ = reinterpret_cast<int32_t *>(malloc(output_channel * sizeof(int32_t)));
    if (filter_zp_ptr_ == nullptr) {
      return RET_ERROR;
    }
    for (int fi = 0; fi < output_channel; fi++) {
      filter_zp_ptr_[fi] = conv_param_->conv_quant_arg_.filter_quant_args_[fi].zp_;
    }

    left_shift_ = reinterpret_cast<int32_t *>(malloc(round_oc * sizeof(int32_t)));
    if (left_shift_ == nullptr) {
      return RET_ERROR;
    }
    memset(left_shift_, 0, round_oc * sizeof(int32_t));
    memcpy(left_shift_, conv_param_->conv_quant_arg_.left_shift_, output_channel * sizeof(int32_t));
    right_shift_ = reinterpret_cast<int32_t *>(malloc(round_oc * sizeof(int32_t)));
    if (right_shift_ == nullptr) {
      return RET_ERROR;
    }
    memset(right_shift_, 0, round_oc * sizeof(int32_t));
    memcpy(right_shift_, conv_param_->conv_quant_arg_.right_shift_, output_channel * sizeof(int32_t));
    multiplier_ = reinterpret_cast<int32_t *>(malloc(round_oc * sizeof(int32_t)));
    if (multiplier_ == nullptr) {
      return RET_ERROR;
    }
    memset(multiplier_, 0, round_oc * sizeof(int32_t));
    memcpy(multiplier_, conv_param_->conv_quant_arg_.quant_multiplier_, output_channel * sizeof(int32_t));
  }
  return RET_OK;
}

int Convolution1x1Int8CPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();

  /* weight */
  size_t size = support_optimize_ ? UP_ROUND(input_channel, C4NUM) * UP_ROUND(output_channel, C8NUM) * sizeof(int8_t)
                                  : UP_ROUND(input_channel, C16NUM) * UP_ROUND(output_channel, C4NUM) * sizeof(int8_t);
  packed_weight_ = reinterpret_cast<int8_t *>(malloc(size));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 int8 Malloc weight error!";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, size);
  if (support_optimize_) {
    RowMajor2Row8x4MajorInt8(reinterpret_cast<int8_t *>(filter_tensor->MutableData()), packed_weight_, output_channel,
                             input_channel);
  } else {
    RowMajor2Row16x4MajorInt8(reinterpret_cast<int8_t *>(filter_tensor->MutableData()), packed_weight_, output_channel,
                              input_channel);
  }

  int col4 = UP_ROUND(output_channel, C4NUM);
  int col8 = UP_ROUND(output_channel, C8NUM);
  size = support_optimize_ ? col8 : col4;
  bias_data_ = malloc(size * sizeof(int32_t));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 int8 Malloc bias_ptr_ error!";
    return RET_ERROR;
  }
  memset(bias_data_, 0, size * sizeof(int32_t));
  if (in_tensors_.size() == 3) {
    memcpy(bias_data_, in_tensors_[kBiasIndex]->MutableData(), output_channel * sizeof(int32_t));
  }

  InitBiasByzp(filter_tensor->MutableData(), input_channel, output_channel, size);
  return RET_OK;
}

int Convolution1x1Int8CPUKernel::InitWeightBiasArm32() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();

  /* weight */
  size_t size = UP_ROUND(input_channel, C16NUM) * UP_ROUND(output_channel, C2NUM) * sizeof(int8_t);
  packed_weight_ = reinterpret_cast<int8_t *>(malloc(size));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 int8 arm32 Malloc weight error!";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, size);
  RowMajor2Row2x16MajorInt8(reinterpret_cast<int8_t *>(filter_tensor->MutableData()), packed_weight_, output_channel,
                            input_channel);

  /* bias */
  int col2 = UP_ROUND(output_channel, C2NUM);
  bias_data_ = malloc(col2 * sizeof(int32_t));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 int8 arm32 Malloc bias_ptr_ error!";
    return RET_ERROR;
  }
  memset(bias_data_, 0, col2 * sizeof(int32_t));
  if (in_tensors_.size() == 3) {
    memcpy(bias_data_, in_tensors_[kBiasIndex]->MutableData(), output_channel * sizeof(int32_t));
  }

  InitBiasByzp(filter_tensor->MutableData(), input_channel, output_channel, col2);
  return RET_OK;
}

int Convolution1x1Int8CPUKernel::Init() {
  matmul_param_ = new (std::nothrow) MatMulParameter();
  if (matmul_param_ == nullptr) {
    MS_LOG(ERROR) << "Init matmul_param_ failed.";
    return RET_ERROR;
  }

  auto ret = SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
  }

  filter_peroc_ = (conv_param_->conv_quant_arg_.filter_arg_num_ != 1);

  CheckSupportOptimize();

#ifdef ENABLE_ARM32
  ret = InitWeightBiasArm32();
#else
  ret = InitWeightBias();
#endif
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int Convolution1x1Int8CPUKernel::InitParam() {
  pre_trans_input_ = (conv_param_->pad_u_ != 0 || conv_param_->pad_l_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);

  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->col_ = conv_param_->output_channel_;
  matmul_param_->col_2_ = UP_ROUND(matmul_param_->col_, C2NUM);
  matmul_param_->col_4_ = UP_ROUND(matmul_param_->col_, C4NUM);
  matmul_param_->col_8_ = UP_ROUND(matmul_param_->col_, C8NUM);
  matmul_param_->row_4_ = UP_ROUND(matmul_param_->row_, C4NUM);
  matmul_param_->row_8_ = UP_ROUND(matmul_param_->row_, C8NUM);
  matmul_param_->deep_4_ = UP_ROUND(matmul_param_->deep_, C4NUM);
  matmul_param_->deep_16_ = UP_ROUND(matmul_param_->deep_, C16NUM);

  int row_pack_count = 0;
  int col_pack_count = 0;
#ifdef ENABLE_ARM32
  row_pack_count = C4NUM;
  col_pack_count = C2NUM;
#else
  if (support_optimize_) {
    row_pack_count = C8NUM;
    col_pack_count = C8NUM;
  } else {
    row_pack_count = C4NUM;
    col_pack_count = C4NUM;
  }
#endif

  /* init input sum size */
  if (conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL) {
    input_sum_size_ = UP_ROUND(matmul_param_->col_, col_pack_count) * UP_ROUND(matmul_param_->row_, row_pack_count);
  } else {
    input_sum_size_ = UP_ROUND(matmul_param_->row_, row_pack_count);
  }

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(matmul_param_->col_, col_pack_count));
  thread_stride_ = UP_DIV(UP_DIV(matmul_param_->col_, col_pack_count), thread_count_);

  thread_count_hw_ = MSMIN(op_parameter_->thread_num_, UP_DIV(matmul_param_->row_, row_pack_count));
  thread_stride_hw_ = UP_DIV(UP_DIV(matmul_param_->row_, row_pack_count), thread_count_hw_);

  if (pre_trans_input_) {
    input_ptr_ = reinterpret_cast<int8_t *>(malloc(matmul_param_->row_ * matmul_param_->deep_ * sizeof(int8_t)));
    if (input_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 int8 Malloc input_ptr_ error!";
      return RET_MEMORY_FAILED;
    }
    memset(input_ptr_, 0, matmul_param_->row_ * matmul_param_->deep_ * sizeof(int8_t));
  }
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

  if (support_optimize_) {
    ParallelLaunch(this->context_->thread_pool_, Convolution1x1Int8Pre, this, thread_count_hw_);
  } else {
    RowMajor2Row16x4MajorInt8(input_ptr_, packed_input_, matmul_param_->row_, matmul_param_->deep_);
    PackInputSum16x4Int8(packed_input_, input_sum_, filter_zp_ptr_, conv_param_);
  }
  return;
}

int Convolution1x1Int8CPUKernel::RunImpl(int task_id) {
  int32_t *cur_input_sum = input_sum_;
  int32_t *cur_left_shift = conv_param_->conv_quant_arg_.left_shift_;
  int32_t *cur_right_shift = conv_param_->conv_quant_arg_.right_shift_;
  int32_t *cur_multiplier = conv_param_->conv_quant_arg_.quant_multiplier_;

#ifdef ENABLE_ARM32
  int cur_stride = thread_stride_ * C2NUM;
  int res_stride = matmul_param_->col_ - task_id * thread_stride_ * C2NUM;
  int cur_oc = MSMIN(cur_stride, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  if (filter_peroc_) {
    cur_input_sum = input_sum_ + task_id * matmul_param_->row_4_ * thread_stride_ * C2NUM;
    cur_left_shift = left_shift_ + task_id * thread_stride_ * C2NUM;
    cur_right_shift = right_shift_ + task_id * thread_stride_ * C2NUM;
    cur_multiplier = multiplier_ + task_id * thread_stride_ * C2NUM;
  }
  Conv1x1Int8Arm32(packed_input_, packed_weight_ + task_id * thread_stride_ * C2NUM * matmul_param_->deep_16_,
                   output_ptr_ + task_id * thread_stride_ * C2NUM, cur_input_sum,
                   reinterpret_cast<int32_t *>(bias_data_) + task_id * thread_stride_ * C2NUM, matmul_param_->row_,
                   cur_oc, matmul_param_->deep_16_, cur_left_shift, cur_right_shift, cur_multiplier, conv_param_);
#else
  if (support_optimize_) {
    int cur_stride = thread_stride_ * C8NUM;
    int res_stride = matmul_param_->col_ - task_id * thread_stride_ * C8NUM;
    int cur_oc = MSMIN(cur_stride, res_stride);
    if (cur_oc <= 0) {
      return RET_OK;
    }
    if (filter_peroc_) {
      cur_input_sum = input_sum_ + task_id * matmul_param_->row_8_ * thread_stride_ * C8NUM;
      cur_left_shift = left_shift_ + task_id * thread_stride_ * C8NUM;
      cur_right_shift = right_shift_ + task_id * thread_stride_ * C8NUM;
      cur_multiplier = multiplier_ + task_id * thread_stride_ * C8NUM;
    }
    Conv1x1Int8Opt(packed_input_, packed_weight_ + task_id * thread_stride_ * C8NUM * matmul_param_->deep_4_,
                   output_ptr_ + task_id * thread_stride_ * C8NUM, cur_input_sum,
                   reinterpret_cast<int32_t *>(bias_data_) + task_id * thread_stride_ * C8NUM, matmul_param_->row_,
                   cur_oc, matmul_param_->deep_4_, cur_left_shift, cur_right_shift, cur_multiplier, conv_param_,
                   matmul_func_);
  } else {
    int cur_stride = thread_stride_ * C4NUM;
    int res_stride = matmul_param_->col_ - task_id * thread_stride_ * C4NUM;
    int cur_oc = MSMIN(cur_stride, res_stride);
    if (cur_oc <= 0) {
      return RET_OK;
    }
    if (filter_peroc_) {
      cur_input_sum = input_sum_ + task_id * matmul_param_->row_4_ * thread_stride_ * C4NUM;
      cur_left_shift = left_shift_ + task_id * thread_stride_ * C4NUM;
      cur_right_shift = right_shift_ + task_id * thread_stride_ * C4NUM;
      cur_multiplier = multiplier_ + task_id * thread_stride_ * C4NUM;
    }
    Conv1x1Int8(packed_input_, packed_weight_ + task_id * thread_stride_ * C4NUM * matmul_param_->deep_16_,
                output_ptr_ + task_id * thread_stride_ * C4NUM, cur_input_sum,
                reinterpret_cast<int32_t *>(bias_data_) + task_id * thread_stride_ * C4NUM, matmul_param_->row_, cur_oc,
                matmul_param_->deep_16_, cur_left_shift, cur_right_shift, cur_multiplier, conv_param_);
  }
#endif
  return RET_OK;
}

int Convolution1x1Int8CPUKernel::RunPre(int task_id) {
  int cur_stride = thread_stride_hw_ * C8NUM;
  int res_stride = matmul_param_->row_ - task_id * thread_stride_hw_ * C8NUM;
  int cur_hw = MSMIN(cur_stride, res_stride);
  if (cur_hw <= 0) {
    return RET_OK;
  }

  if (filter_peroc_) {
    Conv1x1PreOptPeroc(input_ptr_ + task_id * thread_stride_hw_ * C8NUM * matmul_param_->deep_,
                       packed_input_ + task_id * thread_stride_hw_ * C8NUM * matmul_param_->deep_4_,
                       input_sum_ + task_id * thread_stride_hw_ * C8NUM * C8NUM, matmul_param_->deep_,
                       matmul_param_->col_, cur_hw, filter_zp_ptr_, matmul_param_->row_8_ * C8NUM);
  } else {
    Conv1x1PreOptPert(input_ptr_ + task_id * thread_stride_hw_ * C8NUM * matmul_param_->deep_,
                      packed_input_ + task_id * thread_stride_hw_ * C8NUM * matmul_param_->deep_4_,
                      input_sum_ + task_id * thread_stride_hw_ * C8NUM, matmul_param_->deep_, cur_hw, conv_param_);
  }

  return RET_OK;
}

int Convolution1x1Int8Impl(void *cdata, int task_id) {
  auto conv = reinterpret_cast<Convolution1x1Int8CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv1x1 Int8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1Int8CPUKernel::InitRunBuf() {
  input_sum_ = reinterpret_cast<int32_t *>(ctx_->allocator->Malloc(input_sum_size_ * sizeof(int32_t)));
  if (input_sum_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_sum_ failed.";
    return RET_ERROR;
  }

  size_t size = support_optimize_ ? UP_ROUND(matmul_param_->row_, C8NUM) * UP_ROUND(matmul_param_->deep_, C4NUM)
                                  : UP_ROUND(matmul_param_->row_, C4NUM) * UP_ROUND(matmul_param_->deep_, C16NUM);
  packed_input_ = reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(size * sizeof(int8_t)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "conv1x1 int8 Malloc packed_input_ error!";
    return RET_ERROR;
  }
  return RET_OK;
}

void Convolution1x1Int8CPUKernel::FreeRunBuf() {
  if (packed_input_ != nullptr) {
    ctx_->allocator->Free(packed_input_);
    packed_input_ = nullptr;
  }
  if (input_sum_ != nullptr) {
    ctx_->allocator->Free(input_sum_);
    input_sum_ = nullptr;
  }
  return;
}

int Convolution1x1Int8CPUKernel::Run() {
  int error_code = InitRunBuf();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv1x1 int8 InitRunBuf error_code[" << error_code << "]";
    FreeRunBuf();
    return RET_ERROR;
  }

  int8_t *src_in = reinterpret_cast<int8_t *>(in_tensors_[0]->MutableData());
  int8_t *src_out = reinterpret_cast<int8_t *>(out_tensors_[0]->MutableData());

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    Pre1x1Trans(src_in + batch_index * conv_param_->input_h_ * conv_param_->input_w_ * conv_param_->input_channel_,
                src_out + batch_index * matmul_param_->row_ * matmul_param_->col_);
    auto ret = ParallelLaunch(this->context_->thread_pool_, Convolution1x1Int8Impl, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ParallelLaunch run error error_code[" << ret << "]";
      FreeRunBuf();
      return ret;
    }
  }

  FreeRunBuf();

  return RET_OK;
}
}  // namespace mindspore::kernel
