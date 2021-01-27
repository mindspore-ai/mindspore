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
#include "src/runtime/kernel/arm/int8/opt_op_handler.h"

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

int Convolution1x1Int8HwRun(void *cdata, int task_id) {
  auto conv = reinterpret_cast<Convolution1x1Int8CPUKernel *>(cdata);
  auto error_code = conv->HwRun(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv1x1 Int8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1Int8OcRun(void *cdata, int task_id) {
  auto conv = reinterpret_cast<Convolution1x1Int8CPUKernel *>(cdata);
  auto error_code = conv->OcRun(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv1x1 Int8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1Int8OcOptPre(void *cdata, int task_id) {
  auto conv = reinterpret_cast<Convolution1x1Int8CPUKernel *>(cdata);
  auto error_code = conv->OcOptPre(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv1x1 Int8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1Int8CPUKernel::OcRun(int task_id) {
  if (support_optimize_) {
    return RunArm64OptOc(task_id);
  } else {
    return RunArmOc(task_id);
  }
}

int Convolution1x1Int8CPUKernel::HwRun(int task_id) {
  if (support_optimize_) {
    return RunArm64OptHw(task_id);
  } else {
    return RunArmHw(task_id);
  }
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

void Convolution1x1Int8CPUKernel::CheckSupportOptimize() {
  support_optimize_ = false;
  matmul_func_ = MatMulInt8_4x16_r;
#ifdef ENABLE_ARM64
  if (mindspore::lite::IsSupportSDot()) {
    support_optimize_ = true;
    matmul_func_ = MatMulDpInt8_optimize_handler;
  } else {
    support_optimize_ = false;
    matmul_func_ = nullptr;
  }
#endif
  return;
}

int Convolution1x1Int8CPUKernel::InitBiasByzp(const void *src_weight, int input_channel, int output_channel,
                                              int round_oc) {
  /* bias = bias - v2 x zp1 + zp1 x zp2  */
  int32_t *bias_data = reinterpret_cast<int32_t *>(bias_data_);
  auto *weight = static_cast<const int8_t *>(src_weight);
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
    /* filter zp */
    filter_zp_ptr_ = reinterpret_cast<int32_t *>(malloc(round_oc * sizeof(int32_t)));
    if (filter_zp_ptr_ == nullptr) {
      return RET_ERROR;
    }
    for (int fi = 0; fi < output_channel; fi++) {
      filter_zp_ptr_[fi] = conv_param_->conv_quant_arg_.filter_quant_args_[fi].zp_;
    }

    /* left shift */
    left_shift_ = reinterpret_cast<int32_t *>(malloc(round_oc * sizeof(int32_t)));
    if (left_shift_ == nullptr) {
      return RET_ERROR;
    }
    memset(left_shift_, 0, round_oc * sizeof(int32_t));
    memcpy(left_shift_, conv_param_->conv_quant_arg_.left_shift_, output_channel * sizeof(int32_t));

    /* right shift */
    right_shift_ = reinterpret_cast<int32_t *>(malloc(round_oc * sizeof(int32_t)));
    if (right_shift_ == nullptr) {
      return RET_ERROR;
    }
    memset(right_shift_, 0, round_oc * sizeof(int32_t));
    memcpy(right_shift_, conv_param_->conv_quant_arg_.right_shift_, output_channel * sizeof(int32_t));

    /* multiplier */
    multiplier_ = reinterpret_cast<int32_t *>(malloc(round_oc * sizeof(int32_t)));
    if (multiplier_ == nullptr) {
      return RET_ERROR;
    }
    memset(multiplier_, 0, round_oc * sizeof(int32_t));
    memcpy(multiplier_, conv_param_->conv_quant_arg_.quant_multiplier_, output_channel * sizeof(int32_t));
  } else {
    right_shift_ = conv_param_->conv_quant_arg_.right_shift_;
    left_shift_ = conv_param_->conv_quant_arg_.left_shift_;
    multiplier_ = conv_param_->conv_quant_arg_.quant_multiplier_;
  }
  return RET_OK;
}

int Convolution1x1Int8CPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();

  /* weight */
  size_t size = support_optimize_ ? UP_ROUND(input_channel, C4NUM) * UP_ROUND(output_channel, C16NUM) * sizeof(int8_t)
                                  : UP_ROUND(input_channel, C16NUM) * UP_ROUND(output_channel, C4NUM) * sizeof(int8_t);
  packed_weight_ = reinterpret_cast<int8_t *>(malloc(size));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 int8 Malloc weight error!";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, size);
  if (support_optimize_) {
    RowMajor2Row4x16MajorInt8(reinterpret_cast<int8_t *>(filter_tensor->MutableData()), packed_weight_, output_channel,
                              input_channel);
  } else {
    RowMajor2Row16x4MajorInt8(reinterpret_cast<int8_t *>(filter_tensor->MutableData()), packed_weight_, output_channel,
                              input_channel);
  }

  size = support_optimize_ ? UP_ROUND(output_channel, C16NUM) : UP_ROUND(output_channel, C4NUM);
  bias_data_ = malloc(size * sizeof(int32_t));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 int8 Malloc bias_ptr_ error!";
    return RET_ERROR;
  }
  memset(bias_data_, 0, size * sizeof(int32_t));
  if (in_tensors_.size() == 3) {
    memcpy(bias_data_, in_tensors_.at(kBiasIndex)->data_c(), output_channel * sizeof(int32_t));
  }

  InitBiasByzp(filter_tensor->data_c(), input_channel, output_channel, size);
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
    memcpy(bias_data_, in_tensors_.at(kBiasIndex)->data_c(), output_channel * sizeof(int32_t));
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
  matmul_param_->row_4_ = UP_ROUND(matmul_param_->row_, C4NUM);
  matmul_param_->deep_4_ = UP_ROUND(matmul_param_->deep_, C4NUM);
  matmul_param_->deep_16_ = UP_ROUND(matmul_param_->deep_, C16NUM);

  int row_pack_count;
  int col_pack_count;

#ifdef ENABLE_ARM32
  row_pack_count = C4NUM;
  col_pack_count = C2NUM;
#else
  if (support_optimize_) {
    row_pack_count = C4NUM;
    col_pack_count = C16NUM;
  } else {
    row_pack_count = C4NUM;
    col_pack_count = C4NUM;
  }
#endif

  /* init input sum size */
  input_sum_size_ = UP_ROUND(matmul_param_->row_, row_pack_count);

  if (pre_trans_input_) {
    input_ptr_ = reinterpret_cast<int8_t *>(malloc(matmul_param_->row_ * matmul_param_->deep_ * sizeof(int8_t)));
    if (input_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 int8 Malloc input_ptr_ error!";
      return RET_MEMORY_FAILED;
    }
    memset(input_ptr_, 0, matmul_param_->row_ * matmul_param_->deep_ * sizeof(int8_t));
  }

  int hw_thread_count = UP_DIV(matmul_param_->row_, row_pack_count);
  int oc_thread_count = UP_DIV(matmul_param_->col_, col_pack_count);
  thread_count_hw_ = MSMIN(op_parameter_->thread_num_, hw_thread_count);
  thread_stride_hw_ = UP_DIV(hw_thread_count, thread_count_hw_);
  thread_count_oc_ = MSMIN(op_parameter_->thread_num_, oc_thread_count);
  thread_stride_oc_ = UP_DIV(oc_thread_count, thread_count_oc_);
  parallel_by_oc_ = oc_thread_count > op_parameter_->thread_num_;

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
  /* deal with pad and stride */
  output_ptr_ = src_output;
  if (pre_trans_input_) {
    Conv1x1InputPack(src_input, input_ptr_, conv_param_, sizeof(int8_t));
  } else {
    input_ptr_ = src_input;
  }
  return;
}

int Convolution1x1Int8CPUKernel::RunArmHw(int task_id) {
  int cur_stride = thread_stride_hw_ * C4NUM;
  int res_stride = matmul_param_->row_ - task_id * thread_stride_hw_ * C4NUM;
  int cur_hw = MSMIN(cur_stride, res_stride);
  if (cur_hw <= 0) {
    return RET_OK;
  }

  int8_t *hw_in = input_ptr_ + task_id * thread_stride_hw_ * C4NUM * conv_param_->input_channel_;
  int8_t *hw_out = output_ptr_ + task_id * thread_stride_hw_ * C4NUM * conv_param_->output_channel_;
  int8_t *hw_packed_in = packed_input_ + task_id * thread_stride_hw_ * C4NUM * matmul_param_->deep_16_;
  int32_t *hw_input_sum = input_sum_ + task_id * thread_stride_hw_ * C4NUM;

  RowMajor2Row16x4MajorInt8(hw_in, hw_packed_in, cur_hw, matmul_param_->deep_);

  if (filter_peroc_) {
    PackInputSum16x4PerLayer(hw_packed_in, hw_input_sum, 1, UP_ROUND(cur_hw, C4NUM), matmul_param_->deep_16_);
  } else {
    PackInputSum16x4PerLayer(hw_packed_in, hw_input_sum, conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_,
                             UP_ROUND(cur_hw, C4NUM), matmul_param_->deep_16_);
  }

  Conv1x1Int8(hw_packed_in, packed_weight_, hw_out, hw_input_sum, reinterpret_cast<int32_t *>(bias_data_), cur_hw,
              matmul_param_->col_, matmul_param_->deep_16_, left_shift_, right_shift_, multiplier_, conv_param_,
              filter_zp_ptr_);
  return RET_OK;
}

int Convolution1x1Int8CPUKernel::RunArm64OptHw(int task_id) {
  int cur_stride = thread_stride_hw_ * C4NUM;
  int res_stride = matmul_param_->row_ - task_id * thread_stride_hw_ * C4NUM;
  int cur_hw = MSMIN(cur_stride, res_stride);
  if (cur_hw <= 0) {
    return RET_OK;
  }
  int8_t *hw_in = input_ptr_ + task_id * thread_stride_hw_ * C4NUM * conv_param_->input_channel_;
  int8_t *hw_out = output_ptr_ + task_id * thread_stride_hw_ * C4NUM * conv_param_->output_channel_;
  int8_t *hw_packed_in = packed_input_ + task_id * thread_stride_hw_ * C4NUM * matmul_param_->deep_4_;
  int32_t *hw_input_sum = input_sum_ + task_id * thread_stride_hw_ * C4NUM;

  if (filter_peroc_) {
    PackInput4x4AndInputSumPert(hw_in, hw_packed_in, hw_input_sum, matmul_param_->deep_, cur_hw, 1);
  } else {
    PackInput4x4AndInputSumPert(hw_in, hw_packed_in, hw_input_sum, matmul_param_->deep_, cur_hw,
                                conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_);
  }

  Conv1x1Int8Opt(hw_packed_in, packed_weight_, hw_out, hw_input_sum, reinterpret_cast<int32_t *>(bias_data_), cur_hw,
                 matmul_param_->col_, matmul_param_->deep_4_, left_shift_, right_shift_, multiplier_, conv_param_,
                 matmul_func_, filter_zp_ptr_);

  return RET_OK;
}

int Convolution1x1Int8CPUKernel::RunArm64OptOc(int task_id) {
  int stride = thread_stride_oc_ * C16NUM;
  int cur_stride = task_id * stride;
  int res_stride = matmul_param_->col_ - cur_stride;
  int cur_oc = MSMIN(stride, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  int32_t *cur_left_shift = filter_peroc_ ? left_shift_ + cur_stride : conv_param_->conv_quant_arg_.left_shift_;
  int32_t *cur_right_shift = filter_peroc_ ? right_shift_ + cur_stride : conv_param_->conv_quant_arg_.right_shift_;
  int32_t *cur_multiplier = filter_peroc_ ? multiplier_ + cur_stride : conv_param_->conv_quant_arg_.quant_multiplier_;
  int32_t *cur_zp = filter_peroc_ ? filter_zp_ptr_ + cur_stride : filter_zp_ptr_;

  Conv1x1Int8Opt(packed_input_, packed_weight_ + cur_stride * matmul_param_->deep_4_, output_ptr_ + cur_stride,
                 input_sum_, reinterpret_cast<int32_t *>(bias_data_) + cur_stride, matmul_param_->row_, cur_oc,
                 matmul_param_->deep_4_, cur_left_shift, cur_right_shift, cur_multiplier, conv_param_, matmul_func_,
                 cur_zp);

  return RET_OK;
}

int Convolution1x1Int8CPUKernel::RunArmOc(int task_id) {
#ifdef ENABLE_ARM32
  int col_tile = C2NUM;
#else
  int col_tile = C4NUM;
#endif
  int stride = thread_stride_oc_ * col_tile;
  int cur_stride = task_id * stride;
  int res_stride = matmul_param_->col_ - cur_stride;
  int cur_oc = MSMIN(stride, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  int32_t *cur_left_shift = filter_peroc_ ? left_shift_ + cur_stride : conv_param_->conv_quant_arg_.left_shift_;
  int32_t *cur_right_shift = filter_peroc_ ? right_shift_ + cur_stride : conv_param_->conv_quant_arg_.right_shift_;
  int32_t *cur_multiplier = filter_peroc_ ? multiplier_ + cur_stride : conv_param_->conv_quant_arg_.quant_multiplier_;
  int32_t *cur_zp = filter_peroc_ ? filter_zp_ptr_ + cur_stride : filter_zp_ptr_;

  Conv1x1Int8(packed_input_, packed_weight_ + cur_stride * matmul_param_->deep_16_, output_ptr_ + cur_stride,
              input_sum_, reinterpret_cast<int32_t *>(bias_data_) + cur_stride, matmul_param_->row_, cur_oc,
              matmul_param_->deep_16_, cur_left_shift, cur_right_shift, cur_multiplier, conv_param_, cur_zp);

  return RET_OK;
}

int Convolution1x1Int8CPUKernel::OcOptPre(int task_id) {
  int cur_stride = thread_stride_hw_ * C4NUM;
  int res_stride = matmul_param_->row_ - task_id * thread_stride_hw_ * C4NUM;
  int cur_hw = MSMIN(cur_stride, res_stride);
  if (cur_hw <= 0) {
    return RET_OK;
  }
  int8_t *hw_in = input_ptr_ + task_id * thread_stride_hw_ * C4NUM * conv_param_->input_channel_;
  int8_t *hw_packed_in = packed_input_ + task_id * thread_stride_hw_ * C4NUM * matmul_param_->deep_4_;
  int32_t *hw_input_sum = input_sum_ + task_id * thread_stride_hw_ * C4NUM;

  if (filter_peroc_) {
    PackInput4x4AndInputSumPert(hw_in, hw_packed_in, hw_input_sum, matmul_param_->deep_, cur_hw, 1);
  } else {
    PackInput4x4AndInputSumPert(hw_in, hw_packed_in, hw_input_sum, matmul_param_->deep_, cur_hw,
                                conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_);
  }
  return RET_OK;
}

int Convolution1x1Int8CPUKernel::Run() {
  int error_code = InitRunBuf();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv1x1 int8 InitRunBuf error_code[" << error_code << "]";
    FreeRunBuf();
    return RET_ERROR;
  }

  int8_t *src_in = reinterpret_cast<int8_t *>(in_tensors_[0]->data_c());
  int8_t *src_out = reinterpret_cast<int8_t *>(out_tensors_[0]->data_c());

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    Pre1x1Trans(src_in + batch_index * conv_param_->input_h_ * conv_param_->input_w_ * conv_param_->input_channel_,
                src_out + batch_index * matmul_param_->row_ * matmul_param_->col_);
    if (parallel_by_oc_) {
      /* input transpose and input sum */
      if (support_optimize_) {
        ParallelLaunch(this->context_->thread_pool_, Convolution1x1Int8OcOptPre, this, thread_count_hw_);
      } else {
        RowMajor2Row16x4MajorInt8(input_ptr_, packed_input_, matmul_param_->row_, matmul_param_->deep_);
        if (filter_peroc_) {
          PackInputSum16x4PerLayer(packed_input_, input_sum_, 1, matmul_param_->row_4_, matmul_param_->deep_16_);
        } else {
          PackInputSum16x4PerLayer(packed_input_, input_sum_, conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_,
                                   matmul_param_->row_4_, matmul_param_->deep_16_);
        }
      }
      /* matmul parallel by oc */
      error_code = ParallelLaunch(this->context_->thread_pool_, Convolution1x1Int8OcRun, this, thread_count_oc_);
    } else {
      /* matmul parallel by hw */
      error_code = ParallelLaunch(this->context_->thread_pool_, Convolution1x1Int8HwRun, this, thread_count_hw_);
    }
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "ParallelLaunch run error error_code[" << error_code << "]";
      FreeRunBuf();
      return error_code;
    }
  }

  FreeRunBuf();
  return RET_OK;
}
}  // namespace mindspore::kernel
