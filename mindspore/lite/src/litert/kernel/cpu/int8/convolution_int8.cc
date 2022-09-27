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

#include "src/litert/kernel/cpu/int8/convolution_int8.h"
#include "include/errorcode.h"
#include "nnacl/int8/conv_int8.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#ifdef ENABLE_ARM64
#include "src/litert/kernel/cpu/int8/opt_op_handler.h"
#endif

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
void ConvolutionInt8CPUKernel::CheckSupportOptimize() {
  tile_num_ = 8;
#ifdef ENABLE_ARM32
  tile_num_ = 4;
  support_optimize_ = false;
#endif

#if defined(ENABLE_ARM64)
#if !defined(SUPPORT_NNIE) && !defined(SUPPORT_34XX) && !defined(MACHINE_LINUX_ARM64)
  if (mindspore::lite::IsSupportSDot()) {
    matmul_func_ = MatMulRInt8_optimize_handler;
    support_optimize_ = true;
  } else {
#endif
    tile_num_ = 4;
    support_optimize_ = false;
#if !defined(SUPPORT_NNIE) && !defined(SUPPORT_34XX) && !defined(MACHINE_LINUX_ARM64)
  }
#endif
#endif
  conv_param_->tile_num_ = tile_num_;
}

int ConvolutionInt8CPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  if (input_channel <= 0) {
    MS_LOG(ERROR) << "get channel from filter tensor failed.";
    return RET_ERROR;
  }
  auto output_channel = filter_tensor->Batch();
  if (output_channel <= 0) {
    MS_LOG(ERROR) << "get batch from filter tensor failed.";
    return RET_ERROR;
  }
  MS_CHECK_INT_MUL_NOT_OVERFLOW(filter_tensor->Height(), filter_tensor->Width(), RET_ERROR);
  int kernel_plane = filter_tensor->Height() * filter_tensor->Width();
  conv_param_->input_channel_ = input_channel;
  conv_param_->output_channel_ = output_channel;
  int up_round_deep;
  int up_round_oc;
#ifdef ENABLE_ARM32
  up_round_oc = UP_ROUND(output_channel, C2NUM);
  up_round_deep = UP_ROUND(kernel_plane * input_channel, C16NUM);
#else
  if (support_optimize_) {
    up_round_oc = UP_ROUND(output_channel, C8NUM);
    up_round_deep = UP_ROUND(kernel_plane * input_channel, C4NUM);
  } else {
    up_round_oc = UP_ROUND(output_channel, C4NUM);
    up_round_deep = UP_ROUND(kernel_plane * input_channel, C16NUM);
  }
#endif
  MS_CHECK_INT_MUL_NOT_OVERFLOW(up_round_oc, up_round_deep, RET_ERROR);
  int pack_weight_size = up_round_oc * up_round_deep;
  size_t bias_size = up_round_oc * sizeof(int32_t);
  int32_t input_zp = conv_param_->conv_quant_arg_.input_quant_args_[0].zp_;

  // init weight
  auto origin_weight = reinterpret_cast<int8_t *>(in_tensors_.at(kWeightIndex)->data());
  CHECK_NULL_RETURN(origin_weight);
  packed_weight_sub_ = reinterpret_cast<int8_t *>(malloc(pack_weight_size));
  if (packed_weight_sub_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_weight_sub_ failed.";
    return RET_ERROR;
  }
  (void)memset(packed_weight_sub_, 0, pack_weight_size);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(input_channel, kernel_plane, RET_ERROR);
#ifdef ENABLE_ARM32
  RowMajor2Row2x16MajorInt8(origin_weight, packed_weight_sub_, output_channel, input_channel * kernel_plane);
#else
  if (support_optimize_) {
    RowMajor2Row8x4MajorInt8(origin_weight, packed_weight_sub_, output_channel, input_channel * kernel_plane);
  } else {
    RowMajor2Row16x4MajorInt8(origin_weight, packed_weight_sub_, output_channel, input_channel * kernel_plane);
  }
#endif

  // init bias
  bias_data_ = reinterpret_cast<int32_t *>(malloc(bias_size));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_data_ failed.";
    return RET_ERROR;
  }
  (void)memset(bias_data_, 0, bias_size);
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<int32_t *>(in_tensors_.at(kBiasIndex)->data());
    CHECK_NULL_RETURN(ori_bias);
    (void)memcpy(bias_data_, ori_bias, static_cast<size_t>(output_channel) * sizeof(int32_t));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  auto *bias_data = reinterpret_cast<int32_t *>(bias_data_);
  bool filter_peroc = static_cast<bool>(conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL);
  if (filter_peroc) {
    filter_zp_ptr_ = reinterpret_cast<int32_t *>(malloc(output_channel * sizeof(int32_t)));
    if (filter_zp_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
  }
  for (int oc = 0; oc < output_channel; oc++) {
    int32_t filter_zp = conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_;
    if (filter_peroc) {
      filter_zp = conv_param_->conv_quant_arg_.filter_quant_args_[oc].zp_;
      filter_zp_ptr_[oc] = filter_zp;
    }
    int32_t weight_sum_value = up_round_deep * filter_zp;
    for (int i = 0; i < kernel_plane * input_channel; i++) {
      weight_sum_value += origin_weight[oc * kernel_plane * input_channel + i] - filter_zp;
    }
    bias_data[oc] += filter_zp * input_zp * up_round_deep - weight_sum_value * input_zp;
  }

  size_t input_sum_size;
  if (conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL) {
    input_sum_size = static_cast<size_t>(up_round_oc * tile_num_ * thread_count_) * sizeof(int32_t);
  } else {
    input_sum_size = static_cast<size_t>(tile_num_ * thread_count_) * sizeof(int32_t);
  }
  input_sum_ = reinterpret_cast<int32_t *>(malloc(input_sum_size));
  if (input_sum_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_sum_ failed.";
    return RET_ERROR;
  }
  (void)memset(input_sum_, 0, input_sum_size);
  return RET_OK;
}

int ConvolutionInt8CPUKernel::InitTmpBuffer() {
  MS_ASSERT(ctx_->allocator != nullptr);
  int kernel_plane = conv_param_->kernel_h_ * conv_param_->kernel_w_;
  int tmp_size;
  if (support_optimize_) {
    tmp_size = UP_ROUND(kernel_plane * conv_param_->input_channel_, C4NUM);
  } else {
    tmp_size = UP_ROUND(kernel_plane * conv_param_->input_channel_, C16NUM);
  }
  matmul_packed_input_ = reinterpret_cast<int8_t *>(
    ctx_->allocator->Malloc(thread_count_ * tile_num_ * kernel_plane * conv_param_->input_channel_));
  if (matmul_packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc matmul_packed_input_ failed.";
    return RET_ERROR;
  }
  packed_input_ = reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(tmp_size * thread_count_ * tile_num_));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_input_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(in_tensors_[1]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      in_tensors_[1]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", input1 data_type is "
                  << in_tensors_[1]->data_type() << ", output data_type is " << out_tensors_[0]->data_type();
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
    MS_LOG(ERROR) << "Initialization for optimized int8 conv failed.";
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionInt8CPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionInt8CPUKernel::RunImpl(int task_id) {
  auto ori_input_data = reinterpret_cast<int8_t *>(in_tensors_.at(kInputIndex)->data());
  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(kOutputIndex)->data());
  ConvInt8(ori_input_data, packed_input_, matmul_packed_input_, packed_weight_sub_,
           reinterpret_cast<int32_t *>(bias_data_), output_addr, filter_zp_ptr_, input_sum_, task_id, conv_param_,
           matmul_func_, support_optimize_);
  return RET_OK;
}

int ConvolutionInt8Impl(void *cdata, int task_id, float, float) {
  auto conv = reinterpret_cast<ConvolutionInt8CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution Int8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionInt8CPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }

  int error_code = ParallelLaunch(this->ms_context_, ConvolutionInt8Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv int8 error error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  FreeTmpBuffer();
  return RET_OK;
}
}  // namespace mindspore::kernel
