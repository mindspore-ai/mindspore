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

#include "src/runtime/kernel/arm/fp16/convolution_winograd_fp16.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ConvolutionWinogradFP16CPUKernel::WinogradFilterTransformFp16(const float16_t *weight_data, const float *matrix_g,
                                                                  const float *matrix_gt, int oc_block) {
  if (oc_block == 0) {
    MS_LOG(ERROR) << "Divide by zero";
    return RET_ERROR;
  }

  return WinogradWeightTransformFp16(weight_data, reinterpret_cast<float16_t *>(packed_weight_), matrix_g, matrix_gt,
                                     oc_block, input_unit_, kernel_unit_, conv_param_->input_channel_,
                                     conv_param_->output_channel_, true);
}

int ConvolutionWinogradFP16CPUKernel::MallocWeightBiasData() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  int in_channel = weight_tensor->Channel();
  int out_channel = weight_tensor->Batch();
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;
  int oc_block_num = UP_DIV(out_channel, col_tile_);
  // init weight
  auto trans_matrix_data_size = input_unit_ * input_unit_ * in_channel * oc_block_num * col_tile_ * sizeof(float16_t);
  if (!op_parameter_->is_train_session_) {
    if (packed_weight_ == nullptr) {
      packed_weight_ = malloc(trans_matrix_data_size);
      if (packed_weight_ == nullptr) {
        MS_LOG(ERROR) << "malloc packed_weight_ failed.";
        return RET_ERROR;
      }
    }
    memset(packed_weight_, 0, trans_matrix_data_size);
  }

  float matrix_a[64];
  float matrix_at[64];
  float matrix_b[64];
  float matrix_bt[64];
  float coef = 1.0f;
  if (input_unit_ == 8) {
    coef = 0.5f;
  }
  auto ret =
    CookToomFilter(matrix_a, matrix_at, matrix_b, matrix_bt, matrix_g_, matrix_gt_, coef, output_unit_, kernel_unit_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "get matrix g from CookToomFilter failed.";
    return ret;
  }

  if (bias_data_ == nullptr) {
    bias_data_ = malloc(oc_block_num * col_tile_ * sizeof(float16_t));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias_data_ failed.";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, oc_block_num * col_tile_ * sizeof(float16_t));
  return RET_OK;
}

void ConvolutionWinogradFP16CPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  void *weight_origin = (op_parameter_->is_train_session_) ? weight_tensor->data() : origin_weight_;
  MS_ASSERT(weight_origin != nullptr);
  WinogradFilterTransformFp16(reinterpret_cast<float16_t *>(weight_origin), matrix_g_, matrix_gt_, col_tile_);
}

int ConvolutionWinogradFP16CPUKernel::InitTmpBuffer() {
  int channel_out = conv_param_->output_channel_;
  size_t tile_buffer_size =
    thread_count_ * row_tile_ * input_unit_ * input_unit_ * conv_param_->input_channel_ * sizeof(float16_t);
  trans_input_ = reinterpret_cast<float16_t *>(ctx_->allocator->Malloc(tile_buffer_size));
  if (trans_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc trans_input_ failed.";
    return RET_ERROR;
  }

  gemm_out_ = reinterpret_cast<float16_t *>(ctx_->allocator->Malloc(
    thread_count_ * row_tile_ * input_unit_ * input_unit_ * UP_ROUND(channel_out, C8NUM) * sizeof(float16_t)));
  if (gemm_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc gemm_out_ failed.";
    return RET_ERROR;
  }

  tmp_data_ = reinterpret_cast<float16_t *>(
    ctx_->allocator->Malloc(thread_count_ * C8NUM * input_unit_ * input_unit_ * sizeof(float16_t)));
  if (tmp_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_data_ failed.";
    return RET_ERROR;
  }

  col_buffer_ = reinterpret_cast<float16_t *>(
    ctx_->allocator->Malloc(thread_count_ * row_tile_ * conv_param_->input_channel_ * sizeof(float16_t)));
  if (col_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc col_buffer_ failed.";
    return RET_ERROR;
  }

  tmp_buffer_address_list_[0] = trans_input_;
  tmp_buffer_address_list_[1] = gemm_out_;
  tmp_buffer_address_list_[2] = tmp_data_;
  tmp_buffer_address_list_[3] = col_buffer_;
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::ConfigInputOutput() {
  in_func_ = GetInputTransFp16Func(input_unit_);
  if (in_func_ == nullptr) {
    MS_LOG(ERROR) << "in_func_ is null.";
    return RET_ERROR;
  }
  out_func_ = GetOutputTransFp16Func(input_unit_, output_unit_, conv_param_->act_type_);
  if (out_func_ == nullptr) {
    MS_LOG(ERROR) << "out_func_ is null.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 2);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  UpdateOriginWeightAndBias();
  col_tile_ = C8NUM;
#ifdef ENABLE_ARM64
  row_tile_ = C16NUM;
#else
  row_tile_ = C12NUM;
#endif
  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(weight_tensor);
    int in_channel = weight_tensor->Channel();
    int out_channel = weight_tensor->Batch();
    int oc_block_num = UP_DIV(out_channel, col_tile_);
    auto trans_matrix_data_size = input_unit_ * input_unit_ * in_channel * oc_block_num * col_tile_ * sizeof(float16_t);
    set_workspace_size(trans_matrix_data_size);
  }
  kernel_unit_ = conv_param_->kernel_h_;
  input_unit_ = output_unit_ + kernel_unit_ - 1;
  conv_param_->input_unit_ = input_unit_;
  conv_param_->output_unit_ = output_unit_;

  auto ret = InitConvWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::AdjustNumberOfThread() {
  auto out_tensor = out_tensors_.front();
  int cal_plane = UP_DIV(out_tensor->Height(), output_unit_) * UP_DIV(out_tensor->Width(), output_unit_);
  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(cal_plane, C8NUM));
  if (thread_count_ <= 0) {
    MS_LOG(ERROR) << "thread_count_ must be greater than 0!";
    return RET_ERROR;
  }
  conv_param_->thread_num_ = thread_count_;
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }
  ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return ret;
  }
  ret = ConfigInputOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConfigInputOutput failed.";
    return ret;
  }
  ret = AdjustNumberOfThread();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AdjustNumberOfThread failed.";
    return ret;
  }
  conv_param_->out_format_ = out_tensors_[0]->format();
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::RunImpl(int task_id) {
  auto input_ptr = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data());
  auto output_ptr = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data());
  MS_ASSERT(input_ptr != nullptr);
  MS_ASSERT(output_ptr != nullptr);
  if (input_ptr == nullptr || output_ptr == nullptr) {
    MS_LOG(ERROR) << "Convolution Winograd Fp16 get null tensor data!";
    return RET_ERROR;
  }
  ConvWinogardFp16(input_ptr, reinterpret_cast<float16_t *>(packed_weight_),
                   reinterpret_cast<const float16_t *>(bias_data_), output_ptr, tmp_buffer_address_list_, task_id,
                   conv_param_, in_func_, out_func_);
  return RET_OK;
}

static int ConvolutionWinogradFp16Impl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv = reinterpret_cast<ConvolutionWinogradFP16CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionWinograd Fp16 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }
  ret = ParallelLaunch(this->ms_context_, ConvolutionWinogradFp16Impl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv winograd error error_code[" << ret << "]";
  }
  FreeTmpBuffer();
  return ret;
}
}  // namespace mindspore::kernel
