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

#include "src/runtime/kernel/arm/fp32/convolution_winograd_fp32.h"
#include "nnacl/fp32/conv_winograd_fp32.h"
#include "nnacl/pack.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ConvolutionWinogradCPUKernel::WinogradFilterTransform(const float *weight_data, float *matrix_g, float *matrix_gt,
                                                          int oc_block) {
  if (oc_block == 0) {
    MS_LOG(ERROR) << "Divide by zero";
    return RET_ERROR;
  }

  return WinogradWeightTransform(weight_data, trans_weight_, matrix_g, matrix_gt, oc_block, input_unit_, kernel_unit_,
                                 conv_param_->input_channel_, conv_param_->output_channel_, true);
}

int ConvolutionWinogradCPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int in_channel = filter_tensor->Channel();
  int out_channel = filter_tensor->Batch();
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;

  // set data
  auto trans_matrix_data_size =
    input_unit_ * input_unit_ * in_channel * UP_ROUND(out_channel, oc_block_) * sizeof(float);
  if (trans_weight_ == nullptr) {
    trans_weight_ = reinterpret_cast<float *>(malloc(trans_matrix_data_size));
    if (trans_weight_ == nullptr) {
      MS_LOG(ERROR) << "malloc matrix_buffer failed.";
      return RET_MEMORY_FAILED;
    }
  }
  memset(trans_weight_, 0, trans_matrix_data_size);

  float matrix_g[64];
  float matrix_gt[64];
  float matrix_a[64];
  float matrix_at[64];
  float matrix_b[64];
  float matrix_bt[64];
  float coef = 1.0f;
  if (input_unit_ == 8) {
    coef = 0.5f;
  }
  auto ret =
    CookToomFilter(matrix_a, matrix_at, matrix_b, matrix_bt, matrix_g, matrix_gt, coef, output_unit_, kernel_unit_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "get matrix g from CookToomFilter failed.";
    return ret;
  }
  ret = WinogradFilterTransform(origin_weight_, matrix_g, matrix_gt, oc_block_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "winograd filter transform failed.";
    return ret;
  }

  // init bias
  size_t new_bias_size = UP_ROUND(out_channel, C4NUM) * sizeof(float);
  if (bias_data_ == nullptr) {
    bias_data_ = reinterpret_cast<float *>(malloc(new_bias_size));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias_data_ failed.";
      return RET_MEMORY_FAILED;
    }
  }
  if (in_tensors_.size() == kInputSize2) {
    size_t origin_size = out_channel * sizeof(float);
    memcpy(bias_data_, origin_bias_, origin_size);
    memset(reinterpret_cast<float *>(bias_data_) + out_channel, 0, new_bias_size - origin_size);
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
    memset(bias_data_, 0, new_bias_size);
  }
  return RET_OK;
}

int ConvolutionWinogradCPUKernel::InitTmpBuffer() {
  MS_ASSERT(ctx_->allocator != nullptr);
  size_t tile_buffer_size =
    thread_count_ * tile_num_ * input_unit_ * input_unit_ * conv_param_->input_channel_ * sizeof(float);
  trans_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(tile_buffer_size));
  if (trans_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc trans_input_ failed.";
    return RET_MEMORY_FAILED;
  }

  int oc8 = UP_ROUND(conv_param_->output_channel_, C8NUM);
  gemm_out_ = reinterpret_cast<float *>(
    ctx_->allocator->Malloc(thread_count_ * tile_num_ * input_unit_ * input_unit_ * oc8 * sizeof(float)));
  if (gemm_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc gemm_out_ failed.";
    return RET_ERROR;
  }

  tmp_data_ = reinterpret_cast<float *>(
    ctx_->allocator->Malloc(thread_count_ * C4NUM * input_unit_ * input_unit_ * sizeof(float)));
  if (tmp_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_data_ failed.";
    return RET_MEMORY_FAILED;
  }

  col_buffer_ = reinterpret_cast<float *>(
    ctx_->allocator->Malloc(thread_count_ * tile_num_ * conv_param_->input_channel_ * sizeof(float)));
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

int ConvolutionWinogradCPUKernel::ConfigInputOutput() {
  in_func_ = GetInputTransFunc(input_unit_);
  if (in_func_ == nullptr) {
    MS_LOG(ERROR) << "in_func_ is null.";
    return RET_ERROR;
  }
  out_func_ = GetOutputTransFunc(input_unit_, output_unit_, conv_param_->act_type_);
  if (out_func_ == nullptr) {
    MS_LOG(ERROR) << "out_func_ is null.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradCPUKernel::Init() {
  tile_num_ = C12NUM;
#ifdef ENABLE_AVX
  oc_block_ = C16NUM;
#else
  oc_block_ = C8NUM;
#endif
  kernel_unit_ = conv_param_->kernel_h_;
  input_unit_ = output_unit_ + kernel_unit_ - 1;
  conv_param_->input_unit_ = input_unit_;
  conv_param_->output_unit_ = output_unit_;

  auto ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradCPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }
  ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv base init failed.";
    return ret;
  }
  ret = ConfigInputOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConfigInputOutput failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradCPUKernel::RunImpl(int task_id) {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto ori_input_data = reinterpret_cast<float *>(input_tensor->data_c());
  auto output_data = reinterpret_cast<float *>(out_tensors_.front()->data_c());
  ConvWinogardFp32(ori_input_data, trans_weight_, reinterpret_cast<const float *>(bias_data_), output_data,
                   tmp_buffer_address_list_, task_id, conv_param_, in_func_, out_func_);
  return RET_OK;
}

int ConvolutionWinogradImpl(void *cdata, int task_id) {
  auto conv = reinterpret_cast<ConvolutionWinogradCPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionWinograd Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradCPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  if (IsTrain() && is_trainable()) {
    InitWeightBias();
  }

  ret = ParallelLaunch(this->context_->thread_pool_, ConvolutionWinogradImpl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv winograd error error_code[" << ret << "]";
  }

  FreeTmpBuffer();
  return ret;
}

int ConvolutionWinogradCPUKernel::Eval() {
  LiteKernel::Eval();
  if (is_trainable()) {
    InitWeightBias();
  }
  return RET_OK;
}

}  // namespace mindspore::kernel
