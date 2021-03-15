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

#include "src/runtime/kernel/arm/fp32/convolution_depthwise_3x3_fp32.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
ConvolutionDepthwise3x3CPUKernel::~ConvolutionDepthwise3x3CPUKernel() {
  if (packed_weight_ != nullptr) {
    free(packed_weight_);
    packed_weight_ = nullptr;
  }
}

int ConvolutionDepthwise3x3CPUKernel::InitWeightBias() {
  // init weight: k, h, w, c; k == group == output_channel, c == 1
  auto weight_tensor = in_tensors_[kWeightIndex];
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->MutableData());
  int channel = weight_tensor->Batch();
  int c4 = UP_ROUND(channel, C4NUM);
  int pack_weight_size = c4 * C12NUM;

  if (packed_weight_ == nullptr) {
    packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  PackWeightConvDw3x3Fp32(origin_weight, packed_weight_, channel);

  if (bias_data_ == nullptr) {
    bias_data_ = reinterpret_cast<float *>(malloc(c4 * sizeof(float)));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, c4 * sizeof(float));
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_[kBiasIndex];
    auto ori_bias = reinterpret_cast<float *>(bias_tensor->MutableData());
    memcpy(bias_data_, ori_bias, bias_tensor->ElementsNum() * sizeof(float));
  }

  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::Init() {
  auto ret = InitWeightBias();
  if (ret != 0) {
    MS_LOG(ERROR) << "Convolution depthwise 3x3 fp32 InitWeightBias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDepthwise3x3CPUKernel::ReSize() {
  ConvolutionBaseCPUKernel::Init();
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::Execute(int task_id) {
  int units = UP_DIV(conv_param_->output_w_, C2NUM);  // F(2, 3) contains 2 conv units
  int c4 = UP_ROUND(conv_param_->input_channel_, C4NUM);
  auto buffer = buffer_ + C12NUM * c4 * units * task_id;
  int step_oh = UP_DIV(conv_param_->output_h_, conv_param_->thread_num_);
  int start_oh = step_oh * task_id;
  int end_oh = MSMIN(start_oh + step_oh, conv_param_->output_h_);
  ConvDw3x3(output_ptr_, buffer, input_ptr_, packed_weight_, reinterpret_cast<float *>(bias_data_), conv_param_,
            start_oh, end_oh);
  return RET_OK;
}

int ConvDw3x3Run(void *cdata, int task_id) {
  auto conv_dw = reinterpret_cast<ConvolutionDepthwise3x3CPUKernel *>(cdata);
  auto ret = conv_dw->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwise3x3Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::Run() {
  int units = UP_DIV(conv_param_->output_w_, C2NUM);  // F(2, 3) contains 2 conv units
  int c4 = UP_ROUND(conv_param_->input_channel_, C4NUM);
  int buffer_size = units * c4 * C12NUM * conv_param_->thread_num_;
  buffer_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(buffer_size * sizeof(float)));
  if (buffer_ == nullptr) {
    MS_LOG(ERROR) << "ConvDw3x3Run failed to allocate buffer";
    return RET_MEMORY_FAILED;
  }

  if (IsTrain() && is_trainable()) {
    InitWeightBias();
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  input_ptr_ = reinterpret_cast<float *>(input_tensor->data_c());

  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_ptr_ = reinterpret_cast<float *>(output_tensor->data_c());

  auto ret = ParallelLaunch(this->context_->thread_pool_, ConvDw3x3Run, this, conv_param_->thread_num_);
  ctx_->allocator->Free(buffer_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDw3x3Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::Eval() {
  LiteKernel::Eval();
  if (is_trainable()) {
    InitWeightBias();
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
#endif
