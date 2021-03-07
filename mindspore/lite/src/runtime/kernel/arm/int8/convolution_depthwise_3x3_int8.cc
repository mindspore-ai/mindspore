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

#include "src/runtime/kernel/arm/int8/convolution_depthwise_3x3_int8.h"
#include "include/errorcode.h"
#include "nnacl/int8/conv_depthwise_int8.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
ConvolutionDepthwise3x3Int8CPUKernel::~ConvolutionDepthwise3x3Int8CPUKernel() {
  if (sliding_ != nullptr) {
    delete sliding_;
    sliding_ = nullptr;
  }
  if (packed_weight_ != nullptr) {
    free(packed_weight_);
    packed_weight_ = nullptr;
  }
  FreeQuantParam();
}

int ConvolutionDepthwise3x3Int8CPUKernel::InitWeightBias() {
  // init weight, int8 -> int16
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto origin_weight = reinterpret_cast<int8_t *>(weight_tensor->MutableData());
  int channel = weight_tensor->Batch();
  if (channel % 8 != 0) {
    MS_LOG(ERROR) << "ConvolutionDepthwise3x3Int8CPUKernel doesn't support channel " << channel;
    return RET_ERROR;
  }
  int pack_weight_size = channel * weight_tensor->Height() * weight_tensor->Width();
  auto tmp_weight = reinterpret_cast<int8_t *>(malloc(pack_weight_size * sizeof(int8_t)));
  if (tmp_weight == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  PackNCHWToNHWCInt8(origin_weight, tmp_weight, 1, weight_tensor->Height() * weight_tensor->Width(),
                     weight_tensor->Batch());

  packed_weight_ = reinterpret_cast<int16_t *>(malloc(pack_weight_size * sizeof(int16_t)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    free(tmp_weight);
    return RET_ERROR;
  }
  bool filter_per_channel = conv_param_->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL;
  if (filter_per_channel) {
    for (int i = 0; i < weight_tensor->Height() * weight_tensor->Width(); i++) {
      for (int c = 0; c < channel; c++) {
        int weight_zp = conv_param_->conv_quant_arg_.filter_quant_args_[c].zp_;
        packed_weight_[i * channel + c] = (int16_t)(tmp_weight[i * channel + c] - weight_zp);
      }
    }
  } else {
    int weight_zp = conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_;
    for (int i = 0; i < weight_tensor->ElementsNum(); i++) {
      packed_weight_[i] = (int16_t)(tmp_weight[i] - weight_zp);
    }
  }
  free(tmp_weight);

  bias_data_ = reinterpret_cast<int32_t *>(malloc(channel * sizeof(int32_t)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, channel * sizeof(int32_t));
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    auto ori_bias = reinterpret_cast<int32_t *>(bias_tensor->MutableData());
    memcpy(bias_data_, ori_bias, bias_tensor->ElementsNum() * sizeof(int32_t));
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3Int8CPUKernel::Init() {
  sliding_ = new (std::nothrow) SlidingWindowParam;
  if (sliding_ == nullptr) {
    MS_LOG(ERROR) << "new sliding window param.";
    return RET_ERROR;
  }
  auto ret = ConvolutionBaseCPUKernel::SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
  }
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise int8 InitWeightBias error!";
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDepthwise3x3Int8CPUKernel::ReSize() {
  ConvolutionBaseCPUKernel::Init();
  InitSlidingParamConvDw(sliding_, conv_param_, conv_param_->input_channel_);
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  return RET_OK;
}

int ConvolutionDepthwise3x3Int8CPUKernel::Execute(int task_id) {
  auto buffer = buffer_ + 64 * 10 * 10 * task_id;
  ConvDw3x3Int8(output_ptr_, buffer, input_ptr_, packed_weight_, reinterpret_cast<int32_t *>(bias_data_), conv_param_,
                sliding_, task_id);
  return RET_OK;
}

int ConvDw3x3Int8Run(void *cdata, int task_id) {
  auto conv_dw_int8 = reinterpret_cast<ConvolutionDepthwise3x3Int8CPUKernel *>(cdata);
  auto ret = conv_dw_int8->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwise3x3Int8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3Int8CPUKernel::InitBuffer() {
  int buffer_size = 64 * 10 * 10 * conv_param_->thread_num_;
  buffer_ = reinterpret_cast<int8_t *>(context_->allocator->Malloc(buffer_size * sizeof(int8_t)));
  if (buffer_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3Int8CPUKernel::Run() {
  auto ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise int8 ReSize error!";
    return ret;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  input_ptr_ = reinterpret_cast<int8_t *>(input_tensor->MutableData());

  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_ptr_ = reinterpret_cast<int8_t *>(output_tensor->MutableData());

  if (sliding_->top_ > 0 || sliding_->bottom_ < conv_param_->output_h_ || sliding_->left_ > 0 ||
      sliding_->right_ < conv_param_->output_w_) {
    ConvDw3x3Int8Pad(output_ptr_, input_ptr_, packed_weight_, reinterpret_cast<int32_t *>(bias_data_), conv_param_,
                     sliding_);
  }
  ret = ParallelLaunch(this->context_->thread_pool_, ConvDw3x3Int8Run, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    context_->allocator->Free(buffer_);
    MS_LOG(ERROR) << "ConvDwInt8Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  context_->allocator->Free(buffer_);
  return RET_OK;
}
}  // namespace mindspore::kernel
