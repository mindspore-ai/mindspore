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

#include "src/runtime/kernel/arm/fp32/convolution_depthwise_fp32.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
ConvolutionDepthwiseCPUKernel::~ConvolutionDepthwiseCPUKernel() {
  if (packed_weight_ != nullptr) {
    free(packed_weight_);
    packed_weight_ = nullptr;
  }
}

int ConvolutionDepthwiseCPUKernel::InitWeightBias() {
  // init weight: k, h, w, c; k == group == output_channel, c == 1
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->MutableData());
  int channel = weight_tensor->Batch();
  int pack_weight_size = weight_tensor->Batch() * weight_tensor->Height() * weight_tensor->Width();
  if (pack_weight_size >= std::numeric_limits<int>::max() / static_cast<int>(sizeof(float))) {
    MS_LOG(ERROR) << "pack_weight_size is invalid, pack_weight_size: " << pack_weight_size;
    return RET_ERROR;
  }
  packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  PackWeightKHWToHWKFp32(origin_weight, packed_weight_, weight_tensor->Height() * weight_tensor->Width(), channel);

  bias_data_ = reinterpret_cast<float *>(malloc(channel * sizeof(float)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }

  memset(bias_data_, 0, channel * sizeof(float));
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_[kBiasIndex];
    auto ori_bias = reinterpret_cast<float *>(bias_tensor->MutableData());
    memcpy(bias_data_, ori_bias, bias_tensor->ElementsNum() * sizeof(float));
  }

  return RET_OK;
}

int ConvolutionDepthwiseCPUKernel::Init() {
  auto ret = InitWeightBias();
  if (ret != 0) {
    MS_LOG(ERROR) << "Convolution depthwise fp32 InitWeightBias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDepthwiseCPUKernel::ReSize() {
  ConvolutionBaseCPUKernel::Init();
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  return RET_OK;
}

int ConvolutionDepthwiseCPUKernel::Execute(int task_id) {
  ConvDw(output_ptr_, input_ptr_, packed_weight_, reinterpret_cast<float *>(bias_data_), conv_param_, task_id);
  return RET_OK;
}

int ConvDwRun(void *cdata, int task_id) {
  auto conv_dw = reinterpret_cast<ConvolutionDepthwiseCPUKernel *>(cdata);
  auto ret = conv_dw->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseCPUKernel::Run() {
  if (IsTrain() && is_trainable()) {
    PackWeight();
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  input_ptr_ = reinterpret_cast<float *>(input_tensor->MutableData());

  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_ptr_ = reinterpret_cast<float *>(output_tensor->MutableData());

  auto ret = ParallelLaunch(this->context_->thread_pool_, ConvDwRun, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwRun error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

void ConvolutionDepthwiseCPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->MutableData());
  PackWeightKHWToHWKFp32(origin_weight, packed_weight_, weight_tensor->Height() * weight_tensor->Width(),
                         weight_tensor->Batch());
}

int ConvolutionDepthwiseCPUKernel::Eval() {
  LiteKernel::Eval();
  if (is_trainable()) {
    PackWeight();
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
