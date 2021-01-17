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
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DepthwiseConv2D;

namespace mindspore::kernel {
ConvolutionDepthwise3x3CPUKernel::~ConvolutionDepthwise3x3CPUKernel() {
  if (packed_weight_ != nullptr) {
    free(packed_weight_);
    packed_weight_ = nullptr;
  }
  if (sliding_ != nullptr) {
    delete sliding_;
    sliding_ = nullptr;
  }
}

int ConvolutionDepthwise3x3CPUKernel::InitWeightBias() {
  // init weight: k, h, w, c; k == group == output_channel, c == 1
  auto weight_tensor = in_tensors_[kWeightIndex];
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->MutableData());
  int channel = weight_tensor->Batch();
  int pack_weight_size = weight_tensor->Batch() * weight_tensor->Height() * weight_tensor->Width();

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

int ConvolutionDepthwise3x3CPUKernel::Init() {
  sliding_ = new (std::nothrow) SlidingWindowParam;
  if (sliding_ == nullptr) {
    MS_LOG(ERROR) << "new sliding window param failed.";
    return RET_ERROR;
  }
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
  InitSlidingParamConvDw(sliding_, conv_param_, conv_param_->input_channel_);
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::Execute(int task_id) {
  auto buffer = buffer_ + 64 * 10 * 10 * task_id;
  ConvDw3x3(output_ptr_, buffer, input_ptr_, packed_weight_, reinterpret_cast<float *>(bias_data_), conv_param_,
            sliding_, task_id);
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

int ConvolutionDepthwise3x3CPUKernel::InitBuffer() {
  int buffer_size = 64 * 10 * 10 * conv_param_->thread_num_;
  buffer_ = reinterpret_cast<float *>(context_->allocator->Malloc(buffer_size * sizeof(float)));
  if (buffer_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::Run() {
  auto ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise int8 ReSize error!";
    return ret;
  }

  if (IsTrain()) {
    PackWeight();
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  input_ptr_ = reinterpret_cast<float *>(input_tensor->data_c());

  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_ptr_ = reinterpret_cast<float *>(output_tensor->data_c());

  if (sliding_->top_ > 0 || sliding_->bottom_ < conv_param_->output_h_ || sliding_->left_ > 0 ||
      sliding_->right_ < conv_param_->output_w_) {
    ConvDw3x3Pad(output_ptr_, input_ptr_, packed_weight_, reinterpret_cast<float *>(bias_data_), conv_param_, sliding_);
  }
  ret = ParallelLaunch(this->context_->thread_pool_, ConvDw3x3Run, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    context_->allocator->Free(buffer_);
    MS_LOG(ERROR) << "ConvDw3x3Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  context_->allocator->Free(buffer_);
  return RET_OK;
}

void ConvolutionDepthwise3x3CPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->MutableData());
  PackWeightKHWToHWKFp32(origin_weight, packed_weight_, weight_tensor->Height() * weight_tensor->Width(),
                         weight_tensor->Batch());
}

int ConvolutionDepthwise3x3CPUKernel::Eval() {
  LiteKernel::Eval();
  PackWeight();
  return RET_OK;
}

}  // namespace mindspore::kernel
