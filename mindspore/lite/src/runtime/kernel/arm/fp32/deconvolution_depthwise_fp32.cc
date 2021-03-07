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

#include "src/runtime/kernel/arm/fp32/deconvolution_depthwise_fp32.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
DeconvolutionDepthwiseCPUKernel::~DeconvolutionDepthwiseCPUKernel() {
  if (sliding_ != nullptr) {
    delete sliding_;
    sliding_ = nullptr;
  }
  if (packed_weight_ != nullptr) {
    free(packed_weight_);
    packed_weight_ = nullptr;
  }
}

int DeconvolutionDepthwiseCPUKernel::InitSlideParam() {
  conv_param_->input_batch_ = out_tensors_.front()->shape().at(kNHWC_N);
  conv_param_->input_h_ = out_tensors_.front()->shape().at(kNHWC_H);
  conv_param_->input_w_ = out_tensors_.front()->shape().at(kNHWC_W);
  conv_param_->input_channel_ = out_tensors_.front()->shape().at(kNHWC_C);
  conv_param_->output_batch_ = in_tensors_.front()->shape().at(kNHWC_N);
  conv_param_->output_h_ = in_tensors_.front()->shape().at(kNHWC_H);
  conv_param_->output_w_ = in_tensors_.front()->shape().at(kNHWC_W);
  conv_param_->output_channel_ = in_tensors_.front()->shape().at(kNHWC_C);
  InitSlidingParamConvDw(sliding_, conv_param_, C4NUM);
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::InitWeightBias() {
  // init weight: o, h, w, i; o == group, i == 1
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->MutableData());
  int OC4 = UP_DIV(weight_tensor->Batch(), C4NUM);
  int pack_weight_size = C4NUM * OC4 * weight_tensor->Height() * weight_tensor->Width();

  packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  PackNCHWToNC4HW4Fp32(origin_weight, packed_weight_, 1, weight_tensor->Height() * weight_tensor->Width(),
                       weight_tensor->Batch());

  bias_data_ = reinterpret_cast<float *>(malloc(C4NUM * OC4 * sizeof(float)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, C4NUM * OC4 * sizeof(float));
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->MutableData());
    memcpy(bias_data_, ori_bias, in_tensors_.at(kBiasIndex)->ElementsNum() * sizeof(float));
  }

  conv_param_->thread_num_ = MSMIN(thread_count_, OC4);
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::InitPackedInputOutput() {
  if (conv_param_->input_channel_ % C4NUM != 0) {
    need_align_ = true;
    int IC4 = UP_DIV(conv_param_->input_channel_, C4NUM);
    int pack_input_size = conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * C4NUM * IC4;
    packed_input_ = reinterpret_cast<float *>(context_->allocator->Malloc(pack_input_size * sizeof(float)));
    if (packed_input_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }

    int OC4 = UP_DIV(conv_param_->output_channel_, C4NUM);
    int pack_output_size = conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * C4NUM * OC4;
    packed_output_ = reinterpret_cast<float *>(context_->allocator->Malloc(pack_output_size * sizeof(float)));
    if (packed_output_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
    memset(packed_output_, 0, pack_output_size * sizeof(float));
  }
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::Init() {
  sliding_ = new (std::nothrow) SlidingWindowParam;
  if (sliding_ == nullptr) {
    MS_LOG(ERROR) << "new sliding window param failed.";
    return RET_ERROR;
  }

  auto ret = InitWeightBias();
  if (ret != 0) {
    MS_LOG(ERROR) << "Deconvolution depthwise fp32 InitWeightBias failed.ret: " << ret;
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DeconvolutionDepthwiseCPUKernel::ReSize() {
  InitSlideParam();
  ConvolutionBaseCPUKernel::Init();
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::Execute(int task_id) {
  DeconvDwSWFp32(packed_output_, packed_input_, packed_weight_, reinterpret_cast<float *>(bias_data_), conv_param_,
                 sliding_, task_id);
  return RET_OK;
}

int DeconvDwRun(void *cdata, int task_id) {
  auto deconv_dw = reinterpret_cast<DeconvolutionDepthwiseCPUKernel *>(cdata);
  auto ret = deconv_dw->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DeconvolutionDepthwiseRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::Run() {
  if (conv_param_->input_channel_ != conv_param_->output_channel_) {
    MS_LOG(ERROR) << "Only support input channel equals output channel.";
    return RET_ERROR;
  }

  auto ret = InitPackedInputOutput();
  if (ret != 0) {
    MS_LOG(ERROR) << "Deconvolution depthwise fp32 InitPackedInputOutput failed.ret: " << ret;
    FreePackedInputOutput();
    return ret;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  auto input_addr = reinterpret_cast<float *>(input_tensor->MutableData());

  if (need_align_) {
    PackNHWCToNHWC4Fp32(input_addr, packed_input_, conv_param_->input_batch_,
                        conv_param_->input_h_ * conv_param_->input_w_, conv_param_->input_channel_);
  } else {
    packed_input_ = input_addr;
  }

  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());
  if (!need_align_) {
    memset(output_addr, 0, out_tensors_.at(kOutputIndex)->ElementsNum() * sizeof(float));
    packed_output_ = output_addr;
  }

  ret = ParallelLaunch(this->context_->thread_pool_, DeconvDwRun, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DeconvDwRun error: error_code[" << ret << "]";
  }

  if (need_align_) {
    PackNHWC4ToNHWCFp32(packed_output_, output_addr, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
  }
  FreePackedInputOutput();
  return ret;
}

void DeconvolutionDepthwiseCPUKernel::FreePackedInputOutput() {
  if (need_align_) {
    context_->allocator->Free(packed_input_);
    context_->allocator->Free(packed_output_);
    packed_input_ = nullptr;
    packed_output_ = nullptr;
  }
}
}  // namespace mindspore::kernel
