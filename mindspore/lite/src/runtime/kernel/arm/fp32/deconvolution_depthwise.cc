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

#include "src/runtime/kernel/arm/fp32/deconvolution_depthwise.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DepthwiseConv2D;

namespace mindspore::kernel {
int DeconvolutionDepthwiseCPUKernel::InitSlideParam() {
  conv_param_->input_batch_ = outputs_.front()->shape().at(kNHWC_N);
  conv_param_->input_h_ = outputs_.front()->shape().at(kNHWC_H);
  conv_param_->input_w_ = outputs_.front()->shape().at(kNHWC_W);
  conv_param_->input_channel_ = outputs_.front()->shape().at(kNHWC_C);
  conv_param_->output_batch_ = inputs_.front()->shape().at(kNHWC_N);
  conv_param_->output_h_ = inputs_.front()->shape().at(kNHWC_H);
  conv_param_->output_w_ = inputs_.front()->shape().at(kNHWC_W);
  conv_param_->output_channel_ = inputs_.front()->shape().at(kNHWC_C);

  // init sliding window param
  sliding = new SlidingWindowParam;
  InitSlidingParam(sliding, conv_param_, C4NUM);
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::Init() {
  InitSlideParam();
  // conv base init
  ConvolutionBaseCPUKernel::Init();

  // pack input function: convert_func_
  auto input_tensor = inputs_[kInputIndex];
  auto data_type = input_tensor->data_type();
  auto input_format = input_tensor->GetFormat();
  schema::Format execute_format = schema::Format_NHWC4;
  if (input_format != execute_format) {
    convert_func_ = LayoutTransform(data_type, input_format, execute_format);
    if (convert_func_ == nullptr) {
      MS_LOG(ERROR) << "layout convert func is nullptr.";
      return RET_ERROR;
    }
  }

  // init weight: o, h, w, i; o == group, i == 1
  auto weight_tensor = inputs_[kWeightIndex];
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->Data());
  int OC4 = UP_DIV(conv_param_->output_channel_, C4NUM);
  int pack_weight_size = C4NUM * OC4 * conv_param_->kernel_h_ * conv_param_->kernel_w_;

  packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
  memset(packed_weight_, 0, pack_weight_size * sizeof(float));
  PackNCHWToNC4HW4Fp32(origin_weight, packed_weight_, 1, conv_param_->kernel_h_ * conv_param_->kernel_w_,
                       conv_param_->output_channel_);

  // init bias
  bias_data_ = reinterpret_cast<float *>(malloc(C4NUM * OC4 * sizeof(float)));
  memset(bias_data_, 0, C4NUM * OC4 * sizeof(float));
  if (inputs_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(inputs_.at(kBiasIndex)->Data());
    memcpy(bias_data_, ori_bias, conv_param_->output_channel_ * sizeof(float));
  } else {
    MS_ASSERT(inputs_.size() == kInputSize1);
  }

  // init threadNum;
  conv_param_->thread_num_ = MSMIN(conv_param_->thread_num_, OC4);
  ReSize();
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::ReSize() {
  // malloc pack input buffer
  if (convert_func_ != nullptr) {
    int IC4 = UP_DIV(conv_param_->input_channel_, C4NUM);
    int pack_input_size = conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * C4NUM * IC4;
    packed_input_ = reinterpret_cast<float *>(malloc(pack_input_size * sizeof(float)));
    memset(packed_input_, 0, pack_input_size * sizeof(float));
  }

  // malloc tmp output buffer
  if (conv_param_->output_channel_ % C4NUM != 0) {
    need_pack_ = true;
    int OC4 = UP_DIV(conv_param_->output_channel_, C4NUM);
    int pack_output_size = conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * C4NUM * OC4;
    packed_output_ = reinterpret_cast<float *>(malloc(pack_output_size * sizeof(float)));
    memset(packed_output_, 0, pack_output_size * sizeof(float));
  }
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::DoExcute(int task_id) {
  DeconvDwC4Fp32(packed_output_, packed_input_, packed_weight_, reinterpret_cast<float *>(bias_data_), conv_param_,
                 sliding, task_id);
  return RET_OK;
}

int DeconvDwRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv_dw = reinterpret_cast<DeconvolutionDepthwiseCPUKernel *>(cdata);
  auto ret = conv_dw->DoExcute(task_id);
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
  auto input_tensor = inputs_.at(kInputIndex);
  auto input_addr = reinterpret_cast<float *>(input_tensor->Data());

  // pack input: to nhwc4
  if (convert_func_ != nullptr) {
    convert_func_(input_addr, packed_input_, conv_param_->input_batch_, conv_param_->input_h_ * conv_param_->input_w_,
                  conv_param_->input_channel_);
  } else {
    packed_input_ = input_addr;
  }

  output_addr = reinterpret_cast<float *>(outputs_.at(kOutputIndex)->Data());
  memset(output_addr, 0, outputs_.at(kOutputIndex)->ElementsNum() * sizeof(float));
  if (!need_pack_) {
    packed_output_ = output_addr;
  }

  auto ret = LiteBackendParallelLaunch(DeconvDwRun, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwRun error: error_code[" << ret << "]";
    return RET_ERROR;
  }

  if (need_pack_) {
    PackNHWC4ToNHWCFp32(packed_output_, output_addr, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
  }
  return RET_OK;
}
}  // namespace mindspore::kernel

