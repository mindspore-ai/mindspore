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

#include "src/runtime/kernel/arm/fp32/convolution_slidewindow.h"
#include "src/runtime/kernel/arm/nnacl/common_func.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

namespace mindspore::kernel {
using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;

int ConvolutionSWCPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();
  int kernel_h = filter_tensor->Height();
  int kernel_w = filter_tensor->Width();
  conv_param_->input_channel_ = input_channel;
  conv_param_->output_channel_ = output_channel;
  int ic4 = UP_DIV(input_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int oc_block = C4NUM;
  int oc_block_num = UP_DIV(output_channel, C4NUM);
  int pack_weight_size = oc_block_num * oc_block * ic4 * C4NUM * kernel_plane;

  // ==================================init weight======================================//
  auto origin_weight = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->Data());
  packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed weight failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size * sizeof(float));
  for (int oc = 0; oc < output_channel; ++oc) {
    int src_oc_offset = oc * kernel_h * kernel_w * input_channel;
    int dst_oc_offset = oc * kernel_h * kernel_w * ic4 * C4NUM;
    for (int i = 0; i < kernel_h * kernel_w; ++i) {
      const float *src = origin_weight + src_oc_offset + i * input_channel;
      float *dst = packed_weight_ + dst_oc_offset + i * ic4 * C4NUM;
      memcpy(dst, src, input_channel * sizeof(float));
    }
  }

  // ====================================init bias====================================== //
  bias_data_ = reinterpret_cast<float *>(malloc(oc_block_num * oc_block * sizeof(float)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc_block_num * oc_block * sizeof(float));
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->Data());
    memcpy(bias_data_, ori_bias, output_channel * sizeof(float));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  return RET_OK;
}

int ConvolutionSWCPUKernel::InitTmpBuffer() {
  int out_channel = conv_param_->output_channel_;
  int oc4 = UP_DIV(out_channel, C4NUM);
  MS_ASSERT(ctx_->allocator != nullptr);

  /*=============================tmp_output_block_============================*/
  tmp_output_block_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(
    conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * oc4 * C4NUM * sizeof(float)));
  if (tmp_output_block_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp output block failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void ConvolutionSWCPUKernel::ConfigInputOutput() {
  // set output format
  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_tensor->SetFormat(schema::Format_NHWC);
}

int ConvolutionSWCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  auto ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  // config input output
  ConfigInputOutput();
  return ReSize();
}

int ConvolutionSWCPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  FreeTmpBuffer();
  if (nhwc4_input_ != nullptr) {
    free(nhwc4_input_);
    nhwc4_input_ = nullptr;
  }

  ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
  /*=============================nhwc4_input_============================*/
  int ic4 = UP_DIV(conv_param_->input_channel_, C4NUM);
  size_t nhwc4_input_size =
    ic4 * C4NUM * conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * sizeof(float);
  nhwc4_input_ = malloc(nhwc4_input_size);
  if (nhwc4_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc nhwc4 input failed.";
    return RET_ERROR;
  }
  memset(nhwc4_input_, 0, nhwc4_input_size);

  // init sliding window param
  slidingWindow_param_ = new (std::nothrow) SlidingWindowParam;
  if (slidingWindow_param_ == nullptr) {
    MS_LOG(ERROR) << "new SlidingWindowParam fail!";
    return RET_ERROR;
  }
  InitSlidingParamConv(slidingWindow_param_, conv_param_, C4NUM);

  return RET_OK;
}

int ConvolutionSWCPUKernel::RunImpl(int task_id) {
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->Data());
  ConvSWFp32(reinterpret_cast<float *>(nhwc4_input_), packed_weight_, reinterpret_cast<float *>(bias_data_),
             tmp_output_block_, output_addr, task_id, conv_param_, slidingWindow_param_);
  return RET_OK;
}

int ConvolutionSWImpl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv = reinterpret_cast<ConvolutionSWCPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution Sliding Window Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionSWCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }

  // init tmp input, output
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto ori_input_data = input_tensor->Data();
  PackNHWCToNHWC4Fp32(ori_input_data, nhwc4_input_, conv_param_->input_batch_,
                      conv_param_->input_h_ * conv_param_->input_w_, conv_param_->input_channel_);

  int error_code = LiteBackendParallelLaunch(ConvolutionSWImpl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv error error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  auto out_tensor = out_tensors_.front();
  auto out_data = reinterpret_cast<float *>(out_tensor->Data());
  int oc4_res = conv_param_->output_channel_ % C4NUM;
  if (oc4_res != 0) {
    PackNHWC4ToNHWCFp32(tmp_output_block_, out_data, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
  }
  FreeTmpBuffer();
  return RET_OK;
}
}  // namespace mindspore::kernel
