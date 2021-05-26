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
#ifdef ENABLE_AVX
#include "src/runtime/kernel/arm/fp32/convolution_slidewindow_fp32.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"
#include "nnacl/fp32/conv_common_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ConvolutionSWCPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();
  int kernel_h = filter_tensor->Height();
  int kernel_w = filter_tensor->Width();
  conv_param_->input_channel_ = input_channel;
  conv_param_->output_channel_ = output_channel;
  int kernel_plane = kernel_h * kernel_w;
  int oc_block_num = UP_DIV(output_channel, oc_tile_);
  int pack_weight_size = oc_block_num * oc_tile_ * input_channel * kernel_plane;
  packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed weight failed.";
    return RET_NULL_PTR;
  }
  memset(packed_weight_, 0, pack_weight_size * sizeof(float));
  PackNHWCTo1HWCNXFp32(kernel_h, kernel_w, output_channel, oc_block_num, input_channel, packed_weight_,
                       ori_weight_data_);
  if (in_tensors_.size() == kInputSize2) {
    bias_data_ = reinterpret_cast<float *>(malloc(oc_block_num * oc_tile_ * sizeof(float)));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias failed.";
      return RET_NULL_PTR;
    }
    memset(bias_data_, 0, oc_block_num * oc_tile_ * sizeof(float));
    memcpy(bias_data_, ori_bias_data_, output_channel * sizeof(float));
  } else {
    if (bias_data_ != nullptr) {
      free(bias_data_);
      bias_data_ = nullptr;
    }
  }
  return RET_OK;
}

int ConvolutionSWCPUKernel::Init() {
  oc_tile_ = C8NUM;
  oc_res_ = conv_param_->output_channel_ % oc_tile_;
  auto ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionSWCPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  if (slidingWindow_param_ != nullptr) {
    delete slidingWindow_param_;
    slidingWindow_param_ = nullptr;
  }

  ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }

  // init sliding window param
  slidingWindow_param_ = new (std::nothrow) SlidingWindowParam;
  if (slidingWindow_param_ == nullptr) {
    MS_LOG(ERROR) << "new SlidingWindowParam fail!";
    return RET_ERROR;
  }
  InitSlidingParamConv(slidingWindow_param_, conv_param_, oc_tile_);
  return RET_OK;
}

int ConvolutionSWCPUKernel::RunImpl(int task_id) {
  ConvSWFp32(ori_input_data_, packed_weight_, reinterpret_cast<float *>(bias_data_), output_data_, task_id, conv_param_,
             slidingWindow_param_);
  return RET_OK;
}

int ConvolutionSWImpl(void *cdata, int task_id) {
  auto conv = reinterpret_cast<ConvolutionSWCPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution Sliding Window Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionSWCPUKernel::InitTmpBuffer() {
  auto out_data = reinterpret_cast<float *>(out_tensors_.front()->MutableData());
  MS_ASSERT(out_data != nullptr);
  if (oc_res_ == 0) {  // not need to malloc dst
    output_data_ = out_data;
  } else {  // need to malloc dst to algin block
    int out_channel = conv_param_->output_channel_;
    int oc_block_num = UP_DIV(out_channel, oc_tile_);
    MS_ASSERT(ctx_->allocator != nullptr);
    output_data_ = reinterpret_cast<float *>(
      ctx_->allocator->Malloc(conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ *
                              oc_block_num * oc_tile_ * sizeof(float)));
    if (output_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc tmp output data failed.";
      return RET_NULL_PTR;
    }
  }
  return RET_OK;
}

int ConvolutionSWCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitTmpBuffer error!";
    return ret;
  }
  auto input_data = in_tensors_.at(kInputIndex)->MutableData();
  MS_ASSERT(input_data != nullptr);
  ori_input_data_ = reinterpret_cast<float *>(input_data);
  int error_code = static_cast<const lite::InnerContext *>(this->context_)
                     ->thread_pool_->ParallelLaunch(ConvolutionSWImpl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv error error_code[" << error_code << "]";
    FreeTmpBuffer();
    return error_code;
  }
  if (oc_res_ != 0) {
    auto out_data = reinterpret_cast<float *>(out_tensors_.front()->MutableData());
    PackNHWCXToNHWCFp32(output_data_, out_data, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_, oc_tile_);
  }
  FreeTmpBuffer();
  return RET_OK;
}
}  // namespace mindspore::kernel
#endif  // ENABLE_AVX
