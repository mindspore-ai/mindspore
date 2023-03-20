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

#include "src/litert/kernel/cpu/fp32/convolution_depthwise_slidewindow_fp32.h"
#include "src/litert/pack_weight_manager.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
ConvolutionDepthwiseSWCPUKernel::~ConvolutionDepthwiseSWCPUKernel() {
  if (sliding_ != nullptr) {
    delete sliding_;
    sliding_ = nullptr;
  }
}

int ConvolutionDepthwiseSWCPUKernel::InitPackedInputOutput() {
  if (conv_param_->input_channel_ % C4NUM != 0) {
    need_align_ = true;
    int IC4 = UP_DIV(conv_param_->input_channel_, C4NUM);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->input_h_, conv_param_->input_w_, RET_ERROR);
    int conv_input_hw = conv_param_->input_h_ * conv_param_->input_w_;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->input_batch_, conv_input_hw, RET_ERROR);
    int conv_input_bhw = conv_param_->input_batch_ * conv_input_hw;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_input_bhw, C4NUM * IC4, RET_ERROR);
    int pack_input_size = conv_input_bhw * C4NUM * IC4;
    packed_input_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(pack_input_size * sizeof(float)));
    if (packed_input_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }

    int OC4 = UP_DIV(conv_param_->output_channel_, C4NUM);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_h_, conv_param_->output_w_, RET_ERROR);
    int output_hw = conv_param_->output_h_ * conv_param_->output_w_;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_batch_, output_hw, RET_ERROR);
    int output_bhw = conv_param_->output_batch_ * output_hw;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(output_bhw, C4NUM * OC4, RET_ERROR);
    int pack_output_size = output_bhw * C4NUM * OC4;
    packed_output_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(pack_output_size * sizeof(float)));
    if (packed_output_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ConvolutionDepthwiseSWCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  UpdateOriginWeightAndBias();
  sliding_ = new (std::nothrow) SlidingWindowParam;
  if (sliding_ == nullptr) {
    MS_LOG(ERROR) << "new sliding window param failed.";
    return RET_ERROR;
  }
  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(weight_tensor);
    MS_CHECK_TRUE_MSG(weight_tensor->shape().size() == C4NUM, RET_ERROR, "Conv-like: weight-shape only support 4D.");
    int OC4 = UP_DIV(weight_tensor->Batch(), C4NUM);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_tensor->Height(), weight_tensor->Width(), RET_ERROR);
    int weight_size_hw = weight_tensor->Height() * weight_tensor->Width();
    MS_CHECK_INT_MUL_NOT_OVERFLOW(C4NUM * OC4, weight_size_hw, RET_ERROR);
    int pack_weight_size = C4NUM * OC4 * weight_size_hw;
    set_workspace_size(pack_weight_size * sizeof(float));
  }
  auto ret = InitConvWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Convolution depthwise fp32 InitConvWeightBias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDepthwiseSWCPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBaseCPUKernel::Prepare() return is:" << ret;
    return ret;
  }
  InitSlidingParamConvDw(sliding_, conv_param_, C4NUM);
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  if (conv_param_->thread_num_ <= 0) {
    MS_LOG(ERROR) << "conv_param_->thread_num_ must be greater than 0!";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseSWCPUKernel::DoExecute(int task_id) {
  ConvDwSWFp32(packed_output_, packed_input_, reinterpret_cast<float *>(packed_weight_),
               reinterpret_cast<float *>(bias_data_), conv_param_, sliding_, task_id);
  return RET_OK;
}

int ConvDwSWRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv_dw = reinterpret_cast<ConvolutionDepthwiseSWCPUKernel *>(cdata);
  auto ret = conv_dw->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseSWRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseSWCPUKernel::Run() {
  auto ret = InitPackedInputOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Convolution depthwise fp32 InitPackedInputOutput failed.";
    FreePackedInputOutput();
    return RET_ERROR;
  }
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  CHECK_NULL_RETURN(input_tensor);
  auto input_ptr = reinterpret_cast<float *>(input_tensor->data());
  CHECK_NULL_RETURN(input_ptr);
  if (need_align_) {
    PackNHWCToNHWC4Fp32(input_ptr, packed_input_, conv_param_->input_batch_,
                        conv_param_->input_h_ * conv_param_->input_w_, conv_param_->input_channel_);
  } else {
    packed_input_ = input_ptr;
  }

  auto output_tensor = out_tensors_.at(kOutputIndex);
  CHECK_NULL_RETURN(output_tensor);
  auto output_ptr = reinterpret_cast<float *>(output_tensor->data());
  CHECK_NULL_RETURN(output_ptr);
  if (!need_align_) {
    packed_output_ = output_ptr;
  }

  ret = ParallelLaunch(this->ms_context_, ConvDwSWRun, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwSWRun error: error_code[" << ret << "]";
  }

  if (need_align_) {
    PackNHWCXToNHWCFp32(packed_output_, output_ptr, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_, C4NUM);
  }
  FreePackedInputOutput();
  return ret;
}

void ConvolutionDepthwiseSWCPUKernel::FreePackedInputOutput() {
  if (need_align_) {
    ms_context_->allocator->Free(packed_input_);
    ms_context_->allocator->Free(packed_output_);
    packed_input_ = nullptr;
    packed_output_ = nullptr;
  }
}

void ConvolutionDepthwiseSWCPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  void *origin_weight = (op_parameter_->is_train_session_) ? weight_tensor->data() : origin_weight_;
  MS_ASSERT(origin_weight != nullptr);
  PackNCHWToNC4HW4Fp32(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_), 1,
                       weight_tensor->Height() * weight_tensor->Width(), weight_tensor->Batch());
}

int ConvolutionDepthwiseSWCPUKernel::MallocWeightBiasData() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  int OC4 = UP_DIV(weight_tensor->Batch(), C4NUM);
  int pack_weight_size = C4NUM * OC4 * weight_tensor->Height() * weight_tensor->Width();
  if (!op_parameter_->is_train_session_) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, pack_weight_size * sizeof(float));
    packed_weight_ = GetConvPackWeightData(static_cast<size_t>(pack_weight_size) * sizeof(float));
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  int malloc_size = MSMAX(conv_param_->output_channel_, C4NUM * OC4);
  if (bias_data_ == nullptr) {
    CHECK_LESS_RETURN(malloc_size, 0);
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, malloc_size * sizeof(float));
    bias_data_ = malloc(malloc_size * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, malloc_size * sizeof(float));
  conv_param_->thread_num_ = MSMIN(thread_count_, OC4);
  return RET_OK;
}
}  // namespace mindspore::kernel
