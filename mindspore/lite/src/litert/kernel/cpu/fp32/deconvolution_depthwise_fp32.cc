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

#include "src/litert/kernel/cpu/fp32/deconvolution_depthwise_fp32.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
DeconvolutionDepthwiseCPUKernel::~DeconvolutionDepthwiseCPUKernel() {
  if (sliding_ != nullptr) {
    delete sliding_;
    sliding_ = nullptr;
  }
}

int DeconvolutionDepthwiseCPUKernel::InitSlideParam() {
  MS_CHECK_TRUE_RET(in_tensors_.front()->shape().size() == DIMENSION_4D, RET_ERROR);
  MS_CHECK_TRUE_RET(out_tensors_.front()->shape().size() == DIMENSION_4D, RET_ERROR);

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

int DeconvolutionDepthwiseCPUKernel::InitPackedInputOutput() {
  if (conv_param_->input_channel_ % C4NUM != 0) {
    need_align_ = true;
    int IC4 = UP_DIV(conv_param_->input_channel_, C4NUM);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->input_h_, conv_param_->input_w_, RET_ERROR);
    int input_hw = conv_param_->input_h_ * conv_param_->input_w_;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->input_batch_, input_hw, RET_ERROR);
    int input_bhw = conv_param_->input_batch_ * input_hw;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(input_bhw, C4NUM * IC4, RET_ERROR);
    int pack_input_size = input_bhw * C4NUM * IC4;
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
    memset(packed_output_, 0, pack_output_size * sizeof(float));
  }
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(conv_param_);
  CHECK_NOT_EQUAL_RETURN(conv_param_->dilation_h_, C1NUM);
  CHECK_NOT_EQUAL_RETURN(conv_param_->dilation_w_, C1NUM);

  UpdateOriginWeightAndBias();

  sliding_ = new (std::nothrow) SlidingWindowParam;
  if (sliding_ == nullptr) {
    MS_LOG(ERROR) << "new sliding window param failed.";
    return RET_ERROR;
  }
  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    int OC4 = UP_DIV(weight_tensor->Batch(), C4NUM);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_tensor->Height(), weight_tensor->Width(), RET_ERROR);
    int weight_size_hw = weight_tensor->Height() * weight_tensor->Width();
    MS_CHECK_INT_MUL_NOT_OVERFLOW(C4NUM * OC4, weight_size_hw, RET_ERROR);
    int pack_weight_size = C4NUM * OC4 * weight_size_hw;
    set_workspace_size(pack_weight_size * sizeof(float));
  }
  auto ret = InitConvWeightBias();
  if (ret != 0) {
    MS_LOG(ERROR) << "Deconvolution depthwise fp32 InitConvWeightBias failed.ret: " << ret;
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DeconvolutionDepthwiseCPUKernel::ReSize() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.front());
  CHECK_NULL_RETURN(out_tensors_.front());
  CHECK_NULL_RETURN(conv_param_);
  CHECK_NULL_RETURN(sliding_);

  auto ret = ConvolutionBaseCPUKernel::CheckDeconvResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  auto weight_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(weight_tensor);
  CHECK_NOT_EQUAL_RETURN(conv_param_->kernel_h_, weight_tensor->Height());
  CHECK_NOT_EQUAL_RETURN(conv_param_->kernel_w_, weight_tensor->Width());

  ret = InitSlideParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitSlideParam is failed!";
    return ret;
  }
  ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBaseCPUKernel init failed!";
    return ret;
  }
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::DoExecute(int task_id) {
  DeconvDwSWFp32(packed_output_, packed_input_, reinterpret_cast<float *>(packed_weight_),
                 reinterpret_cast<float *>(bias_data_), conv_param_, sliding_, task_id);
  return RET_OK;
}

int DeconvDwRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto deconv_dw = reinterpret_cast<DeconvolutionDepthwiseCPUKernel *>(cdata);
  auto ret = deconv_dw->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DeconvolutionDepthwiseRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::Run() {
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }
  if (conv_param_->input_channel_ != conv_param_->output_channel_) {
    MS_LOG(ERROR) << "Only support input channel equals output channel.";
    return RET_ERROR;
  }

  CHECK_NULL_RETURN(packed_weight_);
  CHECK_NULL_RETURN(bias_data_);

  auto ret = InitPackedInputOutput();
  if (ret != 0) {
    MS_LOG(ERROR) << "Deconvolution depthwise fp32 InitPackedInputOutput failed.ret: " << ret;
    FreePackedInputOutput();
    return ret;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  auto output_tensor = out_tensors_.at(kOutputIndex);
  CHECK_NULL_RETURN(input_tensor);
  CHECK_NULL_RETURN(output_tensor);

  auto input_addr = reinterpret_cast<float *>(input_tensor->data());
  auto output_addr = reinterpret_cast<float *>(output_tensor->data());
  CHECK_NULL_RETURN(input_addr);
  CHECK_NULL_RETURN(output_addr);

  if (need_align_) {
    PackNHWCToNHWC4Fp32(input_addr, packed_input_, conv_param_->input_batch_,
                        conv_param_->input_h_ * conv_param_->input_w_, conv_param_->input_channel_);
  } else {
    packed_input_ = input_addr;
  }

  if (!need_align_) {
    memset(output_addr, 0, output_tensor->ElementsNum() * sizeof(float));
    packed_output_ = output_addr;
  }

  ret = ParallelLaunch(this->ms_context_, DeconvDwRun, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DeconvDwRun error: error_code[" << ret << "]";
  }

  if (need_align_) {
    PackNHWCXToNHWCFp32(packed_output_, output_addr, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_, C4NUM);
  }
  FreePackedInputOutput();
  return ret;
}

int DeconvolutionDepthwiseCPUKernel::MallocWeightBiasData() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  int OC4 = UP_DIV(weight_tensor->Batch(), C4NUM);
  int pack_weight_size = C4NUM * OC4 * weight_tensor->Height() * weight_tensor->Width();
  if (!op_parameter_->is_train_session_) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, pack_weight_size * sizeof(float));
    packed_weight_ = GetConvPackWeightData(pack_weight_size * sizeof(float));
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  if (bias_data_ == nullptr) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, C4NUM * OC4 * sizeof(float));
    bias_data_ = malloc(C4NUM * OC4 * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, C4NUM * OC4 * sizeof(float));
  conv_param_->thread_num_ = MSMIN(thread_count_, OC4);
  return RET_OK;
}

void DeconvolutionDepthwiseCPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  void *origin_weight = IsTrainable() ? weight_tensor->data() : origin_weight_;
  MS_ASSERT(origin_weight != nullptr);
  PackNCHWToNC4HW4Fp32(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_), 1,
                       weight_tensor->Height() * weight_tensor->Width(), weight_tensor->Batch());
}

void DeconvolutionDepthwiseCPUKernel::FreePackedInputOutput() {
  if (need_align_) {
    ms_context_->allocator->Free(packed_input_);
    ms_context_->allocator->Free(packed_output_);
    packed_input_ = nullptr;
    packed_output_ = nullptr;
  }
}
}  // namespace mindspore::kernel
