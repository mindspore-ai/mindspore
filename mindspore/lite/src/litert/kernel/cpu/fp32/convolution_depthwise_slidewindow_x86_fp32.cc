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
#include "src/litert/kernel/cpu/fp32/convolution_depthwise_slidewindow_x86_fp32.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
ConvolutionDepthwiseSWCPUKernelX86::~ConvolutionDepthwiseSWCPUKernelX86() {
  if (sliding_ != nullptr) {
    delete sliding_;
    sliding_ = nullptr;
  }
}

int ConvolutionDepthwiseSWCPUKernelX86::InitPackedInputOutput() {
  CHECK_NULL_RETURN(conv_param_);
  if (conv_param_->input_channel_ % oc_tile_ != 0) {
    input_need_align_ = true;
    int ic_algin = UP_DIV(conv_param_->input_channel_, oc_tile_);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->input_h_, conv_param_->input_w_, RET_ERROR);
    int input_hw = conv_param_->input_h_ * conv_param_->input_w_;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->input_batch_, input_hw, RET_ERROR);
    int input_bhw = conv_param_->input_batch_ * input_hw;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(input_bhw, oc_tile_ * ic_algin, RET_ERROR);
    int pack_input_size = input_bhw * oc_tile_ * ic_algin;
    packed_input_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(pack_input_size * sizeof(float)));
    if (packed_input_ == nullptr) {
      MS_LOG(ERROR) << "Malloc packed_input_ buffer is failed.";
      return RET_NULL_PTR;
    }
  }
  if (conv_param_->output_channel_ % oc_tile_ != 0) {
    output_need_align_ = true;
    int oc_algin = UP_DIV(conv_param_->output_channel_, oc_tile_);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_h_, conv_param_->output_w_, RET_ERROR);
    int output_hw = conv_param_->output_h_ * conv_param_->output_w_;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_batch_, output_hw, RET_ERROR);
    int output_bhw = conv_param_->output_batch_ * output_hw;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(output_bhw, oc_tile_ * oc_algin, RET_ERROR);
    int pack_output_size = output_bhw * oc_tile_ * oc_algin;
    packed_output_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(pack_output_size * sizeof(float)));
    if (packed_output_ == nullptr) {
      MS_LOG(ERROR) << "Malloc packed_output_ buffer is failed.";
      return RET_NULL_PTR;
    }
  }
  return RET_OK;
}

int ConvolutionDepthwiseSWCPUKernelX86::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  UpdateOriginWeightAndBias();
#ifdef ENABLE_AVX
  oc_tile_ = C8NUM;
#endif
  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(weight_tensor);
    MS_CHECK_TRUE_MSG(weight_tensor->shape().size() == C4NUM, RET_ERROR, "Conv-like: weight-shape only support 4D.");
    int oc_algin = UP_DIV(weight_tensor->Batch(), oc_tile_);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_tensor->Height(), weight_tensor->Width(), RET_ERROR);
    int weight_size_hw = weight_tensor->Height() * weight_tensor->Width();
    MS_CHECK_INT_MUL_NOT_OVERFLOW(oc_algin * oc_tile_, weight_size_hw, RET_ERROR);
    int pack_weight_size = oc_algin * oc_tile_ * weight_size_hw;
    set_workspace_size(pack_weight_size * sizeof(float));
  }
  sliding_ = new (std::nothrow) SlidingWindowParam;
  if (sliding_ == nullptr) {
    MS_LOG(ERROR) << "new sliding window param failed.";
    return RET_ERROR;
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

int ConvolutionDepthwiseSWCPUKernelX86::ReSize() {
  ConvolutionBaseCPUKernel::Prepare();
  InitSlidingParamConvDw(sliding_, conv_param_, oc_tile_);
  return RET_OK;
}

int ConvolutionDepthwiseSWCPUKernelX86::Execute(int task_id) {
  DepthwiseSWAvxFp32(packed_output_, packed_input_, reinterpret_cast<float *>(packed_weight_),
                     reinterpret_cast<float *>(bias_data_), conv_param_, sliding_, task_id);
  return RET_OK;
}

int ConvDwSWAvxRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv_dw = reinterpret_cast<ConvolutionDepthwiseSWCPUKernelX86 *>(cdata);
  auto ret = conv_dw->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseSWRun in x86 error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseSWCPUKernelX86::Run() {
  auto ret = InitPackedInputOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Convolution depthwise x86 fp32 InitPackedInputOutput failed.";
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

  if (input_need_align_) {
    PackNHWCToNHWCXFp32(input_ptr, packed_input_, conv_param_->input_batch_,
                        conv_param_->input_h_ * conv_param_->input_w_, conv_param_->input_channel_, oc_tile_);
  } else {
    packed_input_ = input_ptr;
  }

  auto output_tensor = out_tensors_.at(kOutputIndex);
  CHECK_NULL_RETURN(output_tensor);
  auto output_ptr = reinterpret_cast<float *>(output_tensor->data());
  CHECK_NULL_RETURN(output_ptr);

  if (!output_need_align_) {
    packed_output_ = output_ptr;
  }

  ret = ParallelLaunch(this->ms_context_, ConvDwSWAvxRun, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwSWAvxRun error: error_code[" << ret << "]";
  }

  if (output_need_align_) {
    PackNHWCXToNHWCFp32(packed_output_, output_ptr, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_, oc_tile_);
  }
  FreePackedInputOutput();
  return ret;
}

void ConvolutionDepthwiseSWCPUKernelX86::FreePackedInputOutput() {
  if (input_need_align_) {
    ms_context_->allocator->Free(packed_input_);
    packed_input_ = nullptr;
  }
  if (output_need_align_) {
    ms_context_->allocator->Free(packed_output_);
    packed_output_ = nullptr;
  }
}

void ConvolutionDepthwiseSWCPUKernelX86::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  int oc_algin = UP_DIV(weight_tensor->Batch(), oc_tile_);
  void *origin_weight = IsTrainable() ? weight_tensor->data() : origin_weight_;
  MS_ASSERT(origin_weight != nullptr);
  PackNHWCToNXHWCXFp32(weight_tensor->Height(), weight_tensor->Width(), weight_tensor->Batch(), oc_algin,
                       weight_tensor->Channel(), reinterpret_cast<float *>(packed_weight_),
                       reinterpret_cast<float *>(origin_weight));
}

int ConvolutionDepthwiseSWCPUKernelX86::MallocWeightBiasData() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  int oc_algin = UP_DIV(weight_tensor->Batch(), oc_tile_);
  int pack_weight_size = oc_algin * oc_tile_ * weight_tensor->Height() * weight_tensor->Width();
  if (!op_parameter_->is_train_session_) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, pack_weight_size * sizeof(float));
    packed_weight_ = GetConvPackWeightData(pack_weight_size * sizeof(float));
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "Malloc packed_weight_ is failed!";
      return RET_NULL_PTR;
    }
  }

  if (in_tensors_.size() == kInputSize2) {
    auto bias_size = oc_algin * oc_tile_;
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, bias_size * sizeof(float));
    bias_data_ = malloc(bias_size * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc bias_data buffer failed.";
      return RET_NULL_PTR;
    }
    memset(bias_data_, 0, bias_size * sizeof(float));
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
#endif
