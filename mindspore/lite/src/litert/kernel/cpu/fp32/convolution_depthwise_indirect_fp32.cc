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

#include "src/litert/kernel/cpu/fp32/convolution_depthwise_indirect_fp32.h"
#include "include/errorcode.h"
#include "src/litert/pack_weight_manager.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
ConvolutionDepthwiseIndirectCPUKernel::~ConvolutionDepthwiseIndirectCPUKernel() {
  if (zero_ptr_ != nullptr) {
    free(zero_ptr_);
    zero_ptr_ = nullptr;
  }
  if (indirect_buffer_ != nullptr) {
    free(indirect_buffer_);
    indirect_buffer_ = nullptr;
  }
}

int ConvolutionDepthwiseIndirectCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  UpdateOriginWeightAndBias();
  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_[kWeightIndex];
    CHECK_NULL_RETURN(weight_tensor);
    MS_CHECK_TRUE_MSG(weight_tensor->shape().size() == C4NUM, RET_ERROR, "Conv-like: weight-shape only support 4D.");
#ifdef ENABLE_AVX
    int div_flag = C8NUM;
#else
    int div_flag = C4NUM;
#endif
    int batch_flag = UP_DIV(weight_tensor->Batch(), div_flag);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_tensor->Height(), weight_tensor->Width(), RET_ERROR);
    int weight_size_hw = weight_tensor->Height() * weight_tensor->Width();
    MS_CHECK_INT_MUL_NOT_OVERFLOW(div_flag * batch_flag, weight_size_hw, RET_ERROR);
    int pack_weight_size = div_flag * batch_flag * weight_size_hw;
    set_workspace_size(pack_weight_size * sizeof(float));
  }
  auto ret = InitConvWeightBias();
  if (ret != 0) {
    MS_LOG(ERROR) << "Convolution depthwise Indirect fp32 InitConvWeightBias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDepthwiseIndirectCPUKernel::MallocIndirectBuffer() {
  // malloc indirect buffer
  step_w = conv_param_->dilation_w_ == 1 ? conv_param_->stride_w_ : conv_param_->kernel_w_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->kernel_h_, conv_param_->kernel_w_, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(step_w, conv_param_->kernel_h_, RET_ERROR);
  int step_w_2d = step_w * conv_param_->kernel_h_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW((conv_param_->output_w_ - 1), step_w_2d, RET_ERROR);
  step_h = (conv_param_->kernel_h_ * conv_param_->kernel_w_) + (conv_param_->output_w_ - 1) * step_w_2d;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_h_, step_h, RET_ERROR);
  int step_h_2d = conv_param_->output_h_ * step_h;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_batch_, step_h_2d, RET_ERROR);
  int buffer_size = conv_param_->output_batch_ * step_h_2d;
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, buffer_size * sizeof(float *));
  indirect_buffer_ = reinterpret_cast<float **>(malloc(buffer_size * sizeof(float *)));
  if (indirect_buffer_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseIndirectCPUKernel::ReSize() {
  if (indirect_buffer_ != nullptr) {
    free(indirect_buffer_);
    indirect_buffer_ = nullptr;
  }
  auto ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBaseCPUKernel::Prepare() return is:" << ret;
    return ret;
  }
  ret = MallocIndirectBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseIndirect MallocIndirectBuffer failed";
    return RET_ERROR;
  }
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  if (conv_param_->thread_num_ <= 0) {
    MS_LOG(ERROR) << "conv_param_->thread_num_ must be greater than 0!";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseIndirectCPUKernel::DoExecute(int task_id) {
  ConvDwIndirection(output_ptr_, indirect_buffer_, reinterpret_cast<float *>(packed_weight_),
                    reinterpret_cast<float *>(bias_data_), zero_ptr_, conv_param_, task_id);
  return RET_OK;
}

int ConvDwIndirectRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv_dw = reinterpret_cast<ConvolutionDepthwiseIndirectCPUKernel *>(cdata);
  auto ret = conv_dw->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseIndirectRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseIndirectCPUKernel::MallocPackedInput() {
#ifdef ENABLE_AVX
  int div_flag = C8NUM;
#else
  int div_flag = C4NUM;
#endif
  int IC_DIV = UP_DIV(conv_param_->input_channel_, div_flag);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->input_h_, conv_param_->input_w_, RET_ERROR);
  int conv_input_hw = conv_param_->input_h_ * conv_param_->input_w_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->input_batch_, conv_input_hw, RET_ERROR);
  int conv_input_bhw = conv_param_->input_batch_ * conv_input_hw;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_input_bhw, div_flag * IC_DIV, RET_ERROR);
  int pack_input_size = conv_input_bhw * div_flag * IC_DIV;
  packed_input_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(pack_input_size * sizeof(float)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseIndirectCPUKernel::Run() {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto input_ptr = reinterpret_cast<float *>(input_tensor->data());
  CHECK_NULL_RETURN(input_ptr);
#ifdef ENABLE_AVX
  int div_flag = C8NUM;
#else
  int div_flag = C4NUM;
#endif
  if (conv_param_->input_channel_ % div_flag != 0) {
    auto ret = MallocPackedInput();
    if (ret != 0) {
      MS_LOG(ERROR) << "Convolution depthwise fp32 indirect buffer MallocPackedInput failed.";
      return RET_ERROR;
    }
#ifdef ENABLE_AVX
    PackNHWCToNHWC8Fp32(input_ptr, packed_input_, conv_param_->input_batch_,
                        conv_param_->input_h_ * conv_param_->input_w_, conv_param_->input_channel_);
#else
    PackNHWCToNHWC4Fp32(input_ptr, packed_input_, conv_param_->input_batch_,
                        conv_param_->input_h_ * conv_param_->input_w_, conv_param_->input_channel_);
#endif
  } else {
    packed_input_ = input_ptr;
  }
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }
  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_ptr_ = reinterpret_cast<float *>(output_tensor->data());
  CHECK_NULL_RETURN(output_ptr_);
  ConvDwInitIndirection(indirect_buffer_, packed_input_, zero_ptr_, conv_param_, step_h, step_w);

  auto ret = ParallelLaunch(this->ms_context_, ConvDwIndirectRun, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwIndirectRun error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  if (conv_param_->input_channel_ % div_flag != 0) {
    ms_context_->allocator->Free(packed_input_);
  }
  return RET_OK;
}

void ConvolutionDepthwiseIndirectCPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  void *origin_weight = (op_parameter_->is_train_session_) ? weight_tensor->data() : origin_weight_;
  MS_ASSERT(origin_weight != nullptr);
#ifdef ENABLE_AVX
  PackDepthwiseIndirectWeightC8Fp32(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_),
                                    weight_tensor->Height(), weight_tensor->Width(), weight_tensor->Batch());
#else
  PackDepthwiseIndirectWeightC4Fp32(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_),
                                    weight_tensor->Height(), weight_tensor->Width(), weight_tensor->Batch());
#endif
}

int ConvolutionDepthwiseIndirectCPUKernel::MallocWeightBiasData() {
  auto weight_tensor = in_tensors_[kWeightIndex];
#ifdef ENABLE_AVX
  int div_flag = C8NUM;
#else
  int div_flag = C4NUM;
#endif
  int batch_flag = UP_DIV(weight_tensor->Batch(), div_flag);
  int pack_weight_size = div_flag * batch_flag * weight_tensor->Height() * weight_tensor->Width();
  if (!op_parameter_->is_train_session_) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, pack_weight_size * sizeof(float));
    packed_weight_ = GetConvPackWeightData(static_cast<size_t>(pack_weight_size * sizeof(float)));
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  if (bias_data_ == nullptr) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, batch_flag * div_flag * sizeof(float));
    bias_data_ = malloc(batch_flag * div_flag * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, batch_flag * div_flag * sizeof(float));

  // malloc zero ptr
  zero_ptr_ = reinterpret_cast<float *>(malloc(batch_flag * div_flag * sizeof(float)));
  if (zero_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(zero_ptr_, 0, batch_flag * div_flag * sizeof(float));
  return RET_OK;
}
}  // namespace mindspore::kernel
