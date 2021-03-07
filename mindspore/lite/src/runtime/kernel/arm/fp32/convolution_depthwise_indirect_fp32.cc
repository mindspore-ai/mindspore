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

#include "src/runtime/kernel/arm/fp32/convolution_depthwise_indirect_fp32.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
ConvolutionDepthwiseIndirectCPUKernel::~ConvolutionDepthwiseIndirectCPUKernel() {
  if (packed_weight_ != nullptr) {
    free(packed_weight_);
    packed_weight_ = nullptr;
  }
  if (zero_ptr_ != nullptr) {
    free(zero_ptr_);
    zero_ptr_ = nullptr;
  }
  if (indirect_buffer_ != nullptr) {
    free(indirect_buffer_);
    indirect_buffer_ = nullptr;
  }
}

int ConvolutionDepthwiseIndirectCPUKernel::InitWeightBias() {
  // init weight: o, h, w, i; o == group, i == 1
  auto weight_tensor = in_tensors_[kWeightIndex];
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->MutableData());
#ifdef ENABLE_AVX
  int div_flag = C8NUM;
#else
  int div_flag = C4NUM;
#endif
  int batch_flag = UP_DIV(weight_tensor->Batch(), div_flag);
  int pack_weight_size = div_flag * batch_flag * weight_tensor->Height() * weight_tensor->Width();

  packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
#ifdef ENABLE_AVX
  PackDepthwiseIndirectWeightC8Fp32(origin_weight, packed_weight_, weight_tensor->Height(), weight_tensor->Width(),
                                    weight_tensor->Batch());
#else
  PackDepthwiseIndirectWeightC4Fp32(origin_weight, packed_weight_, weight_tensor->Height(), weight_tensor->Width(),
                                    weight_tensor->Batch());
#endif

  bias_data_ = reinterpret_cast<float *>(malloc(batch_flag * div_flag * sizeof(float)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }

  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_[kBiasIndex];
    auto ori_bias = reinterpret_cast<float *>(bias_tensor->MutableData());
    memcpy(bias_data_, ori_bias, bias_tensor->ElementsNum() * sizeof(float));
  } else {
    memset(bias_data_, 0, batch_flag * div_flag * sizeof(float));
  }

  // malloc zero ptr
  zero_ptr_ = reinterpret_cast<float *>(malloc(batch_flag * div_flag * sizeof(float)));
  if (zero_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(zero_ptr_, 0, batch_flag * div_flag * sizeof(float));
  return RET_OK;
}

int ConvolutionDepthwiseIndirectCPUKernel::Init() {
  auto ret = InitWeightBias();
  if (ret != 0) {
    MS_LOG(ERROR) << "Convolution depthwise Indirect fp32 InitWeightBias failed.";
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
  step_h =
    (conv_param_->kernel_h_ * conv_param_->kernel_w_) + (conv_param_->output_w_ - 1) * step_w * conv_param_->kernel_h_;
  int buffer_size = conv_param_->output_batch_ * conv_param_->output_h_ * step_h;
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
  ConvolutionBaseCPUKernel::Init();
  auto ret = MallocIndirectBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseIndirect MallocIndirectBuffer failed";
    return RET_ERROR;
  }
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  return RET_OK;
}

int ConvolutionDepthwiseIndirectCPUKernel::Execute(int task_id) {
  ConvDwIndirection(output_ptr_, indirect_buffer_, packed_weight_, reinterpret_cast<float *>(bias_data_), zero_ptr_,
                    conv_param_, task_id);
  return RET_OK;
}

int ConvDwIndirectRun(void *cdata, int task_id) {
  auto conv_dw = reinterpret_cast<ConvolutionDepthwiseIndirectCPUKernel *>(cdata);
  auto ret = conv_dw->Execute(task_id);
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
  int pack_input_size = conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * div_flag * IC_DIV;
  packed_input_ = reinterpret_cast<float *>(context_->allocator->Malloc(pack_input_size * sizeof(float)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseIndirectCPUKernel::Run() {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto input_ptr = reinterpret_cast<float *>(input_tensor->data_c());
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

  if (IsTrain() && is_trainable()) {
    PackWeight();
  }

  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_ptr_ = reinterpret_cast<float *>(output_tensor->data_c());

  ConvDwInitIndirection(indirect_buffer_, packed_input_, zero_ptr_, conv_param_, step_h, step_w);

  auto ret = ParallelLaunch(this->context_->thread_pool_, ConvDwIndirectRun, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwIndirectRun error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  if (conv_param_->input_channel_ % div_flag != 0) {
    context_->allocator->Free(packed_input_);
  }
  return RET_OK;
}

void ConvolutionDepthwiseIndirectCPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_[kWeightIndex];
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->MutableData());
#ifdef ENABLE_AVX
  PackDepthwiseIndirectWeightC8Fp32(origin_weight, packed_weight_, weight_tensor->Height(), weight_tensor->Width(),
                                    weight_tensor->Batch());
#else
  PackDepthwiseIndirectWeightC4Fp32(origin_weight, packed_weight_, weight_tensor->Height(), weight_tensor->Width(),
                                    weight_tensor->Batch());
#endif
}

int ConvolutionDepthwiseIndirectCPUKernel::Eval() {
  LiteKernel::Eval();
  if (is_trainable()) {
    PackWeight();
  }
  return RET_OK;
}

}  // namespace mindspore::kernel
