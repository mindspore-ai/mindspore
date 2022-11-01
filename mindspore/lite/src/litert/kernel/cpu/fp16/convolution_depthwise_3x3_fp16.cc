/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifdef ENABLE_ARM
#include "src/litert/kernel/cpu/fp16/convolution_depthwise_3x3_fp16.h"
#include "include/errorcode.h"
#include "nnacl/fp16/pack_fp16.h"
#include "nnacl/fp16/conv_depthwise_fp16.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
void ConvolutionDepthwise3x3Fp16CPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  int channel = weight_tensor->Batch();
  void *origin_weight = (op_parameter_->is_train_session_) ? weight_tensor->data() : origin_weight_;
  CHECK_NULL_RETURN_VOID(origin_weight);
  PackWeightConvDw3x3Fp16(reinterpret_cast<float16_t *>(origin_weight), reinterpret_cast<float16_t *>(packed_weight_),
                          channel);
}

int ConvolutionDepthwise3x3Fp16CPUKernel::MallocWeightBiasData() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  int channel = weight_tensor->Batch();
  MS_CHECK_TRUE_RET(channel > 0, RET_ERROR);
  int c8 = UP_ROUND(channel, C8NUM);
  int pack_weight_size = c8 * C12NUM;
  if (!op_parameter_->is_train_session_) {
    if (packed_weight_ == nullptr) {
      CHECK_LESS_RETURN(MAX_MALLOC_SIZE, pack_weight_size * sizeof(float16_t));
      packed_weight_ = malloc(pack_weight_size * sizeof(float16_t));
      if (packed_weight_ == nullptr) {
        MS_LOG(ERROR) << "Malloc buffer failed.";
        return RET_ERROR;
      }
    }
  }
  if (bias_data_ == nullptr) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, c8 * sizeof(float16_t));
    bias_data_ = malloc(c8 * sizeof(float16_t));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, c8 * sizeof(float16_t));
  return RET_OK;
}

int ConvolutionDepthwise3x3Fp16CPUKernel::Prepare() {
  UpdateOriginWeightAndBias();
  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(weight_tensor);
    MS_CHECK_TRUE_MSG(weight_tensor->shape().size() == C4NUM, RET_ERROR, "Conv-like: weight-shape only support 4D.");
    int channel = weight_tensor->Batch();
    int c8 = UP_ROUND(channel, C8NUM);
    int pack_weight_size = c8 * C12NUM;
    set_workspace_size(pack_weight_size * sizeof(float16_t));
  }
  auto ret = InitConvWeightBias();
  if (ret != 0) {
    MS_LOG(ERROR) << "Convolution depthwise 3x3 fp16 InitConvWeightBias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDepthwise3x3Fp16CPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBaseCPUKernel::Prepare() failed!";
    return ret;
  }
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  return RET_OK;
}

int ConvolutionDepthwise3x3Fp16CPUKernel::DoExecute(int task_id) {
  int units = UP_DIV(conv_param_->output_w_, C2NUM);  // F(2, 3) contains 2 conv units
  int c8 = UP_ROUND(conv_param_->input_channel_, C8NUM);
  auto buffer = buffer_ + C12NUM * c8 * units * task_id;
  int step_oh = UP_DIV(conv_param_->output_h_, conv_param_->thread_num_);
  int start_oh = step_oh * task_id;
  int end_oh = MSMIN(start_oh + step_oh, conv_param_->output_h_);
  ConvDw3x3Fp16(output_ptr_, buffer, input_ptr_, reinterpret_cast<float16_t *>(packed_weight_),
                reinterpret_cast<float16_t *>(bias_data_), conv_param_, start_oh, end_oh);
  return RET_OK;
}

int ConvDw3x3Fp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv_dw = reinterpret_cast<ConvolutionDepthwise3x3Fp16CPUKernel *>(cdata);
  auto ret = conv_dw->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwise3x3Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3Fp16CPUKernel::Run() {
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }

  int units = UP_DIV(conv_param_->output_w_, C2NUM);  // F(2, 3) contains 2 conv units
  int c8 = UP_ROUND(conv_param_->input_channel_, C8NUM);
  int buffer_size = units * c8 * C12NUM * conv_param_->thread_num_;
  buffer_ = reinterpret_cast<float16_t *>(ctx_->allocator->Malloc(buffer_size * sizeof(float16_t)));
  if (buffer_ == nullptr) {
    MS_LOG(ERROR) << "ConvDw3x3Fp16Run failed to allocate buffer";
    return RET_MEMORY_FAILED;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  CHECK_NULL_RETURN(input_tensor);
  input_ptr_ = reinterpret_cast<float16_t *>(input_tensor->data());
  CHECK_NULL_RETURN(input_ptr_);
  auto output_tensor = out_tensors_.at(kOutputIndex);
  CHECK_NULL_RETURN(output_tensor);
  output_ptr_ = reinterpret_cast<float16_t *>(output_tensor->data());
  CHECK_NULL_RETURN(output_ptr_);
  auto ret = ParallelLaunch(this->ms_context_, ConvDw3x3Fp16Run, this, conv_param_->thread_num_);
  ctx_->allocator->Free(buffer_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDw3x3Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
#endif
