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

#include "src/litert/kernel/cpu/fp32/convolution_depthwise_3x3_fp32.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"

#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ConvolutionDepthwise3x3CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  UpdateOriginWeightAndBias();
  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(weight_tensor);
    MS_CHECK_TRUE_MSG(weight_tensor->shape().size() == C4NUM, RET_ERROR, "Conv-like: weight-shape only support 4D.");
    int channel = weight_tensor->Batch();
    int c4 = UP_ROUND(channel, C4NUM);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(c4, C12NUM, RET_ERROR);
    int pack_weight_size = c4 * C12NUM;
    set_workspace_size(pack_weight_size * sizeof(float));
  }
  auto ret = InitConvWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Convolution depthwise 3x3 fp32 InitConvWeightBias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDepthwise3x3CPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBaseCPUKernel::Prepare() return is:" << ret;
    return ret;
  }
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::DoExecute(int task_id) {
  int units = UP_DIV(conv_param_->output_w_, C2NUM);  // F(2, 3) contains 2 conv units
  int c4 = UP_ROUND(conv_param_->input_channel_, C4NUM);
  int c12c4_units = C12NUM * c4 * units;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(c12c4_units, task_id, RET_ERROR);
  auto buffer = buffer_ + c12c4_units * task_id;
  if (conv_param_->thread_num_ == 0) {
    MS_LOG(ERROR) << "conv_param_->thread_num_ must be not equal to 0";
    return RET_ERROR;
  }
  int step_oh = UP_DIV(conv_param_->output_h_, conv_param_->thread_num_);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(step_oh, task_id, RET_ERROR);
  int start_oh = step_oh * task_id;
  int end_oh = MSMIN(start_oh + step_oh, conv_param_->output_h_);
  ConvDw3x3(output_ptr_, buffer, input_ptr_, reinterpret_cast<float *>(packed_weight_),
            reinterpret_cast<float *>(bias_data_), conv_param_, start_oh, end_oh);
  return RET_OK;
}

int ConvDw3x3Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv_dw = reinterpret_cast<ConvolutionDepthwise3x3CPUKernel *>(cdata);
  auto ret = conv_dw->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwise3x3Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::Run() {
  int units = UP_DIV(conv_param_->output_w_, C2NUM);  // F(2, 3) contains 2 conv units
  int c4 = UP_ROUND(conv_param_->input_channel_, C4NUM);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(C12NUM, c4, RET_ERROR);
  int c12c4 = C12NUM * c4;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(c12c4, units, RET_ERROR);
  int c12c4_units = c12c4 * units;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(c12c4_units, conv_param_->thread_num_, RET_ERROR);
  int buffer_size = c12c4_units * conv_param_->thread_num_;
  buffer_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(buffer_size * sizeof(float)));
  if (buffer_ == nullptr) {
    MS_LOG(ERROR) << "ConvDw3x3Run failed to allocate buffer";
    return RET_MEMORY_FAILED;
  }
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    ctx_->allocator->Free(buffer_);
    return RET_ERROR;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  input_ptr_ = reinterpret_cast<float *>(input_tensor->data());
  CHECK_NULL_RETURN(input_ptr_);
  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_ptr_ = reinterpret_cast<float *>(output_tensor->data());
  CHECK_NULL_RETURN(output_ptr_);
  auto ret = ParallelLaunch(this->ms_context_, ConvDw3x3Run, this, conv_param_->thread_num_);
  ctx_->allocator->Free(buffer_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDw3x3Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

void ConvolutionDepthwise3x3CPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  int channel = weight_tensor->Batch();
  void *origin_weight = (op_parameter_->is_train_session_) ? weight_tensor->data() : origin_weight_;
  MS_ASSERT(origin_weight != nullptr);
  PackWeightConvDw3x3Fp32(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_), channel);
}

int ConvolutionDepthwise3x3CPUKernel::MallocWeightBiasData() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  int channel = weight_tensor->Batch();
  MS_CHECK_TRUE_RET(channel > 0, RET_ERROR);
  int c4 = UP_ROUND(channel, C4NUM);
  int pack_weight_size = c4 * C12NUM;
  if (!op_parameter_->is_train_session_) {
    if (packed_weight_ == nullptr) {
      CHECK_LESS_RETURN(MAX_MALLOC_SIZE, pack_weight_size * sizeof(float));
      packed_weight_ = GetConvPackWeightData(pack_weight_size * sizeof(float));
      if (packed_weight_ == nullptr) {
        MS_LOG(ERROR) << "Malloc buffer failed.";
        return RET_ERROR;
      }
    }
  }

  if (bias_data_ == nullptr) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, c4 * sizeof(float));
    bias_data_ = malloc(c4 * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, c4 * sizeof(float));
  return RET_OK;
}
}  // namespace mindspore::kernel
#endif
