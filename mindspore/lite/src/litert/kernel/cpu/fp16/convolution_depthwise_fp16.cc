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

#include "src/litert/kernel/cpu/fp16/convolution_depthwise_fp16.h"
#include "nnacl/fp16/pack_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
void ConvolutionDepthwiseFp16CPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  void *origin_weight = (op_parameter_->is_train_session_) ? weight_tensor->data() : origin_weight_;
  CHECK_NULL_RETURN_VOID(origin_weight);
  PackNCHWToNHWCFp16(reinterpret_cast<float16_t *>(origin_weight), reinterpret_cast<float16_t *>(packed_weight_), 1,
                     weight_tensor->Height() * weight_tensor->Width(), weight_tensor->Batch(), 0, 0);
}

int ConvolutionDepthwiseFp16CPUKernel::MallocWeightBiasData() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  int channel = weight_tensor->Batch();
  MS_CHECK_TRUE_RET(channel > 0, RET_ERROR);
  int pack_weight_size = channel * weight_tensor->Height() * weight_tensor->Width();
  if (!op_parameter_->is_train_session_) {
    if (packed_weight_ == nullptr) {
      CHECK_LESS_RETURN(MAX_MALLOC_SIZE, pack_weight_size * sizeof(float16_t));
      packed_weight_ = reinterpret_cast<float16_t *>(malloc(pack_weight_size * sizeof(float16_t)));
      if (packed_weight_ == nullptr) {
        MS_LOG(ERROR) << "Malloc buffer failed.";
        return RET_ERROR;
      }
    }
  }
  if (bias_data_ == nullptr) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, channel * sizeof(float16_t));
    bias_data_ = malloc(channel * sizeof(float16_t));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, channel * sizeof(float16_t));
  return RET_OK;
}

int ConvolutionDepthwiseFp16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  UpdateOriginWeightAndBias();
  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(weight_tensor);
    MS_CHECK_TRUE_MSG(weight_tensor->shape().size() == C4NUM, RET_ERROR, "Conv-like: weight-shape only support 4D.");
    int channel = weight_tensor->Batch();
    int pack_weight_size = channel * weight_tensor->Height() * weight_tensor->Width();
    set_workspace_size(pack_weight_size * sizeof(float16_t));
  }
  auto ret = InitConvWeightBias();
  if (ret != 0) {
    MS_LOG(ERROR) << "Convolution depthwise fp16 InitConvWeightBias failed.";
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDepthwiseFp16CPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  return RET_OK;
}

int ConvolutionDepthwiseFp16CPUKernel::DoExecute(int task_id) {
  auto input_ptr = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data());
  auto output_ptr = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data());
  MS_ASSERT(input_ptr != nullptr);
  MS_ASSERT(output_ptr != nullptr);
  if (input_ptr == nullptr || output_ptr == nullptr) {
    MS_LOG(ERROR) << "Convolution depthwise Fp16 get null tensor data!";
    return RET_ERROR;
  }
  ConvDwFp16(output_ptr, input_ptr, reinterpret_cast<float16_t *>(packed_weight_),
             reinterpret_cast<float16_t *>(bias_data_), conv_param_, task_id);
  return RET_OK;
}

static int ConvDwFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv_dw_fp16 = reinterpret_cast<ConvolutionDepthwiseFp16CPUKernel *>(cdata);
  auto ret = conv_dw_fp16->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseFp16Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseFp16CPUKernel::Run() {
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->ms_context_, ConvDwFp16Run, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwFp16Run error: error_code[" << ret << "]";
  }
  return ret;
}
}  // namespace mindspore::kernel
