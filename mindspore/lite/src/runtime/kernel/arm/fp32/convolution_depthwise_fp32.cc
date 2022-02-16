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

#include "src/runtime/kernel/arm/fp32/convolution_depthwise_fp32.h"
#include "include/errorcode.h"
#ifdef SERVER_INFERENCE
#include "src/pack_weight_manager.h"
#endif
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ConvolutionDepthwiseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(weight_tensor);
    int pack_weight_size = weight_tensor->Batch() * weight_tensor->Height() * weight_tensor->Width();
    if (pack_weight_size >= std::numeric_limits<int>::max() / static_cast<int>(sizeof(float))) {
      MS_LOG(ERROR) << "pack_weight_size is invalid, pack_weight_size: " << pack_weight_size;
      return RET_ERROR;
    }
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

int ConvolutionDepthwiseCPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBaseCPUKernel::Prepare() return is:" << ret;
    return ret;
  }
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  if (conv_param_->thread_num_ <= 0) {
    MS_LOG(ERROR) << "conv_param_->thread_num_ must be greater than 0!";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseCPUKernel::DoExecute(int task_id) {
  auto ret = ConvDw(output_ptr_, input_ptr_, reinterpret_cast<float *>(packed_weight_),
                    reinterpret_cast<float *>(bias_data_), conv_param_, task_id);
  return ret;
}

int ConvDwRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv_dw = reinterpret_cast<ConvolutionDepthwiseCPUKernel *>(cdata);
  auto ret = conv_dw->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseCPUKernel::Run() {
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  input_ptr_ = reinterpret_cast<float *>(input_tensor->data());
  MS_CHECK_FALSE(input_ptr_ == nullptr, RET_ERROR);
  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_ptr_ = reinterpret_cast<float *>(output_tensor->data());
  MS_CHECK_FALSE(output_ptr_ == nullptr, RET_ERROR);

  auto ret = ParallelLaunch(this->ms_context_, ConvDwRun, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwRun error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

void ConvolutionDepthwiseCPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  void *origin_weight = (op_parameter_->is_train_session_) ? weight_tensor->data() : origin_weight_;
  MS_ASSERT(origin_weight != nullptr);
  PackWeightKHWToHWKFp32(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_),
                         weight_tensor->Height() * weight_tensor->Width(), weight_tensor->Batch());
}

int ConvolutionDepthwiseCPUKernel::MallocWeightBiasData() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  int channel = weight_tensor->Batch();
  MS_CHECK_TRUE_RET(channel > 0, RET_ERROR);
  int pack_weight_size = weight_tensor->Batch() * weight_tensor->Height() * weight_tensor->Width();
  if (pack_weight_size >= std::numeric_limits<int>::max() / static_cast<int>(sizeof(float))) {
    MS_LOG(ERROR) << "pack_weight_size is invalid, pack_weight_size: " << pack_weight_size;
    return RET_ERROR;
  }
  if (!op_parameter_->is_train_session_) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, pack_weight_size * sizeof(float));
#ifdef SERVER_INFERENCE
    auto packed = lite::PackWeightManager::GetInstance()->GetPackedTensor(
      in_tensors_[1], static_cast<size_t>(pack_weight_size) * sizeof(float));
    packed_weight_ = packed.second;
    weight_is_packed_ = packed.first;
    if (weight_is_packed_ == lite::MALLOC && packed_weight_ == nullptr) {
      packed_weight_ = malloc(pack_weight_size * sizeof(float));
    }
#else
    packed_weight_ = malloc(pack_weight_size * sizeof(float));
#endif
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, channel * sizeof(float));
  bias_data_ = malloc(channel * sizeof(float));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, channel * sizeof(float));
  return RET_OK;
}
}  // namespace mindspore::kernel
