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

#include "src/litert/kernel/cpu/fp32/convolution_depthwise_fp32.h"
#include "nnacl/intrinsics/ms_simd_cpu_info.h"
#include "include/errorcode.h"
#include "src/litert/pack_weight_manager.h"
#include "nnacl/fp32/conv_depthwise_avx_fp32.h"
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ConvolutionDepthwiseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  UpdateOriginWeightAndBias();
  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(weight_tensor);
    MS_CHECK_TRUE_MSG(weight_tensor->shape().size() == C4NUM, RET_ERROR, "Conv-like: weight-shape only support 4D.");
    MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_tensor->Height(), weight_tensor->Width(), RET_ERROR);
    int weight_size_hw = weight_tensor->Height() * weight_tensor->Width();
    MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_tensor->Batch(), weight_size_hw, RET_ERROR);
    int pack_weight_size = weight_tensor->Batch() * weight_size_hw;
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

int ConvolutionDepthwiseCPUKernel::InitConvDwCalcInfo() {
  if (conv_dw_calc_param_ == nullptr) {
    conv_dw_calc_param_ = new ConvDwCalcParam();
    CHECK_NULL_RETURN(conv_dw_calc_param_);
  }

  if (conv_dw_calc_param_->num_pixels_ != nullptr && !in_tensors_.at(kWeightIndex)->IsConst()) {
    free(conv_dw_calc_param_->num_pixels_);
    conv_dw_calc_param_->num_pixels_ = nullptr;
  }
  if (conv_dw_calc_param_->num_pixels_ == nullptr) {
    conv_dw_calc_param_->num_pixels_ = malloc(conv_param_->kernel_w_ * sizeof(int));
    CHECK_NULL_RETURN(conv_dw_calc_param_->num_pixels_);
  }

  if (conv_dw_calc_param_->out_w_start_ != nullptr && !in_tensors_.at(kWeightIndex)->IsConst()) {
    free(conv_dw_calc_param_->out_w_start_);
    conv_dw_calc_param_->out_w_start_ = nullptr;
  }
  if (conv_dw_calc_param_->out_w_start_ == nullptr) {
    conv_dw_calc_param_->out_w_start_ = malloc(conv_param_->kernel_w_ * sizeof(int));
    CHECK_NULL_RETURN(conv_dw_calc_param_->out_w_start_);
  }

  if (conv_dw_calc_param_->out_w_end_ != nullptr && !in_tensors_.at(kWeightIndex)->IsConst()) {
    free(conv_dw_calc_param_->out_w_end_);
    conv_dw_calc_param_->out_w_end_ = nullptr;
  }
  if (conv_dw_calc_param_->out_w_end_ == nullptr) {
    conv_dw_calc_param_->out_w_end_ = malloc(conv_param_->kernel_w_ * sizeof(int));
    CHECK_NULL_RETURN(conv_dw_calc_param_->out_w_end_);
  }

  int *num_pixels = reinterpret_cast<int *>(conv_dw_calc_param_->num_pixels_);
  int *out_w_start = reinterpret_cast<int *>(conv_dw_calc_param_->out_w_start_);
  int *out_w_end = reinterpret_cast<int *>(conv_dw_calc_param_->out_w_end_);
  conv_dw_calc_param_->first_calc_kw_ = -1;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->dilation_w_, (conv_param_->kernel_w_ - 1), RET_ERROR);
  for (int kw = 0; kw < conv_param_->kernel_w_; kw++) {
    out_w_start[kw] = MSMAX(
      0, (conv_param_->pad_l_ - conv_param_->dilation_w_ * kw + conv_param_->stride_w_ - 1) / conv_param_->stride_w_);
    out_w_end[kw] = MSMIN(conv_param_->output_w_, (conv_param_->input_w_ + conv_param_->pad_l_ -
                                                   conv_param_->dilation_w_ * kw + conv_param_->stride_w_ - 1) /
                                                    conv_param_->stride_w_);
    num_pixels[kw] = out_w_end[kw] - out_w_start[kw];
    if (conv_dw_calc_param_->first_calc_kw_ == -1 && out_w_start[kw] == 0 && num_pixels[kw] == conv_param_->output_w_) {
      conv_dw_calc_param_->first_calc_kw_ = kw;
    }
  }
  return RET_OK;
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
  ret = InitConvDwCalcInfo();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBaseCPUKernel::InitConvDwCalcInfo() return is:" << ret;
    return ret;
  }
  return RET_OK;
}

int ConvolutionDepthwiseCPUKernel::DoExecute(int task_id) {
  int ret;
#ifdef ENABLE_AVX512
  if (X86_Avx512_Support()) {
    ret = ConvDwAVX512(output_ptr_, input_ptr_, reinterpret_cast<float *>(packed_weight_),
                       reinterpret_cast<float *>(bias_data_), conv_param_, task_id, conv_dw_calc_param_);
  } else {
    ret = ConvDwAVX(output_ptr_, input_ptr_, reinterpret_cast<float *>(packed_weight_),
                    reinterpret_cast<float *>(bias_data_), conv_param_, task_id, conv_dw_calc_param_);
  }
#elif defined(ENABLE_AVX)
  ret = ConvDwAVX(output_ptr_, input_ptr_, reinterpret_cast<float *>(packed_weight_),
                  reinterpret_cast<float *>(bias_data_), conv_param_, task_id, conv_dw_calc_param_);
#else
  ret = ConvDw(output_ptr_, input_ptr_, reinterpret_cast<float *>(packed_weight_),
               reinterpret_cast<float *>(bias_data_), conv_param_, task_id);
#endif
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
  MS_CHECK_FALSE(conv_dw_calc_param_ == nullptr, RET_ERROR);
  MS_CHECK_FALSE(conv_dw_calc_param_->num_pixels_ == nullptr, RET_ERROR);
  MS_CHECK_FALSE(conv_dw_calc_param_->out_w_start_ == nullptr, RET_ERROR);
  MS_CHECK_FALSE(conv_dw_calc_param_->out_w_end_ == nullptr, RET_ERROR);

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
    packed_weight_ = GetConvPackWeightData(static_cast<size_t>(pack_weight_size) * sizeof(float));
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, channel * sizeof(float));
  if (bias_data_ == nullptr) {
    bias_data_ = malloc(channel * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, channel * sizeof(float));
  return RET_OK;
}
}  // namespace mindspore::kernel
