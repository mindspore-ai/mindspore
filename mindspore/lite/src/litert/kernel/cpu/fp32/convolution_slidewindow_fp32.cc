/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp32/convolution_slidewindow_fp32.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"
#include "nnacl/fp32/conv_common_fp32.h"
#include "nnacl/fp32/conv_1x1_x86_fp32.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
void ConvolutionSWCPUKernel::InitGlobalVariable() {
  oc_tile_ = C1NUM;
  oc_res_ = conv_param_->output_channel_ % oc_tile_;
  if (conv_param_->kernel_h_ == 1 && conv_param_->kernel_w_ == 1) {
    // 1x1 conv is aligned to C1NUM
    in_tile_ = C1NUM;
    ic_res_ = conv_param_->input_channel_ % in_tile_;
  }
}

int ConvolutionSWCPUKernel::Prepare() {
  InitGlobalVariable();

  if (op_parameter_->is_train_session_) {
    auto filter_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(filter_tensor);
    MS_CHECK_TRUE_MSG(filter_tensor->shape().size() == C4NUM, RET_ERROR, "Conv-like: weight-shape only support 4D.");
    auto input_channel = filter_tensor->Channel();
    auto output_channel = filter_tensor->Batch();
    int kernel_h = filter_tensor->Height();
    int kernel_w = filter_tensor->Width();
    MS_CHECK_INT_MUL_NOT_OVERFLOW(kernel_h, kernel_w, RET_ERROR);
    int kernel_hw = kernel_h * kernel_w;
    int oc_block_num = UP_DIV(output_channel, oc_tile_);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(input_channel, kernel_hw, RET_ERROR);
    int kernel_chw = input_channel * kernel_hw;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(oc_block_num * oc_tile_, kernel_chw, RET_ERROR);
    int pack_weight_size = oc_block_num * oc_tile_ * kernel_chw;
    set_workspace_size(pack_weight_size * sizeof(float));
  }
  auto ret = InitConvWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  // is not 1x1 conv
  if (!(conv_param_->kernel_h_ == 1 && conv_param_->kernel_w_ == 1)) {
    conv_param_->out_format_ = out_tensors_[0]->format();
  }
  return ReSize();
}

int ConvolutionSWCPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  if (slidingWindow_param_ != nullptr) {
    delete slidingWindow_param_;
    slidingWindow_param_ = nullptr;
  }

  ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase prepare failed.";
    return RET_ERROR;
  }
  // init sliding window param
  slidingWindow_param_ = new (std::nothrow) SlidingWindowParam;
  if (slidingWindow_param_ == nullptr) {
    MS_LOG(ERROR) << "new SlidingWindowParam fail!";
    return RET_ERROR;
  }
  InitSlidingParamConv(slidingWindow_param_, conv_param_, in_tile_, oc_tile_);
  return RET_OK;
}

int ConvolutionSWCPUKernel::RunImpl(int task_id) {
  MS_LOG(ERROR) << "new SlidingWindow run fail, do not support slidewindows fp32 implement!";
  return RET_ERROR;
}

int ConvolutionSWImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv = reinterpret_cast<ConvolutionSWCPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution Sliding Window Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionSWCPUKernel::InitTmpBuffer() {
  CHECK_NULL_RETURN(in_tensors_.at(kInputIndex));
  auto input_data = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->MutableData());
  CHECK_NULL_RETURN(input_data);
  if (ic_res_ != 0 && conv_param_->kernel_h_ == 1 && conv_param_->kernel_w_ == 1) {
    // 1x1 conv input is align to in_tile
    int in_channel = conv_param_->input_channel_;
    int ic_block_num = UP_DIV(in_channel, in_tile_);
    MS_ASSERT(ctx_->allocator != nullptr);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->input_h_, conv_param_->input_w_, RET_ERROR);
    int input_hw = conv_param_->input_h_ * conv_param_->input_w_;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->input_batch_, input_hw, RET_ERROR);
    int input_bhw = conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(input_bhw, ic_block_num * in_tile_, RET_ERROR);
    input_data_ =
      reinterpret_cast<float *>(ctx_->allocator->Malloc(input_bhw * ic_block_num * in_tile_ * sizeof(float)));
    if (input_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc tmp input_data_ failed.";
      return RET_NULL_PTR;
    }
    PackNHWCToNHWCXFp32(input_data, input_data_, conv_param_->input_batch_, input_hw, conv_param_->input_channel_,
                        oc_tile_);
  } else {
    input_data_ = input_data;
  }

  auto out_data = reinterpret_cast<float *>(out_tensors_.front()->MutableData());
  CHECK_NULL_RETURN(out_data);
  if (oc_res_ == 0) {  // not need to malloc dst
    output_data_ = out_data;
  } else {  // need to malloc dst to align block
    int out_channel = conv_param_->output_channel_;
    int oc_block_num = UP_DIV(out_channel, oc_tile_);
    MS_ASSERT(ctx_->allocator != nullptr);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_h_, conv_param_->output_w_, RET_ERROR);
    int output_hw = conv_param_->output_h_ * conv_param_->output_w_;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_batch_, output_hw, RET_ERROR);
    int output_bhw = conv_param_->output_batch_ * output_hw;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(output_bhw, oc_block_num * oc_tile_, RET_ERROR);
    output_data_ =
      reinterpret_cast<float *>(ctx_->allocator->Malloc(output_bhw * oc_block_num * oc_tile_ * sizeof(float)));
    if (output_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc tmp output data failed.";
      return RET_NULL_PTR;
    }
  }
  return RET_OK;
}

int ConvolutionSWCPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitTmpBuffer error!";
    FreeTmpBuffer();
    return ret;
  }

  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }

  int error_code = ParallelLaunch(this->ms_context_, ConvolutionSWImpl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv error error_code[" << error_code << "]";
    FreeTmpBuffer();
    return error_code;
  }
  if (oc_res_ != 0) {
    auto out_data = reinterpret_cast<float *>(out_tensors_.front()->MutableData());
    CHECK_NULL_RETURN(out_data);
    PackNHWCXToNHWCFp32(output_data_, out_data, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_, oc_tile_);
  }
  FreeTmpBuffer();
  return RET_OK;
}

void ConvolutionSWCPUKernel::PackWeight() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();
  int kernel_h = filter_tensor->Height();
  int kernel_w = filter_tensor->Width();
  int oc_block_num = UP_DIV(output_channel, oc_tile_);
  void *origin_weight = (op_parameter_->is_train_session_) ? filter_tensor->data() : origin_weight_;
  MS_ASSERT(origin_weight != nullptr);
  PackNHWCToNXHWCXFp32(kernel_h, kernel_w, output_channel, oc_block_num, input_channel,
                       reinterpret_cast<float *>(packed_weight_), reinterpret_cast<float *>(origin_weight));
}

int ConvolutionSWCPUKernel::MallocWeightBiasData() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();
  int kernel_h = filter_tensor->Height();
  int kernel_w = filter_tensor->Width();
  MS_CHECK_TRUE_RET(input_channel > 0 && output_channel > 0 && kernel_h > 0 && kernel_w > 0, RET_ERROR);
  conv_param_->input_channel_ = input_channel;
  conv_param_->output_channel_ = output_channel;
  int kernel_plane = kernel_h * kernel_w;
  int oc_block_num = UP_DIV(output_channel, oc_tile_);
  int pack_weight_size = oc_block_num * oc_tile_ * input_channel * kernel_plane;
  if (!op_parameter_->is_train_session_) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, pack_weight_size * sizeof(float));
    packed_weight_ = GetConvPackWeightData(pack_weight_size * sizeof(float));
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "malloc packed weight failed.";
      return RET_NULL_PTR;
    }
  }

  if (in_tensors_.size() == kInputSize2) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, oc_block_num * oc_tile_ * sizeof(float));
    bias_data_ = malloc(oc_block_num * oc_tile_ * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias failed.";
      return RET_NULL_PTR;
    }
    memset(bias_data_, 0, oc_block_num * oc_tile_ * sizeof(float));
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
#endif  // ENABLE_AVX
