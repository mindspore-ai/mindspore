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

#include "src/runtime/kernel/arm/fp16/convolution_fp16.h"
#include <vector>
#include "include/errorcode.h"
#include "nnacl/fp16/conv_fp16.h"
#include "nnacl/fp16/matmul_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/pack_fp16.h"
#include "nnacl/fp16/winograd_utils_fp16.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ConvolutionFP16CPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int in_channel = filter_tensor->Channel();
  int out_channel = filter_tensor->Batch();
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;
  int oc8 = UP_ROUND(out_channel, col_tile_);
  int kernel_plane = filter_tensor->Height() * filter_tensor->Width();
  int pack_weight_size = oc8 * in_channel * kernel_plane;

  // init weight
  if (packed_weight_ == nullptr) {
    packed_weight_ = reinterpret_cast<float16_t *>(malloc(pack_weight_size * sizeof(float16_t)));
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "malloc packed_weight_ failed.";
      return RET_ERROR;
    }
  }
  memset(packed_weight_, 0, pack_weight_size * sizeof(float16_t));
  void *weight_origin_tmp = IsTrainable() ? filter_tensor->data_c() : origin_weight_;
  RowMajor2Col8MajorFp16(weight_origin_tmp, packed_weight_, out_channel, in_channel * kernel_plane, false);

  // init bias
  if (bias_data_ == nullptr) {
    bias_data_ = malloc(oc8 * sizeof(float16_t));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias_data_ failed.";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, oc8 * sizeof(float16_t));
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    void *bias_origin_tmp = IsTrainable() ? bias_tensor->data_c() : origin_bias_;
    memcpy(bias_data_, bias_origin_tmp, out_channel * sizeof(float16_t));
  }
  return RET_OK;
}

int ConvolutionFP16CPUKernel::InitTmpBuffer() {
  int unit_size =
    conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ * row_tile_ * thread_count_;

  packed_input_ = reinterpret_cast<float16_t *>(ctx_->allocator->Malloc(unit_size * sizeof(float16_t)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_input_ failed.";
    return RET_ERROR;
  }

  col_major_input_ = reinterpret_cast<float16_t *>(ctx_->allocator->Malloc(unit_size * sizeof(float16_t)));
  if (col_major_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc col_major_input_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionFP16CPUKernel::Init() {
#ifdef ENABLE_ARM64
  row_tile_ = C16NUM;
#else
  row_tile_ = C12NUM;
#endif
  col_tile_ = C8NUM;
  auto ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void ConvolutionFP16CPUKernel::AdjustNumberOfThread() {
  auto out_tensor = out_tensors_.front();
  int out_plane = out_tensor->Height() * out_tensor->Width();
  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(out_plane, row_tile_));
  conv_param_->thread_num_ = thread_count_;
}

int ConvolutionFP16CPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }
  ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init fail!ret: " << ret;
    return ret;
  }
  return RET_OK;
}

int ConvolutionFP16CPUKernel::RunImpl(int task_id) {
  auto input_ptr = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data_c());
  auto output_ptr = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data_c());
  MS_ASSERT(input_ptr != nullptr);
  MS_ASSERT(output_ptr != nullptr);
  if (input_ptr == nullptr || output_ptr == nullptr) {
    MS_LOG(ERROR) << "Convolution Fp16 get null tensor data!";
    return RET_ERROR;
  }
  ConvFp16(input_ptr, packed_input_, packed_weight_, reinterpret_cast<float16_t *>(bias_data_), col_major_input_,
           output_ptr, task_id, conv_param_);
  return RET_OK;
}

static int ConvolutionFp16Impl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv = reinterpret_cast<ConvolutionFP16CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionFp16 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionFP16CPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  if (IsTrainable() && (IsTrain() || IsRepack())) {
    ret = InitWeightBias();
    if (ret != 0) {
      MS_LOG(ERROR) << "Convolution 1x1 fp16 repack weight failure";
      return RET_ERROR;
    }
    is_repack_ = false;
  }
  ret = ParallelLaunch(this->ms_context_, ConvolutionFp16Impl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv fp16 error ret[" << ret << "]";
  }

  FreeTmpBuffer();
  return ret;
}

int ConvolutionFP16CPUKernel::Eval() {
  if (IsTrainable()) {
    is_repack_ = true;
  }
  return InnerKernel::Eval();
}
}  // namespace mindspore::kernel
