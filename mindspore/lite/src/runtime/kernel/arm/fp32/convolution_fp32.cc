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

#include "src/runtime/kernel/arm/fp32/convolution_fp32.h"
#include "include/errorcode.h"
#include "nnacl/common_func.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "nnacl/fp32/conv_common_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
#ifdef ENABLE_AVX
#define OC_BLOCK C16NUM
#elif ENABLE_ARM32
#define OC_BLOCK C4NUM
#else
#define OC_BLOCK C8NUM
#endif

int ConvolutionCPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int in_channel = filter_tensor->Channel();
  int out_channel = filter_tensor->Batch();
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;
  int kernel_plane = filter_tensor->Height() * filter_tensor->Width();
  int oc_block_num = UP_ROUND(out_channel, OC_BLOCK);
  int pack_weight_size = oc_block_num * in_channel * kernel_plane;

  packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed weight failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size * sizeof(float));
#ifdef ENABLE_AVX
  RowMajor2Col16Major(origin_weight_, packed_weight_, out_channel, in_channel * kernel_plane);
#elif ENABLE_ARM32
  RowMajor2Col4Major(origin_weight_, packed_weight_, out_channel, in_channel * kernel_plane);
#else
  RowMajor2Col8Major(origin_weight_, packed_weight_, out_channel, in_channel * kernel_plane);
#endif

  bias_data_ = reinterpret_cast<float *>(malloc(oc_block_num * sizeof(float)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc_block_num * sizeof(float));

  if (in_tensors_.size() == kInputSize2) {
    memcpy(bias_data_, origin_bias_, out_channel * sizeof(float));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  return RET_OK;
}

int ConvolutionCPUKernel::InitTmpBuffer() {
  MS_ASSERT(ctx_->allocator != nullptr);

#ifdef ENABLE_AVX
  int unit_size = conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ * C6NUM * thread_count_;
#elif ENABLE_SSE
  int unit_size = conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ * C4NUM * thread_count_;
#else
  int unit_size =
    conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ * C12NUM * thread_count_;
#endif
  packed_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(unit_size * sizeof(float)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed input failed.";
    return RET_ERROR;
  }

  col_major_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(unit_size * sizeof(float)));
  if (col_major_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc col_major_input_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::Init() {
  auto ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }
  ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv base init failed.";
    return ret;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::RunImpl(int task_id) {
  auto ori_input_data = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->data_c());
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->data_c());
  ConvFp32(ori_input_data, packed_input_, packed_weight_, reinterpret_cast<float *>(bias_data_), col_major_input_,
           output_addr, task_id, conv_param_);
  return RET_OK;
}

int ConvolutionImpl(void *cdata, int task_id) {
  auto conv = reinterpret_cast<ConvolutionCPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  if (IsTrain() && is_trainable()) {
    PackWeight();
  }

  ret = ParallelLaunch(this->context_->thread_pool_, ConvolutionImpl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv error error_code[" << ret << "]";
  }
  FreeTmpBuffer();
  return ret;
}

void ConvolutionCPUKernel::PackWeight() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int in_channel = filter_tensor->Channel();
  int out_channel = filter_tensor->Batch();
  int kernel_plane = filter_tensor->Height() * filter_tensor->Width();
  int oc_block_num = UP_ROUND(out_channel, OC_BLOCK);
  int pack_weight_size = oc_block_num * in_channel * kernel_plane;

  auto origin_weight = reinterpret_cast<float *>(filter_tensor->data_c());
  memset(packed_weight_, 0, pack_weight_size * sizeof(float));
#ifdef ENABLE_AVX
  RowMajor2Col16Major(origin_weight, packed_weight_, out_channel, in_channel * kernel_plane);
#elif ENABLE_ARM32
  RowMajor2Col4Major(origin_weight, packed_weight_, out_channel, in_channel * kernel_plane);
#else
  RowMajor2Col8Major(origin_weight, packed_weight_, out_channel, in_channel * kernel_plane);
#endif
}

int ConvolutionCPUKernel::Eval() {
  LiteKernel::Eval();
  if (is_trainable()) {
    PackWeight();
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
