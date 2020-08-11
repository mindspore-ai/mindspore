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

#include "src/runtime/kernel/arm/fp32/convolution_depthwise_3x3.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DepthwiseConv2D;

namespace mindspore::kernel {
int ConvolutionDepthwise3x3CPUKernel::InitWeightBias() {
  // init weight: o, h, w, i; o == group, i == 1
  auto weight_tensor = in_tensors_[kWeightIndex];
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->Data());
  // o h w 1 -> o/4 h w 1 4
  int OC4 = UP_DIV(conv_param_->output_channel_, C4NUM);
  int weight_c4_size = OC4 * C4NUM * 9;
  auto tmp_weight = reinterpret_cast<float *>(malloc(weight_c4_size * sizeof(float)));
  if (tmp_weight == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(tmp_weight, 0, weight_c4_size * sizeof(float));
  PackNCHWToNC4HW4Fp32(origin_weight, tmp_weight, 1, conv_param_->kernel_h_ * conv_param_->kernel_w_,
                       conv_param_->output_channel_);

  // weight transform
  int packed_weight_size = OC4 * C4NUM * 16;
  packed_weight_ = reinterpret_cast<float *>(malloc(packed_weight_size * sizeof(float)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, packed_weight_size * sizeof(float));
  ConvDw3x3Fp32FilterTrans(packed_weight_, tmp_weight, OC4);

  // init bias
  bias_data_ = reinterpret_cast<float *>(malloc(C4NUM * OC4 * sizeof(float)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, C4NUM * OC4 * sizeof(float));
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->Data());
    memcpy(bias_data_, ori_bias, conv_param_->output_channel_ * sizeof(float));
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::InitBuffer() {
  if (conv_param_->input_channel_ % C4NUM != 0) {
    need_align_ = true;
    int IC4 = UP_DIV(conv_param_->input_channel_, C4NUM);
    int pack_input_size = conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * C4NUM * IC4;
    packed_input_ = reinterpret_cast<float *>(malloc(pack_input_size * sizeof(float)));
    if (packed_input_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
    memset(packed_input_, 0, pack_input_size * sizeof(float));

    int OC4 = UP_DIV(conv_param_->output_channel_, C4NUM);
    int pack_output_size = conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * C4NUM * OC4;
    packed_output_ = reinterpret_cast<float *>(malloc(pack_output_size * sizeof(float)));
    if (packed_output_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }

  // malloc transform buffer
  trans_size_ = UP_DIV(conv_param_->output_w_, 2) * UP_DIV(conv_param_->output_h_, 2) * 16 * C4NUM;
  size_t trans_buffer_size = thread_count_ * trans_size_ * sizeof(float);
  trans_buffer_ = reinterpret_cast<float *>(malloc(trans_buffer_size));
  if (trans_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc trans buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  // conv base init
  ConvolutionBaseCPUKernel::Init();

  auto ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise3x3 fp32 initWeightBias error!";
    return ret;
  }

  // init threadNum;
  conv_param_->thread_num_ = MSMIN(thread_count_, UP_DIV(conv_param_->output_channel_, C4NUM));

  ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise3x3 fp32 initBuffer error!";
    return ret;
  }

  // malloc one block buffer
  block_buffer_ = reinterpret_cast<float *>(malloc(thread_count_ * 16 * C4NUM * sizeof(float)));
  if (block_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc block buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::ReSize() {
  if (need_align_) {
    free(packed_input_);
    free(packed_output_);
  }
  free(trans_buffer_);

  // conv base init
  ConvolutionBaseCPUKernel::Init();

  auto ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise3x3 fp32 initBuffer error!";
    return ret;
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::Execute(int task_id) {
  auto trans_buf = trans_buffer_ + task_id * trans_size_;
  auto block_buf = block_buffer_ + task_id * 16 * C4NUM;
  ConvDw3x3Fp32(packed_output_, packed_input_, packed_weight_, reinterpret_cast<float *>(bias_data_), trans_buf,
                block_buf, conv_param_, task_id);
  return RET_OK;
}

int ConvDw3x3Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv_dw_3x3 = reinterpret_cast<ConvolutionDepthwise3x3CPUKernel *>(cdata);
  auto ret = conv_dw_3x3->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwise3x3Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwise3x3CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return ret;
  }
  if (conv_param_->input_channel_ != conv_param_->output_channel_) {
    MS_LOG(ERROR) << "Only support input channel equals output channel.";
    return RET_ERROR;
  }
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto input_addr = reinterpret_cast<float *>(input_tensor->Data());

  // pack input: to nhwc4
  if (need_align_) {
    PackNHWCToNHWC4Fp32(input_addr, packed_input_, conv_param_->input_batch_,
                        conv_param_->input_h_ * conv_param_->input_w_, conv_param_->input_channel_);
  } else {
    packed_input_ = input_addr;
  }

  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->Data());
  if (!need_align_) {
    packed_output_ = output_addr;
  }

  ret = LiteBackendParallelLaunch(ConvDw3x3Run, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDw3x3Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }

  if (need_align_) {
    PackNHWC4ToNHWCFp32(packed_output_, output_addr, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
