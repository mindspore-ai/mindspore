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

#include "src/runtime/kernel/arm/int8/convolution_depthwise_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/int8/conv_depthwise_int8.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DepthwiseConv2D;

namespace mindspore::kernel {
void ConvolutionDepthwiseInt8CPUKernel::FreeTmpBuffer() {
  if (packed_input_ != nullptr) {
    free(packed_input_);
    packed_input_ = nullptr;
  }
  if (need_align_) {
    if (packed_output_ != nullptr) {
      free(packed_output_);
      packed_output_ = nullptr;
    }
  }
}

ConvolutionDepthwiseInt8CPUKernel::~ConvolutionDepthwiseInt8CPUKernel() {
  if (sliding != nullptr) {
    delete sliding;
    sliding = nullptr;
  }
  if (packed_weight_ != nullptr) {
    free(packed_weight_);
    packed_weight_ = nullptr;
  }
  FreeTmpBuffer();
  FreeQuantParam();
}

int ConvolutionDepthwiseInt8CPUKernel::InitWeightBias() {
  // init weight, int8 -> int16
  // o, h, w, i -> o/8, h, w, i, 8; o == group, i == 1
  auto weight_tensor = in_tensors_[kWeightIndex];
  auto origin_weight = reinterpret_cast<int8_t *>(weight_tensor->Data());
  int OC4 = UP_DIV(weight_tensor->Batch(), C4NUM);
  int pack_weight_size = C4NUM * OC4 * weight_tensor->Height() * weight_tensor->Width();
  packed_weight_ = reinterpret_cast<int16_t *>(malloc(pack_weight_size * sizeof(int16_t)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  PackDepthwiseInt8Weight(origin_weight, packed_weight_, weight_tensor->Height() * weight_tensor->Width(),
                          weight_tensor->Batch(), &(conv_param_->conv_quant_arg_));

  bias_data_ = reinterpret_cast<int32_t *>(malloc(C4NUM * OC4 * sizeof(int32_t)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, C4NUM * OC4 * sizeof(int32_t));
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    auto ori_bias = reinterpret_cast<int32_t *>(bias_tensor->Data());
    memcpy(bias_data_, ori_bias, bias_tensor->ElementsNum() * sizeof(int32_t));
  }

  conv_param_->thread_num_ = MSMIN(thread_count_, OC4);
  return RET_OK;
}

int ConvolutionDepthwiseInt8CPUKernel::InitBuffer() {
  int pack_input_size = conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * C4NUM *
                        UP_DIV(conv_param_->input_channel_, 4);
  packed_input_ = reinterpret_cast<int16_t *>(malloc(pack_input_size * sizeof(int16_t)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }

  if (conv_param_->input_channel_ % C4NUM != 0) {
    need_align_ = true;
    int pack_output_size = conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * C4NUM *
                           UP_DIV(conv_param_->output_channel_, C4NUM);
    packed_output_ = reinterpret_cast<int8_t *>(malloc(pack_output_size * sizeof(int8_t)));
    if (packed_input_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ConvolutionDepthwiseInt8CPUKernel::Init() {
  sliding = new (std::nothrow) SlidingWindowParam;
  if (sliding == nullptr) {
    MS_LOG(ERROR) << "new sliding window param.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDepthwiseInt8CPUKernel::ReSize() {
  FreeTmpBuffer();
  ConvolutionBaseCPUKernel::Init();
  InitSlidingParamConvDw(sliding, conv_param_, C4NUM);

  auto ret = ConvolutionBaseCPUKernel::SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
  }
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise int8 InitWeightBias error!";
    return ret;
  }
  ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise int8 ReSize error!";
    return ret;
  }
  return RET_OK;
}

int ConvolutionDepthwiseInt8CPUKernel::Execute(int task_id) {
  ConvDwInt8(packed_output_, packed_input_, packed_weight_, reinterpret_cast<int32_t *>(bias_data_), conv_param_,
             sliding, task_id);
  return RET_OK;
}

int ConvDwInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv_dw_int8 = reinterpret_cast<ConvolutionDepthwiseInt8CPUKernel *>(cdata);
  auto ret = conv_dw_int8->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  if (conv_param_->input_channel_ != conv_param_->output_channel_) {
    MS_LOG(ERROR) << "Only support input channel equals output channel.";
    return RET_ERROR;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  auto input_addr = reinterpret_cast<int8_t *>(input_tensor->Data());
  PackDepthwiseInt8Input(input_addr, packed_input_, conv_param_);

  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(kOutputIndex)->Data());
  if (!need_align_) {
    packed_output_ = output_addr;
  }

  ret = LiteBackendParallelLaunch(ConvDwInt8Run, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwInt8Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }

  if (need_align_) {
    PackNHWC4ToNHWCInt8(packed_output_, output_addr, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
  }
  return RET_OK;
}

kernel::LiteKernel *CpuConvDwInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const Context *ctx,
                                               const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_DepthwiseConv2D);
  auto kernel =
    new (std::nothrow) kernel::ConvolutionDepthwiseInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_DepthwiseConv2D, CpuConvDwInt8KernelCreator)
}  // namespace mindspore::kernel
