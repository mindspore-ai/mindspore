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

#include "src/runtime/kernel/arm/fp16/convolution_depthwise_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/pack_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/cast_fp16.h"
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
ConvolutionDepthwiseFp16CPUKernel::~ConvolutionDepthwiseFp16CPUKernel() {
  delete sliding_;
  if (packed_weight_ != nullptr) {
    delete packed_weight_;
    packed_weight_ = nullptr;
  }
  if (packed_input_ != nullptr) {
    delete packed_input_;
    packed_input_ = nullptr;
  }
  if (packed_output_ != nullptr) {
    delete packed_output_;
    packed_output_ = nullptr;
  }
}

int ConvolutionDepthwiseFp16CPUKernel::InitBuffer() {
  // malloc pack input buffer
  int C8 = UP_DIV(conv_param_->input_channel_, C8NUM);
  int pack_input_size = conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * C8NUM * C8;
  packed_input_ = reinterpret_cast<float16_t *>(malloc(pack_input_size * sizeof(float16_t)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(packed_input_, 0, pack_input_size * sizeof(float16_t));

  // malloc pack output buffer
  int pack_output_size = conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * C8NUM * C8;
  packed_output_ = reinterpret_cast<float16_t *>(malloc(pack_output_size * sizeof(float16_t)));
  if (packed_output_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseFp16CPUKernel::InitWeightBias() {
  // init weight: o, h, w, i; o == group, i == 1
  int OC8 = UP_DIV(conv_param_->output_channel_, C8NUM);
  auto weight_tensor = in_tensors_[kWeightIndex];
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->Data());
  int pack_weight_size = C8NUM * OC8 * conv_param_->kernel_h_ * conv_param_->kernel_w_;

  packed_weight_ = reinterpret_cast<float16_t *>(malloc(pack_weight_size * sizeof(float16_t)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size * sizeof(float16_t));
  PackNCHWFp32ToNC8HW8Fp16(origin_weight, packed_weight_, 1, conv_param_->kernel_h_ * conv_param_->kernel_w_,
                           conv_param_->output_channel_);

  // init bias
  bias_data_ = reinterpret_cast<float16_t *>(malloc(C8NUM * OC8 * sizeof(float16_t)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, C8NUM * OC8 * sizeof(float16_t));
  auto bias_fp16 = reinterpret_cast<float16_t *>(bias_data_);
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->Data());
    for (int i = 0; i < conv_param_->output_channel_; i++) {
      bias_fp16[i] = (float16_t)ori_bias[i];
    }
  }

  conv_param_->thread_num_ = MSMIN(thread_count_, OC8);
  return RET_OK;
}

int ConvolutionDepthwiseFp16CPUKernel::Init() {
  // conv base init
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  // init sliding_ window param
  sliding_ = new SlidingWindowParam;
  InitSlidingParamConvDw(sliding_, conv_param_, C8NUM);

  ret = InitWeightBias();
  if (ret != 0) {
    MS_LOG(ERROR) << "Convolution depthwise fp16 InitWeightBias failed.";
    return RET_ERROR;
  }

  ret = InitBuffer();
  if (ret != 0) {
    MS_LOG(ERROR) << "Convolution depthwise fp16 InitBuffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseFp16CPUKernel::ReSize() {
  if (packed_input_ != nullptr) {
    delete packed_input_;
    packed_input_ = nullptr;
  }
  if (packed_output_ != nullptr) {
    delete packed_output_;
    packed_output_ = nullptr;
  }

  ConvolutionBaseCPUKernel::Init();
  InitSlidingParamConvDw(sliding_, conv_param_, C8NUM);

  auto ret = InitBuffer();
  if (ret != 0) {
    MS_LOG(ERROR) << "Convolution depthwise fp16 InitBuffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseFp16CPUKernel::Execute(int task_id) {
  ConvDwC8Fp16(packed_output_, packed_input_, packed_weight_, reinterpret_cast<float16_t *>(bias_data_), conv_param_,
               sliding_, task_id);
  return RET_OK;
}

int ConvDwFp16Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv_dw_fp16 = reinterpret_cast<ConvolutionDepthwiseFp16CPUKernel *>(cdata);
  auto ret = conv_dw_fp16->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseFp16Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseFp16CPUKernel::Run() {
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
  float16_t *input_addr;
  if (input_tensor->data_type() == kNumberTypeFloat32) {
    input_addr =
      reinterpret_cast<float16_t *>(context_->allocator->Malloc(input_tensor->ElementsNum() * sizeof(float16_t)));
    if (input_addr == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
    Float32ToFloat16(reinterpret_cast<float *>(input_tensor->Data()), input_addr, input_tensor->ElementsNum());
  } else {
    input_addr = reinterpret_cast<float16_t *>(input_tensor->Data());
  }

  // pack input: to nhwc8
  PackNHWCToNHWC8Fp16(input_addr, packed_input_, conv_param_->input_batch_,
                      conv_param_->input_h_ * conv_param_->input_w_, conv_param_->input_channel_);

  ret = LiteBackendParallelLaunch(ConvDwFp16Run, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwFp16Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }

  auto output_addr = reinterpret_cast<float16_t *>(out_tensors_.at(kOutputIndex)->Data());
  PackNHWC8ToNHWCFp16(packed_output_, output_addr, conv_param_->output_batch_,
                      conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);

  if (input_tensor->data_type() == kNumberTypeFloat32) {
    context_->allocator->Free(input_addr);
  }
  return RET_OK;
}

kernel::LiteKernel *CpuConvDwFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const Context *ctx,
                                               const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_DepthwiseConv2D);
  auto kernel = new (std::nothrow) ConvolutionDepthwiseFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_DepthwiseConv2D, CpuConvDwFp16KernelCreator)
}  // namespace mindspore::kernel
