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

#include "src/runtime/kernel/arm/fp32/deconvolution_depthwise.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DeDepthwiseConv2D;

namespace mindspore::kernel {
DeconvolutionDepthwiseCPUKernel::~DeconvolutionDepthwiseCPUKernel() {
  delete sliding_;
  if (packed_weight_ != nullptr) {
    delete packed_weight_;
    packed_weight_ = nullptr;
  }
  if (need_align_) {
    if (packed_input_ != nullptr) {
      delete packed_input_;
      packed_input_ = nullptr;
    }
    if (packed_output_ != nullptr) {
      delete packed_output_;
      packed_output_ = nullptr;
    }
  }
}

int DeconvolutionDepthwiseCPUKernel::InitSlideParam() {
  conv_param_->input_batch_ = out_tensors_.front()->shape().at(kNHWC_N);
  conv_param_->input_h_ = out_tensors_.front()->shape().at(kNHWC_H);
  conv_param_->input_w_ = out_tensors_.front()->shape().at(kNHWC_W);
  conv_param_->input_channel_ = out_tensors_.front()->shape().at(kNHWC_C);
  conv_param_->output_batch_ = in_tensors_.front()->shape().at(kNHWC_N);
  conv_param_->output_h_ = in_tensors_.front()->shape().at(kNHWC_H);
  conv_param_->output_w_ = in_tensors_.front()->shape().at(kNHWC_W);
  conv_param_->output_channel_ = in_tensors_.front()->shape().at(kNHWC_C);

  // init sliding window param
  sliding_ = new SlidingWindowParam;
  InitSlidingParamConvDw(sliding_, conv_param_, C4NUM);
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::InitWeightBias() {
  // init weight: o, h, w, i; o == group, i == 1
  auto weight_tensor = in_tensors_[kWeightIndex];
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->Data());
  int OC4 = UP_DIV(conv_param_->output_channel_, C4NUM);
  int pack_weight_size = C4NUM * OC4 * conv_param_->kernel_h_ * conv_param_->kernel_w_;

  packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size * sizeof(float));
  PackNCHWToNC4HW4Fp32(origin_weight, packed_weight_, 1, conv_param_->kernel_h_ * conv_param_->kernel_w_,
                       conv_param_->output_channel_);

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

  // init threadNum;
  conv_param_->thread_num_ = MSMIN(conv_param_->thread_num_, OC4);
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::InitBuffer() {
  // malloc pack input and output buffer
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
    memset(packed_output_, 0, pack_output_size * sizeof(float));
  }
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  InitSlideParam();
  // conv base init
  ConvolutionBaseCPUKernel::Init();

  auto ret = InitWeightBias();
  if (ret != 0) {
    MS_LOG(ERROR) << "Deconvolution depthwise fp32 InitWeightBias failed.";
    return RET_ERROR;
  }

  ret = InitBuffer();
  if (ret != 0) {
    MS_LOG(ERROR) << "Deconvolution depthwise fp32 InitBuffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::ReSize() {
  if (need_align_) {
    if (packed_input_ != nullptr) {
      delete packed_input_;
      packed_input_ = nullptr;
    }
    if (packed_output_ != nullptr) {
      delete packed_output_;
      packed_output_ = nullptr;
    }
  }
  InitSlideParam();

  // conv base init
  ConvolutionBaseCPUKernel::Init();

  auto ret = InitBuffer();
  if (ret != 0) {
    MS_LOG(ERROR) << "Deconvolution depthwise fp32 InitBuffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::Execute(int task_id) {
  DeconvDwC4Fp32(packed_output_, packed_input_, packed_weight_, reinterpret_cast<float *>(bias_data_), conv_param_,
                 sliding_, task_id);
  return RET_OK;
}

int DeconvDwRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto deconv_dw = reinterpret_cast<DeconvolutionDepthwiseCPUKernel *>(cdata);
  auto ret = deconv_dw->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DeconvolutionDepthwiseRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeconvolutionDepthwiseCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
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
    memset(output_addr, 0, out_tensors_.at(kOutputIndex)->ElementsNum() * sizeof(float));
    packed_output_ = output_addr;
  }

  auto ret = LiteBackendParallelLaunch(DeconvDwRun, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DeconvDwRun error: error_code[" << ret << "]";
    return RET_ERROR;
  }

  if (need_align_) {
    PackNHWC4ToNHWCFp32(packed_output_, output_addr, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
  }
  return RET_OK;
}

kernel::LiteKernel *CpuDeconvDwFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                 const std::vector<lite::tensor::Tensor *> &outputs,
                                                 OpParameter *opParameter, const lite::Context *ctx,
                                                 const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_DeDepthwiseConv2D);
  auto kernel =
    new (std::nothrow) kernel::DeconvolutionDepthwiseCPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DeDepthwiseConv2D, CpuDeconvDwFp32KernelCreator)
}  // namespace mindspore::kernel
