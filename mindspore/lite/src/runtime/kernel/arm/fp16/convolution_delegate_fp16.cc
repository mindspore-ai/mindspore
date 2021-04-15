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

#include "src/runtime/kernel/arm/fp16/convolution_delegate_fp16.h"
#include <vector>
#include "src/runtime/kernel/arm/fp16/convolution_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_winograd_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_1x1_fp16.h"
#include "src/runtime/kernel/arm/fp16/group_convolution_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_depthwise_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_depthwise_slidewindow_fp16.h"
#include "src/runtime/kernel/arm/base/group_convolution_creator.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DFusion;
using mindspore::schema::Format::Format_NHWC;

namespace mindspore::kernel {
void ConvolutionDelegateFP16CPUKernel::FreeCopiedData() {
  if ((origin_weight_ != nullptr) && (need_free_ & WEIGHT_NEED_FREE)) {
    free(origin_weight_);
    origin_weight_ = nullptr;
  }
  if ((origin_bias_ != nullptr) && (need_free_ & BIAS_NEED_FREE)) {
    free(origin_bias_);
    origin_bias_ = nullptr;
  }
}

void *ConvolutionDelegateFP16CPUKernel::CopyData(lite::Tensor *tensor) {
  auto data_type = tensor->data_type();
  if (data_type != kNumberTypeFloat32 && data_type != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Not supported data type: " << data_type;
    return nullptr;
  }
  auto copied_data = malloc(tensor->Size());
  if (copied_data == nullptr) {
    MS_LOG(ERROR) << "Malloc copied_data failed.";
    return nullptr;
  }
  memcpy(copied_data, tensor->data_c(), tensor->Size());
  return copied_data;
}

int ConvolutionDelegateFP16CPUKernel::Init() {
  if (!InferShapeDone()) {
    origin_weight_ = CopyData(in_tensors_.at(kWeightIndex));
    need_free_ = need_free_ | WEIGHT_NEED_FREE;
    if (in_tensors_.size() == 3) {
      origin_bias_ = CopyData(in_tensors_.at(kBiasIndex));
      need_free_ = need_free_ | BIAS_NEED_FREE;
    }
    return RET_OK;
  }
  origin_weight_ = in_tensors_.at(kWeightIndex)->data_c();
  if (in_tensors_.size() == 3) {
    origin_bias_ = in_tensors_.at(kBiasIndex)->data_c();
  }
  return ReSize();
}

static void SetInputOutputShapeInfo(ConvParameter *conv_param, lite::Tensor *input, lite::Tensor *output,
                                    const InnerContext *ctx) {
  conv_param->input_batch_ = input->Batch();
  conv_param->input_h_ = input->Height();
  conv_param->input_w_ = input->Width();
  conv_param->input_channel_ = input->Channel();
  conv_param->output_batch_ = output->Batch();
  conv_param->output_h_ = output->Height();
  conv_param->output_w_ = output->Width();
  conv_param->output_channel_ = output->Channel();
  conv_param->op_parameter_.thread_num_ = ctx->thread_num_;
}

int ConvolutionDelegateFP16CPUKernel::ReSize() {
  // Update shape info of input and output
  kernel::SetInputOutputShapeInfo(reinterpret_cast<ConvParameter *>(op_parameter_), in_tensors_.front(),
                                  out_tensors_.front(), context_);
  if (fp16_conv_kernel_ == nullptr) {
    fp16_conv_kernel_ =
      CpuConvFp16KernelSelect(in_tensors_, out_tensors_, op_parameter_, context_, origin_weight_, origin_bias_);
    if (fp16_conv_kernel_ == nullptr) {
      MS_LOG(ERROR) << "Selecting execute kernel failed for conv_kernel, got a nullptr.";
      return RET_ERROR;
    }
  }
  // copied weight and bias are not be used anymore,free them.
  FreeCopiedData();
  return fp16_conv_kernel_->ReSize();
}

kernel::LiteKernel *CpuConvDwFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const InnerContext *ctx) {
  MS_ASSERT(opParameter != nullptr);
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->input_channel_ < 32) {
    kernel = new (std::nothrow) kernel::ConvolutionDepthwiseSWFp16CPUKernel(opParameter, inputs, outputs, ctx);
  } else {
    kernel = new (std::nothrow) kernel::ConvolutionDepthwiseFp16CPUKernel(opParameter, inputs, outputs, ctx);
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuConvFp16KernelSelect(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                            const lite::InnerContext *ctx, void *origin_weight, void *origin_bias) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  bool use_winograd = false;
  int out_unit;
  CheckIfUseWinogradFp16(&use_winograd, &out_unit, conv_param);
  kernel::LiteKernel *kernel = nullptr;

  if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
    kernel = new (std::nothrow)
      kernel::Convolution1x1FP16CPUKernel(op_parameter, inputs, outputs, ctx, origin_weight, origin_bias);
  } else if (use_winograd) {
    kernel = new (std::nothrow) kernel::ConvolutionWinogradFP16CPUKernel(op_parameter, inputs, outputs, ctx, out_unit,
                                                                         origin_weight, origin_bias);
  } else {
    kernel = new (std::nothrow)
      kernel::ConvolutionFP16CPUKernel(op_parameter, inputs, outputs, ctx, origin_weight, origin_bias);
  }
  // Once kernel is selected, init func will invoke InitWeightAndBias
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "kernel init failed.";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuGroupConvFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                                  const InnerContext *ctx) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  GroupConvCreator group_conv_creator(inputs, outputs, op_parameter, ctx, false, kNumberTypeFloat16);
  group_conv_creator.SetShapeOfTensors();

  for (int i = 0; i < conv_param->group_; ++i) {
    ConvParameter *new_conv_param = CreateNewConvParameter(conv_param);
    std::vector<lite::Tensor *> new_inputs;
    std::vector<lite::Tensor *> new_outputs;
    auto ret = group_conv_creator.GetSingleConvParam(new_conv_param, &new_inputs, &new_outputs, i);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "GetSingleConv for fp16 group conv failed.";
      return nullptr;
    }
    group_conv_creator.get_group_conv()->emplace_back(new (std::nothrow) ConvolutionDelegateFP16CPUKernel(
      reinterpret_cast<OpParameter *>(new_conv_param), new_inputs, new_outputs, ctx));
  }
  return new (std::nothrow)
    GroupConvolutionFP16CPUKernel(op_parameter, inputs, outputs, ctx, *(group_conv_creator.get_group_conv()),
                                  reinterpret_cast<ConvParameter *>(op_parameter)->group_);
}

/* creator func */
kernel::LiteKernel *CpuConvFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DFusion);

  auto weight_data_type = inputs.at(1)->data_type();
  TypeId bias_data_type = weight_data_type;
  if (inputs.size() == 3) {
    bias_data_type = inputs.at(2)->data_type();
  }
  if (weight_data_type != kNumberTypeFloat16 || bias_data_type != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Convfp16 only support fp16 weight and fp16 bias.";
    return nullptr;
  }
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->group_ == 1) {
    kernel = new (std::nothrow) kernel::ConvolutionDelegateFP16CPUKernel(opParameter, inputs, outputs, ctx);
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = CpuConvDwFp16KernelCreator(inputs, outputs, opParameter, ctx);
  } else {
    kernel = CpuGroupConvFp16KernelCreator(inputs, outputs, opParameter, ctx);
  }

  if (kernel == nullptr) {
    MS_LOG(DEBUG) << "Create conv fp16 kernel failed.";
    free(opParameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(INFO) << "Init fp16 kernel failed, name: " << opParameter->name_
                 << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Conv2DFusion, CpuConvFp16KernelCreator)
}  // namespace mindspore::kernel
