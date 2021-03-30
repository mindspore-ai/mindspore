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

#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/convolution_delegate_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_creator_manager.h"
#include "src/runtime/kernel/arm/fp32/convolution_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_1x1_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_winograd_fp32.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DFusion;

namespace mindspore::kernel {
float *ConvolutionDelegateCPUKernel::CopyData(lite::Tensor *tensor) {
  auto data = reinterpret_cast<float *>(malloc(tensor->Size()));
  if (data == nullptr) {
    MS_LOG(ERROR) << "Malloc data failed.";
    return nullptr;
  }
  memcpy(data, tensor->data_c(), tensor->Size());
  return data;
}

int ConvolutionDelegateCPUKernel::GetWeightAndBias() {
  auto ret = GetWeightData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get weight data failed.";
    return ret;
  }
  ret = GetBiasData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get bias data failed.";
    return ret;
  }
  return RET_OK;
}

int ConvolutionDelegateCPUKernel::GetWeightData() {
  if (InferShapeDone()) {
    origin_weight_ = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->data_c());
    return RET_OK;
  }
  origin_weight_ = CopyData(in_tensors_.at(kWeightIndex));
  if (origin_weight_ == nullptr) {
    MS_LOG(ERROR) << "Copy weight data failed.";
    return RET_ERROR;
  }
  need_free_weight_ = true;
  return RET_OK;
}

int ConvolutionDelegateCPUKernel::GetBiasData() {
  if (in_tensors_.size() == 3) {
    if (InferShapeDone()) {
      origin_bias_ = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->data_c());
      return RET_OK;
    } else {
      origin_bias_ = CopyData(in_tensors_.at(kBiasIndex));
      if (origin_bias_ == nullptr) {
        MS_LOG(ERROR) << "Copy bias data failed.";
        return RET_ERROR;
      }
      need_free_bias_ = true;
      return RET_OK;
    }
  }
  return RET_OK;
}

int ConvolutionDelegateCPUKernel::Init() {
  auto ret = GetWeightAndBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get weight and bias failed.";
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDelegateCPUKernel::ReSize() {
  // Update shape info of input and output
  SetInputOutputShapeInfo();
  if (conv_kernel_ == nullptr) {
    // need to select actual execute kernel here
    conv_kernel_ = CpuConvFp32KernelSelect();
    if (!conv_kernel_) {
      MS_LOG(ERROR) << "Selecting execute kernel failed for conv_kernel, got a nullptr.";
      return RET_ERROR;
    }
    conv_kernel_->set_name(this->name_);
  }
  FreeCopiedData();
  return conv_kernel_->ReSize();
}

void ConvolutionDelegateCPUKernel::SetInputOutputShapeInfo() {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  conv_param->input_batch_ = input->Batch();
  conv_param->input_h_ = input->Height();
  conv_param->input_w_ = input->Width();
  conv_param->input_channel_ = input->Channel();
  conv_param->output_batch_ = output->Batch();
  conv_param->output_h_ = output->Height();
  conv_param->output_w_ = output->Width();
  conv_param->output_channel_ = output->Channel();
  conv_param->op_parameter_.thread_num_ = context_->thread_num_;
}

kernel::LiteKernel *ConvolutionDelegateCPUKernel::CpuConvFp32KernelSelect() {
  kernel::LiteKernel *kernel = nullptr;
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
    kernel = new (std::nothrow)
      kernel::Convolution1x1CPUKernel(op_parameter_, in_tensors_, out_tensors_, context_, origin_weight_, origin_bias_);
  } else {
    int out_unit;
    if (CheckIfUseWinograd(&out_unit, conv_param)) {
      kernel = new (std::nothrow) kernel::ConvolutionWinogradCPUKernel(
        op_parameter_, in_tensors_, out_tensors_, context_, out_unit, origin_weight_, origin_bias_);
    } else {
      kernel = new (std::nothrow)
        kernel::ConvolutionCPUKernel(op_parameter_, in_tensors_, out_tensors_, context_, origin_weight_, origin_bias_);
    }
  }

  if (kernel != nullptr) {
    auto ret = kernel->Init();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "conv kernel init failed.";
      delete kernel;
      return nullptr;
    }
  }
  return kernel;
}

/* creator func */
kernel::LiteKernel *CpuConvFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                             const InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(op_parameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DFusion);
  MS_ASSERT(desc.data_type == kNumberTypeFloat32);

  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->group_ == 1) {
    kernel = new (std::nothrow) kernel::ConvolutionDelegateCPUKernel(op_parameter, inputs, outputs, ctx);
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = DispatchConvDw(inputs, outputs, op_parameter, ctx);
  } else {
    kernel = DispatchGroupConv(inputs, outputs, op_parameter, ctx);
  }

  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(op_parameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK && ret != RET_INFER_INVALID) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2DFusion, CpuConvFp32KernelCreator)
}  // namespace mindspore::kernel
