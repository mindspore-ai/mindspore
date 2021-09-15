/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32/convolution_delegate_fp32.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/convolution_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_1x1_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_winograd_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_depthwise_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_depthwise_slidewindow_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_depthwise_slidewindow_x86_fp32.h"
#include "src/runtime/kernel/arm/base/group_convolution_creator.h"
#include "src/runtime/kernel/arm/fp32/group_convolution_fp32.h"
#include "nnacl/base/conv_common_base.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
#include "src/runtime/kernel/arm/fp32/convolution_depthwise_3x3_fp32.h"
#endif
#if defined(ENABLE_ARM64)
#include "src/runtime/kernel/arm/fp32/convolution_depthwise_indirect_fp32.h"
#endif
#ifdef ENABLE_AVX
#include "src/runtime/kernel/arm/fp32/convolution_slidewindow_fp32.h"
#endif

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DFusion;

namespace mindspore::kernel {
namespace {
constexpr int kMaxDwConvSWSize = 32;
}  // namespace

float *ConvolutionDelegateCPUKernel::CopyData(const lite::Tensor *tensor) {
  auto data = reinterpret_cast<float *>(malloc(tensor->Size()));
  if (data == nullptr) {
    MS_LOG(ERROR) << "Malloc data failed.";
    return nullptr;
  }
  MS_ASSERT(tensor->data() != nullptr);
  memcpy(data, tensor->data(), tensor->Size());
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
  if (in_tensors_.at(kWeightIndex)->data() == nullptr) {
    return RET_OK;
  }
  if (InferShapeDone()) {
    origin_weight_ = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->data());
    CHECK_NULL_RETURN(origin_weight_);
    return RET_OK;
  }
  origin_weight_ = CopyData(in_tensors_.at(kWeightIndex));
  CHECK_NULL_RETURN(origin_weight_);
  need_free_weight_ = true;
  return RET_OK;
}

int ConvolutionDelegateCPUKernel::GetBiasData() {
  if (in_tensors_.size() == 3) {
    if (InferShapeDone()) {
      CHECK_NULL_RETURN(in_tensors_.at(kBiasIndex));
      origin_bias_ = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->data());
      CHECK_NULL_RETURN(origin_bias_);
      return RET_OK;
    } else {
      origin_bias_ = CopyData(in_tensors_.at(kBiasIndex));
      CHECK_NULL_RETURN(origin_bias_);
      need_free_bias_ = true;
      return RET_OK;
    }
  }
  return RET_OK;
}

int ConvolutionDelegateCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
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
  auto ret = SetInputOutputShapeInfo();
  if (ret != RET_OK) {
    return ret;
  }
  if (conv_kernel_ == nullptr) {
    // need to select actual execute kernel here
    conv_kernel_ = CpuConvFp32KernelSelect();
    if (conv_kernel_ == nullptr) {
      MS_LOG(ERROR) << "Selecting execute kernel failed for conv_kernel, got a nullptr.";
      return RET_ERROR;
    }
  }
  FreeCopiedData();
  return conv_kernel_->ReSize();
}

int ConvolutionDelegateCPUKernel::SetInputOutputShapeInfo() {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  CHECK_NULL_RETURN(conv_param);
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(output);
  conv_param->input_batch_ = input->Batch();
  conv_param->input_h_ = input->Height();
  conv_param->input_w_ = input->Width();
  conv_param->input_channel_ = input->Channel();
  conv_param->output_batch_ = output->Batch();
  conv_param->output_h_ = output->Height();
  conv_param->output_w_ = output->Width();
  conv_param->output_channel_ = output->Channel();
  conv_param->op_parameter_.thread_num_ = op_parameter_->thread_num_;
  return RET_OK;
}

bool ConvolutionDelegateCPUKernel::CheckAvxUseSWConv(const ConvParameter *conv_param) {
  if (conv_param->input_channel_ / op_parameter_->thread_num_ <= 64 &&
      conv_param->input_h_ >= conv_param->thread_num_ &&
      (conv_param->kernel_h_ < 7 || conv_param->input_h_ / conv_param->kernel_h_ >= 4) &&
      (conv_param->kernel_w_ < 7 || conv_param->input_w_ / conv_param->kernel_w_ >= 4)) {
    return true;
  } else {
    return false;
  }
}

kernel::InnerKernel *ConvolutionDelegateCPUKernel::CpuConvFp32KernelSelect() {
  kernel::InnerKernel *kernel = nullptr;
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
#ifdef ENABLE_AVX
    if (conv_param->pad_d_ == 0 && conv_param->pad_l_ == 0 && conv_param->pad_r_ == 0 && conv_param->pad_u_ == 0 &&
        conv_param->stride_h_ == 1 && conv_param->stride_w_ == 1 && conv_param->input_channel_ % 8 == 0 &&
        (conv_param->input_w_ * conv_param->input_h_ >= conv_param->thread_num_)) {
      kernel = new (std::nothrow) kernel::ConvolutionSWCPUKernel(
        op_parameter_, in_tensors_, out_tensors_, static_cast<const lite::InnerContext *>(this->ms_context_),
        origin_weight_, origin_bias_);
    } else {
      kernel = new (std::nothrow) kernel::Convolution1x1CPUKernel(
        op_parameter_, in_tensors_, out_tensors_, static_cast<const lite::InnerContext *>(this->ms_context_),
        origin_weight_, origin_bias_);
    }
#else
    kernel = new (std::nothrow) kernel::Convolution1x1CPUKernel(
      op_parameter_, in_tensors_, out_tensors_, static_cast<const lite::InnerContext *>(this->ms_context_),
      origin_weight_, origin_bias_);
#endif
  } else {
    int out_unit;
    if (CheckIfUseWinograd(&out_unit, conv_param)) {
      kernel = new (std::nothrow) kernel::ConvolutionWinogradCPUKernel(
        op_parameter_, in_tensors_, out_tensors_, static_cast<const lite::InnerContext *>(this->ms_context_), out_unit,
        origin_weight_, origin_bias_);
    } else {
#ifdef ENABLE_AVX
      if (CheckAvxUseSWConv(conv_param)) {
        kernel = new (std::nothrow) kernel::ConvolutionSWCPUKernel(
          op_parameter_, in_tensors_, out_tensors_, static_cast<const lite::InnerContext *>(this->ms_context_),
          origin_weight_, origin_bias_);
      } else {
        kernel = new (std::nothrow) kernel::ConvolutionCPUKernel(
          op_parameter_, in_tensors_, out_tensors_, static_cast<const lite::InnerContext *>(this->ms_context_),
          origin_weight_, origin_bias_);
      }
#else
      kernel = new (std::nothrow) kernel::ConvolutionCPUKernel(
        op_parameter_, in_tensors_, out_tensors_, static_cast<const lite::InnerContext *>(this->ms_context_),
        origin_weight_, origin_bias_);
#endif
    }
  }

  if (kernel != nullptr) {
    auto ret = kernel->Init();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "conv kernel init failed.";
      delete kernel;
      op_parameter_ = nullptr;
      return nullptr;
    }
  }

  kernel->set_name("act_" + name_);
  return kernel;
}

kernel::InnerKernel *CpuConvDwFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                const InnerContext *ctx) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Get null opParameter for CpuConvDwFp32KernelCreator.";
    return nullptr;
  }
  kernel::InnerKernel *kernel = nullptr;
  auto shape = outputs.front()->shape();
  if (std::find(shape.begin(), shape.end(), -1) == shape.end()) {
#ifdef ENABLE_AVX
    kernel = new (std::nothrow) kernel::ConvolutionDepthwiseSWCPUKernelX86(opParameter, inputs, outputs, ctx);
#else
    auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
    if (CheckConvDw1DWinograd(conv_param, ctx->thread_num_)) {
      kernel = new (std::nothrow) kernel::ConvolutionDepthwise3x3CPUKernel(opParameter, inputs, outputs, ctx);
    }
#endif
#if defined(ENABLE_ARM64)
    if (kernel == nullptr && CheckConvDwUseIndirectBuffer(conv_param)) {
      kernel = new (std::nothrow) kernel::ConvolutionDepthwiseIndirectCPUKernel(opParameter, inputs, outputs, ctx);
    }
#endif
    if (kernel == nullptr && conv_param->input_channel_ < kMaxDwConvSWSize) {
      kernel = new (std::nothrow) kernel::ConvolutionDepthwiseSWCPUKernel(opParameter, inputs, outputs, ctx);
    }
#endif
  }
  if (kernel == nullptr) {
    kernel = new (std::nothrow) kernel::ConvolutionDepthwiseCPUKernel(opParameter, inputs, outputs, ctx);
  }
  return kernel;
}

kernel::InnerKernel *CpuGroupConvFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs,
                                                   OpParameter *op_parameter, const lite::InnerContext *ctx) {
  auto *group_conv_creator = new GroupConvCreator(inputs, outputs, op_parameter, ctx, false, kNumberTypeFloat32);
  auto group_kernel = new (std::nothrow) GroupConvolutionFp32CPUKernel(
    op_parameter, inputs, outputs, ctx, group_conv_creator, reinterpret_cast<ConvParameter *>(op_parameter)->group_);
  if (group_kernel == nullptr) {
    MS_LOG(ERROR) << "New GroupConvolutionFp32CPUKernel failed.";
    return nullptr;
  }
  return group_kernel;
}

/* creator func */
kernel::InnerKernel *CpuConvFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                              const lite::Context *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(op_parameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DFusion);
  MS_ASSERT(desc.data_type == kNumberTypeFloat32);

  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::InnerKernel *kernel = nullptr;
  if (conv_param->group_ == 1) {
    kernel = new (std::nothrow)
      kernel::ConvolutionDelegateCPUKernel(op_parameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = CpuConvDwFp32KernelCreator(inputs, outputs, op_parameter, static_cast<const lite::InnerContext *>(ctx));
  } else {
    kernel = CpuGroupConvFp32KernelCreator(inputs, outputs, op_parameter, static_cast<const lite::InnerContext *>(ctx));
  }

  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(op_parameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2DFusion, CpuConvFp32KernelCreator)
}  // namespace mindspore::kernel
