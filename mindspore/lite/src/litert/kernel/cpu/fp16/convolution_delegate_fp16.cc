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

#include "src/litert/kernel/cpu/fp16/convolution_delegate_fp16.h"
#include <vector>
#include "src/litert/kernel/cpu/fp16/convolution_fp16.h"
#include "src/litert/kernel/cpu/fp16/convolution_winograd_fp16.h"
#include "src/litert/kernel/cpu/fp16/convolution_1x1_fp16.h"
#include "src/litert/kernel/cpu/fp16/group_convolution_fp16.h"
#include "src/litert/kernel/cpu/fp16/convolution_depthwise_fp16.h"
#include "src/litert/kernel/cpu/fp16/convolution_depthwise_slidewindow_fp16.h"
#include "src/litert/kernel/cpu/fp16/convolution_depthwise_3x3_fp16.h"
#include "src/litert/kernel/cpu/base/group_convolution_creator.h"
#include "nnacl/base/conv_common_base.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DFusion;

namespace mindspore::kernel {
void ConvolutionDelegateFP16CPUKernel::FreeCopiedData() {
  if ((origin_weight_ != nullptr) && (need_free_ & WEIGHT_NEED_FREE)) {
    free(origin_weight_);
    origin_weight_ = nullptr;
    need_free_ = need_free_ & ~WEIGHT_NEED_FREE;
  }
  if ((origin_bias_ != nullptr) && (need_free_ & BIAS_NEED_FREE)) {
    free(origin_bias_);
    origin_bias_ = nullptr;
    need_free_ = need_free_ & ~BIAS_NEED_FREE;
  }
}

void *ConvolutionDelegateFP16CPUKernel::CopyData(const lite::Tensor *tensor) {
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
  MS_ASSERT(tensor->data() != nullptr);
  memcpy(copied_data, tensor->data(), tensor->Size());
  return copied_data;
}

int ConvolutionDelegateFP16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(weight_tensor);
    origin_weight_ = weight_tensor->data() != nullptr ? CopyData(weight_tensor) : nullptr;
    need_free_ = need_free_ | WEIGHT_NEED_FREE;
    if (in_tensors_.size() == C3NUM) {
      origin_bias_ = CopyData(in_tensors_.at(kBiasIndex));
      need_free_ = need_free_ | BIAS_NEED_FREE;
    }
    return RET_OK;
  }
  origin_weight_ = in_tensors_.at(kWeightIndex)->data();
  if (in_tensors_.size() == C3NUM) {
    origin_bias_ = in_tensors_.at(kBiasIndex)->data();
    MS_ASSERT(origin_bias_ != nullptr);
  }
  return ReSize();
}

static void SetInputOutputShapeInfo(ConvParameter *conv_param, const lite::Tensor *input, const lite::Tensor *output,
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
  CHECK_NULL_RETURN(in_tensors_.front());
  CHECK_NULL_RETURN(out_tensors_.front());
  // Update shape info of input and output
  kernel::SetInputOutputShapeInfo(reinterpret_cast<ConvParameter *>(op_parameter_), in_tensors_.front(),
                                  out_tensors_.front(), static_cast<const lite::InnerContext *>(this->ms_context_));
  if (fp16_conv_kernel_ == nullptr) {
    fp16_conv_kernel_ =
      CpuConvFp16KernelSelect(in_tensors_, out_tensors_, op_parameter_,
                              static_cast<const lite::InnerContext *>(ms_context_), origin_weight_, origin_bias_);
    if (fp16_conv_kernel_ == nullptr) {
      MS_LOG(ERROR) << "Selecting execute kernel failed for conv_kernel, got a nullptr.";
      return RET_ERROR;
    }
  }
  // copied weight and bias are not be used anymore,free them.
  FreeCopiedData();
  auto ret = fp16_conv_kernel_->ReSize();
  set_workspace_size(fp16_conv_kernel_->workspace_size());
  return ret;
}

bool ConvolutionDelegateFP16CPUKernel::CheckInputsValid() const {
  // the data type of input and weight must be the same, while the bias data type of int8 convolution is int32.
  MS_CHECK_TRUE_RET(in_tensors_.size() >= kInputSize1, false);
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  MS_CHECK_TRUE_RET(input_tensor != nullptr && weight_tensor != nullptr, false);
  MS_CHECK_TRUE_RET(input_tensor->data() != nullptr, false);
  return input_tensor->data_type() == weight_tensor->data_type();
}

kernel::LiteKernel *CpuConvDwFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const InnerContext *ctx) {
  MS_ASSERT(opParameter != nullptr);
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  kernel::LiteKernel *kernel = nullptr;
#if defined(ENABLE_ARM)
  if (CheckConvDw1DWinograd(conv_param, ctx->thread_num_)) {
    kernel = new (std::nothrow) kernel::ConvolutionDepthwise3x3Fp16CPUKernel(opParameter, inputs, outputs, ctx);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "kernel is nullptr.";
      free(opParameter);
      return nullptr;
    }
    return kernel;
  }
#endif
  if (conv_param->input_channel_ < C32NUM) {
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

kernel::LiteKernel *ConvolutionDelegateFP16CPUKernel::CpuConvFp16KernelSelect(
  const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
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
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr";
    free(op_parameter);
    return nullptr;
  }
  kernel->set_name(this->name());

  // Once kernel is selected, init func will invoke InitWeightAndBias
  auto ret = kernel->Prepare();
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
  auto *group_conv_creator =
    new (std::nothrow) GroupConvCreator(inputs, outputs, op_parameter, false, kNumberTypeFloat16);
  if (group_conv_creator == nullptr) {
    MS_LOG(ERROR) << "new GroupConvCreator fail";
    free(op_parameter);
    return nullptr;
  }
  auto kernel = new (std::nothrow) GroupConvolutionFP16CPUKernel(
    op_parameter, inputs, outputs, ctx, group_conv_creator, reinterpret_cast<ConvParameter *>(op_parameter)->group_);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new GroupConvolutionFP16CPUKernel fail";
    free(op_parameter);
  }
  return kernel;
}

/* creator func */
kernel::LiteKernel *CpuConvFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DFusion);
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->group_ == 1) {
    kernel = new (std::nothrow) kernel::ConvolutionDelegateFP16CPUKernel(opParameter, inputs, outputs, ctx);
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = CpuConvDwFp16KernelCreator(inputs, outputs, opParameter, ctx);
  } else {
    kernel = CpuGroupConvFp16KernelCreator(inputs, outputs, opParameter, ctx);
  }

  if (conv_param->group_ == 1 && kernel == nullptr) {
    MS_LOG(DEBUG) << "Create conv fp16 kernel failed.";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Conv2DFusion, CpuConvFp16KernelCreator)
}  // namespace mindspore::kernel
