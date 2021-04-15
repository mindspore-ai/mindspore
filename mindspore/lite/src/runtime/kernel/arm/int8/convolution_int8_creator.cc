/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/int8/convolution_int8_creator.h"
#include "src/runtime/kernel/arm/int8/convolution_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_1x1_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_3x3_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_depthwise_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_depthwise_3x3_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_depthwise_slidewindow_int8.h"
#include "src/runtime/kernel/arm/int8/group_convolution_int8.h"
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
kernel::LiteKernel *CpuConvDwInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                               const InnerContext *ctx, const kernel::KernelKey &desc) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::LiteKernel *kernel = nullptr;

  auto act_quant_size =
    MSMAX(inputs.at(kInputIndex)->quant_params().size(), outputs.at(kOutputIndex)->quant_params().size());
  if (act_quant_size == 1) {  // per tensor
    if (CheckConvDwUse3X3(conv_param) && conv_param->input_channel_ % C8NUM == 0) {
#ifdef ENABLE_ARM64
      kernel = new (std::nothrow) kernel::ConvolutionDepthwise3x3Int8CPUKernel(op_parameter, inputs, outputs, ctx);
#endif
    }
    if (kernel == nullptr) {
      kernel = new (std::nothrow) kernel::ConvolutionDepthwiseInt8CPUKernel(op_parameter, inputs, outputs, ctx);
    }
  } else {  // per channel
    kernel = new (std::nothrow) kernel::ConvolutionDepthwiseSWInt8CPUKernel(op_parameter, inputs, outputs, ctx);
  }
  return kernel;
}

/* Kernel creator func part */
kernel::LiteKernel *CpuConvInt8KernelSelect(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                            const InnerContext *ctx) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->kernel_h_ == 3 && conv_param->kernel_w_ == 3 && conv_param->stride_h_ == 1 &&
      conv_param->stride_w_ == 1 && conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1) {
#ifdef ENABLE_ARM64
    if (mindspore::lite::IsSupportSDot()) {
      kernel = new (std::nothrow) ConvolutionInt8CPUKernel(op_parameter, inputs, outputs, ctx);
    } else {
      kernel = new (std::nothrow) Convolution3x3Int8CPUKernel(op_parameter, inputs, outputs, ctx);
    }
#else
    kernel = new (std::nothrow) kernel::Convolution3x3Int8CPUKernel(op_parameter, inputs, outputs, ctx);
#endif
  } else if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
    kernel = new (std::nothrow) Convolution1x1Int8CPUKernel(op_parameter, inputs, outputs, ctx);
  } else {
    kernel = new (std::nothrow) ConvolutionInt8CPUKernel(op_parameter, inputs, outputs, ctx);
  }
  return kernel;
}

kernel::LiteKernel *CpuGroupConvInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                                  const lite::InnerContext *ctx, int group) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  GroupConvCreator group_conv_creator(inputs, outputs, op_parameter, ctx, true, kNumberTypeInt8);
  group_conv_creator.SetShapeOfTensors();
  for (int i = 0; i < conv_param->group_; ++i) {
    ConvParameter *new_conv_param = CreateNewConvParameter(conv_param);
    std::vector<lite::Tensor *> new_inputs;
    std::vector<lite::Tensor *> new_outputs;
    auto ret = group_conv_creator.GetSingleConvParam(new_conv_param, &new_inputs, &new_outputs, i);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "GetSingleConv for int8 group conv failed.";
      return nullptr;
    }
    group_conv_creator.CopyQuantParam(&new_inputs);
    group_conv_creator.get_group_conv()->emplace_back(
      CpuConvInt8KernelSelect(new_inputs, new_outputs, reinterpret_cast<OpParameter *>(new_conv_param), ctx));
  }
  return new (std::nothrow)
    GroupConvolutionInt8CPUKernel(op_parameter, inputs, outputs, ctx, *(group_conv_creator.get_group_conv()), group);
}

kernel::LiteKernel *CpuConvInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                             const InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(op_parameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DFusion);
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::LiteKernel *kernel = nullptr;

  if (conv_param->group_ == 1) {
    kernel = CpuConvInt8KernelSelect(inputs, outputs, op_parameter, ctx);
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = CpuConvDwInt8KernelCreator(inputs, outputs, op_parameter, ctx, desc);
  } else {
    MS_ASSERT(conv_param->group_ > 1);
    kernel = CpuGroupConvInt8KernelCreator(inputs, outputs, op_parameter, ctx, conv_param->group_);
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(op_parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Conv2DFusion, CpuConvInt8KernelCreator)
}  // namespace mindspore::kernel
