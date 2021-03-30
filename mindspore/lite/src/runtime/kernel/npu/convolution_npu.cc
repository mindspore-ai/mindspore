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

#include "src/runtime/kernel/npu/convolution_npu.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
#include "src/runtime/kernel/npu/convolution_depthwise_npu.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Conv2DFusion;

namespace mindspore::kernel {
int ConvolutionNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs,
                                    const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter) {
  if (conv_param_->stride_h_ > inputs[0]->Height() || conv_param_->stride_w_ > inputs[0]->Width()) {
    MS_LOG(ERROR) << "Npu convolution does not support stride greater than input size.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionNPUKernel::SetConvParam() {
  conv_->set_attr_strides(ge::AttrValue::LIST_INT({conv_param_->stride_h_, conv_param_->stride_w_}));
  conv_->set_attr_dilations(ge::AttrValue::LIST_INT({conv_param_->dilation_h_, conv_param_->dilation_w_}));
  conv_->set_attr_groups(conv_param_->group_);

  if (conv_param_->pad_mode_ == Pad_same) {
    conv_->set_attr_pad_mode(ge::AttrValue::STR{"SAME"});
    conv_->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
  } else if (conv_param_->pad_mode_ == Pad_valid) {
    conv_->set_attr_pad_mode(ge::AttrValue::STR{"VALID"});
    conv_->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
  } else {
    conv_->set_attr_pad_mode(ge::AttrValue::STR{"SPECIFIC"});
    conv_->set_attr_pads(
      ge::AttrValue::LIST_INT({conv_param_->pad_u_, conv_param_->pad_d_, conv_param_->pad_l_, conv_param_->pad_r_}));
  }
  return RET_OK;
}

int ConvolutionNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                       const std::vector<lite::Tensor *> &outputs,
                                       const std::vector<ge::Operator *> &npu_inputs) {
  // set conv attr param
  conv_ = new (std::nothrow) hiai::op::Convolution(name_ + "_conv");
  if (conv_ == nullptr) {
    MS_LOG(ERROR) << "New convolution operator for convolution op " << name_ << " failed.";
    return RET_ERROR;
  }
  auto ret = SetConvParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set npu op parameter for convolution op " << name_ << " failed.";
    return RET_ERROR;
  }

  ret = InitWeightConst(inputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set weight and bias for convolution op " << name_ << " failed when running npu";
    return RET_ERROR;
  }
  conv_->set_input_filter(*weight_);
  if (inputs.size() == 3) {
    ret = InitBiasConst(inputs);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set bias for convolution op " << name_ << " failed when running npu";
      return RET_ERROR;
    }
    conv_->set_input_bias(*bias_);
  }
  conv_->set_input_x(*npu_inputs[0]);

  if (conv_param_->act_type_ != ActType_No) {
    ret = SetActivation(conv_, conv_param_->act_type_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

ge::Operator *mindspore::kernel::ConvolutionNPUKernel::GetNPUOp() {
  if (conv_param_->act_type_ == ActType_No) {
    return conv_;
  } else {
    return act_;
  }
}

ConvolutionNPUKernel::~ConvolutionNPUKernel() {
  if (conv_ != nullptr) {
    delete conv_;
    conv_ = nullptr;
  }
}

kernel::LiteKernel *NpuConvKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                         const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                         const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(op_parameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DFusion);
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::NPUKernel *kernel = nullptr;
  if (conv_param->group_ == 1) {
    kernel = new (std::nothrow) kernel::ConvolutionNPUKernel(op_parameter, inputs, outputs, ctx);
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = new (std::nothrow) kernel::ConvolutionDepthwiseNPUKernel(op_parameter, inputs, outputs, ctx);
  } else {
    MS_LOG(ERROR) << "npu do not support group conv!";
    kernel = nullptr;
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << op_parameter->name_ << "is nullptr.";
    free(op_parameter);
    return nullptr;
  }

  auto ret = kernel->IsSupport(inputs, outputs, op_parameter);
  if (ret != RET_OK) {
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Conv2DFusion, NpuConvKernelCreator)
}  // namespace mindspore::kernel
