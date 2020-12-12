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

#include "src/runtime/kernel/npu/convolution_depthwise_npu.h"
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_DepthwiseConv2D;

namespace mindspore::kernel {
int ConvolutionDepthwiseNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter) {
  return RET_OK;
}

int ConvolutionDepthwiseNPUKernel::SetConvDwParam() {
  conv_dw_->set_attr_strides(ge::AttrValue::LIST_INT({conv_param_->stride_h_, conv_param_->stride_w_}));
  conv_dw_->set_attr_dilations(ge::AttrValue::LIST_INT({conv_param_->dilation_h_, conv_param_->dilation_w_}));

  if (conv_param_->pad_mode_ == Pad_Same) {
    conv_dw_->set_attr_pad_mode(ge::AttrValue::STR{"SAME"});
    conv_dw_->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
  } else if (conv_param_->pad_mode_ == Pad_Valid) {
    conv_dw_->set_attr_pad_mode(ge::AttrValue::STR{"VALID"});
    conv_dw_->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
  } else {
    conv_dw_->set_attr_pad_mode(ge::AttrValue::STR{"SPECIFIC"});
    conv_dw_->set_attr_pads(
      ge::AttrValue::LIST_INT({conv_param_->pad_u_, conv_param_->pad_d_, conv_param_->pad_l_, conv_param_->pad_r_}));
  }
  return RET_OK;
}

int ConvolutionDepthwiseNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                                const std::vector<lite::Tensor *> &outputs,
                                                const std::vector<ge::Operator *> &npu_inputs) {
  auto ret = SetPreTranspose(npu_inputs[0]);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "New pre transpose npu operator (NHWC -> NCHW) for op " << name_ << " failed.";
    return RET_ERROR;
  }

  // set conv attr param
  conv_dw_ = new (std::nothrow) hiai::op::ConvolutionDepthwise(name_ + "_conv_depthwise");
  if (conv_dw_ == nullptr) {
    MS_LOG(ERROR) << "New convolution depthwise operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  ret = SetConvDwParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set npu op parameter for convolution depthwise op " << name_ << " failed.";
    return RET_ERROR;
  }

  ret = InitWeightBiasConst(inputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set weight and bias for convolution depthwise op " << name_ << " failed when running npu";
    return RET_ERROR;
  }
  conv_dw_->set_input_filter(*weight_);
  if (inputs.size() == 3) {
    conv_dw_->set_input_bias(*bias_);
  }
  conv_dw_->set_input_x(*pre_trans_);

  if (conv_param_->act_type_ != ActType_No) {
    ret = SetActivation(conv_dw_, conv_param_->act_type_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }

  if (conv_param_->act_type_ == ActType_No) {
    ret = SetPostTranspose(conv_dw_);
  } else {
    ret = SetPostTranspose(act_);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "New post transpose npu operator (NCHW -> NHWC) for op " << name_ << " failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

ge::Operator *mindspore::kernel::ConvolutionDepthwiseNPUKernel::GetNPUOp() { return post_trans_; }

ConvolutionDepthwiseNPUKernel::~ConvolutionDepthwiseNPUKernel() {
  if (conv_dw_ != nullptr) {
    delete conv_dw_;
    conv_dw_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_DepthwiseConv2D, NPUKernelCreator<ConvolutionDepthwiseNPUKernel>)
}  // namespace mindspore::kernel
