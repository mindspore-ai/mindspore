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

#include "src/runtime/kernel/npu/deconvolution_npu.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;

namespace mindspore::kernel {
int DeconvolutionNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs,
                                      const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter) {
  if (conv_param_->group_ != 1) {
    MS_LOG(WARNING) << "Only support group equals 1 for npu deconvolution op";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeconvolutionNPUKernel::SetConvParam() {
  deconv_->set_attr_strides(ge::AttrValue::LIST_INT({conv_param_->stride_h_, conv_param_->stride_w_}));
  deconv_->set_attr_dilations(ge::AttrValue::LIST_INT({conv_param_->dilation_h_, conv_param_->dilation_w_}));
  deconv_->set_attr_groups(conv_param_->group_);

  if (conv_param_->pad_mode_ == Pad_same) {
    deconv_->set_attr_pad_mode(ge::AttrValue::STR{"SAME"});
    deconv_->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
  } else if (conv_param_->pad_mode_ == Pad_valid) {
    deconv_->set_attr_pad_mode(ge::AttrValue::STR{"VALID"});
    deconv_->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
  } else {
    deconv_->set_attr_pad_mode(ge::AttrValue::STR{"SPECIFIC"});
    deconv_->set_attr_pads(
      ge::AttrValue::LIST_INT({conv_param_->pad_u_, conv_param_->pad_d_, conv_param_->pad_l_, conv_param_->pad_r_}));
  }
  return RET_OK;
}

int DeconvolutionNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                         const std::vector<lite::Tensor *> &outputs,
                                         const std::vector<ge::Operator *> &npu_inputs) {
  // set conv attr param
  deconv_ = new (std::nothrow) hiai::op::ConvTranspose(name_ + "_deconv");
  if (deconv_ == nullptr) {
    MS_LOG(ERROR) << "New deconvolution operator for deconvolution op " << name_ << " failed.";
    return RET_ERROR;
  }
  auto ret = SetConvParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set npu op parameter for deconvolution op " << name_ << " failed.";
    return RET_ERROR;
  }

  ret = InitWeightConst(inputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set weight and bias for deconvolution op " << name_ << " failed when running npu";
    return RET_ERROR;
  }
  deconv_->set_input_filter(*weight_);
  if (inputs.size() == 3) {
    ret = InitBiasConst(inputs);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set bias for deconvolution op " << name_ << " failed when running npu";
      return RET_ERROR;
    }
    deconv_->set_input_bias(*bias_);
  }
  deconv_->set_input_x(*npu_inputs[0]);

  if (conv_param_->act_type_ != ActType_No) {
    ret = SetActivation(deconv_, conv_param_->act_type_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

ge::Operator *mindspore::kernel::DeconvolutionNPUKernel::GetNPUOp() {
  if (conv_param_->act_type_ == ActType_No) {
    return deconv_;
  } else {
    return act_;
  }
}

DeconvolutionNPUKernel::~DeconvolutionNPUKernel() {
  if (deconv_ != nullptr) {
    delete deconv_;
    deconv_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Conv2dTransposeFusion, NPUKernelCreator<DeconvolutionNPUKernel>)
}  // namespace mindspore::kernel
