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

#include "src/delegate/npu/op/convolution_depthwise_npu.h"
#include "src/delegate/npu/npu_converter_utils.h"
namespace mindspore {
int ConvolutionDepthwiseNPUOp::SetConvDwParam(const schema::Conv2DFusion *conv_prim) {
  auto stride_h = static_cast<int>(*(conv_prim->stride()->begin()));
  auto stride_w = static_cast<int>(*(conv_prim->stride()->begin() + 1));
  auto dilation_h = static_cast<int>(*(conv_prim->dilation()->begin()));
  auto dilation_w = static_cast<int>(*(conv_prim->dilation()->begin() + 1));
  conv_dw_->set_attr_strides(ge::AttrValue::LIST_INT({stride_h, stride_w}));
  conv_dw_->set_attr_dilations(ge::AttrValue::LIST_INT({dilation_h, dilation_w}));

  if (conv_prim->pad_mode() == schema::PadMode_SAME) {
    conv_dw_->set_attr_pad_mode(ge::AttrValue::STR{"SAME"});
    conv_dw_->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
  } else if (conv_prim->pad_mode() == schema::PadMode_VALID) {
    conv_dw_->set_attr_pad_mode(ge::AttrValue::STR{"VALID"});
    conv_dw_->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
  } else {
    conv_dw_->set_attr_pad_mode(ge::AttrValue::STR{"VALID"});
    auto pad_u = static_cast<int>(*(conv_prim->pad_list()->begin() + PAD_UP));
    auto pad_d = static_cast<int>(*(conv_prim->pad_list()->begin() + PAD_DOWN));
    auto pad_l = static_cast<int>(*(conv_prim->pad_list()->begin() + PAD_LEFT));
    auto pad_r = static_cast<int>(*(conv_prim->pad_list()->begin() + PAD_RIGHT));
    conv_dw_->set_attr_pads(ge::AttrValue::LIST_INT({pad_u, pad_d, pad_l, pad_r}));
  }
  return RET_OK;
}

int ConvolutionDepthwiseNPUOp::Init(const schema::Primitive *primitive,
                                    const std::vector<mindspore::MSTensor> &in_tensors,
                                    const std::vector<mindspore::MSTensor> &out_tensors) {
  conv_dw_ = new (std::nothrow) hiai::op::ConvolutionDepthwise(name_ + "_conv_depthwise");
  if (conv_dw_ == nullptr) {
    MS_LOG(ERROR) << "New convolution depthwise operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  auto conv_prim = primitive->value_as_Conv2DFusion();
  if (conv_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  auto ret = SetConvDwParam(conv_prim);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set npu op parameter for convolution depthwise op " << name_ << " failed.";
    return RET_ERROR;
  }
  act_type_ = conv_prim->activation_type();
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    ret = SetActivation(conv_dw_, conv_prim->activation_type());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ConvolutionDepthwiseNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                            const std::vector<mindspore::MSTensor> &out_tensors,
                                            const std::vector<ge::Operator *> &npu_inputs) {
  auto ret = InitWeightConst(in_tensors);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set weight and bias for convolution depthwise op " << name_ << " failed when running npu";
    return RET_ERROR;
  }
  conv_dw_->set_input_filter(*weight_);

  if (in_tensors.size() == CONV_INPUT_SIZE) {
    ret = InitBiasConst(in_tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set bias for convolution depthwise op " << name_ << " failed when running npu";
      return RET_ERROR;
    }
    conv_dw_->set_input_bias(*bias_);
  }
  conv_dw_->set_input_x(*npu_inputs[0]);
  return RET_OK;
}

ge::Operator *ConvolutionDepthwiseNPUOp::GetNPUOp() {
  if (act_type_ == schema::ActivationType_NO_ACTIVATION) {
    return conv_dw_;
  } else {
    return act_;
  }
}

ConvolutionDepthwiseNPUOp::~ConvolutionDepthwiseNPUOp() {
  if (conv_dw_ != nullptr) {
    delete conv_dw_;
    conv_dw_ = nullptr;
  }
}
}  // namespace mindspore
