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

#include "src/delegate/npu/op/deconvolution_npu.h"
#include "src/delegate/npu/npu_converter_utils.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"

namespace mindspore {
int DeconvolutionNPUOp::IsSupport(const schema::Primitive *primitive,
                                  const std::vector<mindspore::MSTensor> &in_tensors,
                                  const std::vector<mindspore::MSTensor> &out_tensors) {
  CHECK_NULL_RETURN(primitive);
  auto deconv_prim = primitive->value_as_Conv2dTransposeFusion();
  if (deconv_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  if (static_cast<int>(deconv_prim->group()) != 1) {
    MS_LOG(WARNING) << "Only support group equals 1 for npu deconvolution op";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int DeconvolutionNPUOp::SetDeconvParam(const schema::Conv2dTransposeFusion *conv_prim) {
  CHECK_NULL_RETURN(conv_prim);
  CHECK_NULL_RETURN(deconv_);

  auto group = static_cast<int>(conv_prim->group());
  auto stride_h = static_cast<int>(*(conv_prim->stride()->begin()));
  auto stride_w = static_cast<int>(*(conv_prim->stride()->begin() + 1));
  auto dilation_h = static_cast<int>(*(conv_prim->dilation()->begin()));
  auto dilation_w = static_cast<int>(*(conv_prim->dilation()->begin() + 1));
  deconv_->set_attr_strides(ge::AttrValue::LIST_INT({stride_h, stride_w}));
  deconv_->set_attr_dilations(ge::AttrValue::LIST_INT({dilation_h, dilation_w}));
  deconv_->set_attr_groups(group);

  if (conv_prim->pad_mode() == schema::PadMode_SAME) {
    deconv_->set_attr_pad_mode(ge::AttrValue::STR{"SAME"});
    deconv_->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
  } else if (conv_prim->pad_mode() == schema::PadMode_VALID) {
    deconv_->set_attr_pad_mode(ge::AttrValue::STR{"VALID"});
    deconv_->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
  } else {
    deconv_->set_attr_pad_mode(ge::AttrValue::STR{"SPECIFIC"});
    auto pad_u = static_cast<int>(*(conv_prim->pad_list()->begin() + PAD_UP));
    auto pad_d = static_cast<int>(*(conv_prim->pad_list()->begin() + PAD_DOWN));
    auto pad_l = static_cast<int>(*(conv_prim->pad_list()->begin() + PAD_LEFT));
    auto pad_r = static_cast<int>(*(conv_prim->pad_list()->begin() + PAD_RIGHT));
    deconv_->set_attr_pads(ge::AttrValue::LIST_INT({pad_u, pad_d, pad_l, pad_r}));
  }
  return RET_OK;
}

int DeconvolutionNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors) {
  CHECK_NULL_RETURN(primitive);

  // set deconv attr param
  deconv_ = new (std::nothrow) hiai::op::ConvTranspose(name_ + "_deconv");
  if (deconv_ == nullptr) {
    MS_LOG(ERROR) << "New deconvolution operator for deconvolution op " << name_ << " failed.";
    return RET_ERROR;
  }

  auto deconv_prim = primitive->value_as_Conv2dTransposeFusion();
  if (deconv_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  auto ret = SetDeconvParam(deconv_prim);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set npu op parameter for convolution op " << name_ << " failed.";
    return RET_ERROR;
  }
  act_type_ = deconv_prim->activation_type();
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    ret = SetActivation(deconv_, act_type_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int DeconvolutionNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                     const std::vector<mindspore::MSTensor> &out_tensors,
                                     const std::vector<ge::Operator *> &npu_inputs) {
  CHECK_NULL_RETURN(deconv_);

  auto ret = InitWeightConst(in_tensors);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set weight and bias for deconvolution op " << name_ << " failed when running npu";
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(weight_);
  deconv_->set_input_filter(*weight_);
  if (in_tensors.size() == CONV_INPUT_SIZE) {
    ret = InitBiasConst(in_tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set bias for deconvolution op " << name_ << " failed when running npu";
      return RET_ERROR;
    }
    CHECK_NULL_RETURN(bias_);
    deconv_->set_input_bias(*bias_);
  }
  deconv_->set_input_x(*npu_inputs[0]);
  return RET_OK;
}

ge::Operator *DeconvolutionNPUOp::GetNPUOp() {
  if (act_type_ == schema::ActivationType_NO_ACTIVATION) {
    return deconv_;
  } else {
    return act_;
  }
}

DeconvolutionNPUOp::~DeconvolutionNPUOp() {
  if (deconv_ != nullptr) {
    delete deconv_;
    deconv_ = nullptr;
  }
}
}  // namespace mindspore
