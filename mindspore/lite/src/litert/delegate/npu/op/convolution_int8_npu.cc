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

#include "src/litert/delegate/npu/op/convolution_int8_npu.h"
#include "src/litert/delegate/npu/npu_converter_utils.h"
namespace mindspore::lite {
int ConvolutionInt8NPUOp::IsSupport(const schema::Primitive *primitive,
                                    const std::vector<mindspore::MSTensor> &in_tensors,
                                    const std::vector<mindspore::MSTensor> &out_tensors) {
  CHECK_LESS_RETURN(in_tensors.size(), kInputSize1);
  if (!in_tensors[1].IsConst()) {
    MS_LOG(WARNING) << "NPU convolution does not support dynamic weight.";
    return RET_NOT_SUPPORT;
  }
  auto conv_prim = primitive->value_as_Conv2DFusion();
  if (conv_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  auto stride_h = static_cast<int>(*(conv_prim->stride()->begin()));
  auto stride_w = static_cast<int>(*(conv_prim->stride()->begin() + 1));
  auto in_shape = in_tensors[0].Shape();  // default format: nhwc, RunPass not called
  if (stride_h > in_shape[NHWC_H] || stride_w > in_shape[NHWC_W]) {
    MS_LOG(WARNING) << "Npu convolution does not support stride greater than input size.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int ConvolutionInt8NPUOp::SetConvParam(const schema::Conv2DFusion *conv_prim) {
  CHECK_NULL_RETURN(conv_prim);
  auto group = static_cast<int>(conv_prim->group());
  CHECK_NULL_RETURN(conv_prim->stride());
  CHECK_LESS_RETURN(conv_prim->stride()->size(), DIMENSION_2D);
  auto stride_h = static_cast<int>(*(conv_prim->stride()->begin()));
  auto stride_w = static_cast<int>(*(conv_prim->stride()->begin() + 1));
  CHECK_NULL_RETURN(conv_prim->dilation());
  CHECK_LESS_RETURN(conv_prim->dilation()->size(), DIMENSION_2D);
  auto dilation_h = static_cast<int>(*(conv_prim->dilation()->begin()));
  auto dilation_w = static_cast<int>(*(conv_prim->dilation()->begin() + 1));
  conv_->set_attr_strides(ge::AttrValue::LIST_INT({stride_h, stride_w}));
  conv_->set_attr_dilations(ge::AttrValue::LIST_INT({dilation_h, dilation_w}));
  conv_->set_attr_groups(group);

  if (conv_prim->pad_mode() == schema::PadMode_SAME) {
    conv_->set_attr_pad_mode(ge::AttrValue::STR{"SAME"});
    conv_->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
  } else if (conv_prim->pad_mode() == schema::PadMode_VALID) {
    conv_->set_attr_pad_mode(ge::AttrValue::STR{"VALID"});
    conv_->set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}));
  } else {
    conv_->set_attr_pad_mode(ge::AttrValue::STR{"SPECIFIC"});
    CHECK_NULL_RETURN(conv_prim->pad_list());
    CHECK_LESS_RETURN(conv_prim->pad_list()->size(), DIMENSION_4D);
    auto pad_u = static_cast<int>(*(conv_prim->pad_list()->begin() + PAD_UP));
    auto pad_d = static_cast<int>(*(conv_prim->pad_list()->begin() + PAD_DOWN));
    auto pad_l = static_cast<int>(*(conv_prim->pad_list()->begin() + PAD_LEFT));
    auto pad_r = static_cast<int>(*(conv_prim->pad_list()->begin() + PAD_RIGHT));
    conv_->set_attr_pads(ge::AttrValue::LIST_INT({pad_u, pad_d, pad_l, pad_r}));
  }
  return RET_OK;
}

int ConvolutionInt8NPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors) {
  // set conv attr param
  conv_ = new (std::nothrow) hiai::op::QuantizedConvolution(name_ + "_quant_conv");
  if (conv_ == nullptr) {
    MS_LOG(ERROR) << "New convolution operator for convolution op " << name_ << " failed.";
    return RET_ERROR;
  }
  auto conv_prim = primitive->value_as_Conv2DFusion();
  if (conv_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  auto ret = SetConvParam(conv_prim);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set npu op parameter for convolution op " << name_ << " failed.";
    return RET_ERROR;
  }

  act_type_ = conv_prim->activation_type();
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    ret = SetActivation(conv_, act_type_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }
  SetQuantParam(conv_, in_tensors);
  return RET_OK;
}

int ConvolutionInt8NPUOp::SetNPUInputs(
  const std::vector<mindspore::MSTensor> &in_tensors, const std::vector<mindspore::MSTensor> &out_tensors,
  const std::vector<ge::Operator *> &npu_inputs,
  const std::unordered_map<int, std::pair<ge::Operator *, int>> &index2_multi_out_index) {
  auto ret = InitWeightConst(in_tensors);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set weight and bias for convolution op " << name_ << " failed when running npu";
    return RET_ERROR;
  }
  conv_->set_input_filter(*weight_);
  if (in_tensors.size() == CONV_INPUT_SIZE) {
    ret = InitBiasConst(in_tensors);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set bias for convolution op " << name_ << " failed when running npu";
      return RET_ERROR;
    }
    conv_->set_input_bias(*bias_);
  }

  if (!index2_multi_out_index.empty()) {
    auto itr = index2_multi_out_index.begin();
    auto in_op = itr->second.first;
    MS_CHECK_TRUE_RET(in_op != nullptr, RET_ERROR);
    conv_->SetInput(itr->first, *in_op, itr->second.second);
  } else {
    CHECK_LESS_RETURN(npu_inputs.size(), 1);
    conv_->set_input_x(*npu_inputs[0]);
  }
  return RET_OK;
}

ge::Operator *ConvolutionInt8NPUOp::GetNPUOp() {
  if (act_type_ == schema::ActivationType_NO_ACTIVATION) {
    return conv_;
  } else {
    return act_;
  }
}

ConvolutionInt8NPUOp::~ConvolutionInt8NPUOp() {
  if (conv_ != nullptr) {
    delete conv_;
    conv_ = nullptr;
  }
}
}  // namespace mindspore::lite
