/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/coreml/op/convolution_coreml.h"
#include <cmath>
#include "src/litert/delegate/delegate_utils.h"
namespace mindspore::lite {
int ConvolutionCoreMLOp::IsSupport() {
  if (!in_tensors_[kWeightIndex].IsConst()) {
    MS_LOG(WARNING) << "CoreML convolution does not support dynamic weight.";
    return RET_NOT_SUPPORT;
  }
  conv_prim_ = op_primitive_->value_as_Conv2DFusion();
  if (conv_prim_ == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(conv_prim_->stride());
  stride_h_ = static_cast<int>(*(conv_prim_->stride()->begin()));
  stride_w_ = static_cast<int>(*(conv_prim_->stride()->begin() + 1));
  CHECK_NULL_RETURN(conv_prim_->dilation());
  dilation_h_ = static_cast<int>(*(conv_prim_->dilation()->begin()));
  dilation_w_ = static_cast<int>(*(conv_prim_->dilation()->begin() + 1));
  // org conv format: NHWC
  if (stride_h_ > in_tensors_[0].Shape()[kNHWC_H] || stride_w_ > in_tensors_[0].Shape()[kNHWC_W]) {
    MS_LOG(WARNING) << "CoreML convolution does not support stride greater than input size.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int ConvolutionCoreMLOp::SetConvParam() {
  auto group = static_cast<int>(conv_prim_->group());
  conv_param_->set_ngroups(group);
  conv_param_->add_stride(stride_h_);
  conv_param_->add_stride(stride_w_);
  conv_param_->add_dilationfactor(dilation_h_);
  conv_param_->add_dilationfactor(dilation_w_);
  if (conv_prim_->pad_mode() == schema::PadMode_SAME) {
    conv_param_->mutable_same();
  } else {
    conv_param_->mutable_valid();
    if (conv_prim_->pad_list() != nullptr) {
      auto pad_u = static_cast<int>(*(conv_prim_->pad_list()->begin() + PAD_UP));
      auto pad_d = static_cast<int>(*(conv_prim_->pad_list()->begin() + PAD_DOWN));
      auto pad_l = static_cast<int>(*(conv_prim_->pad_list()->begin() + PAD_LEFT));
      auto pad_r = static_cast<int>(*(conv_prim_->pad_list()->begin() + PAD_RIGHT));
      auto ret = SetPadding({pad_u, pad_d, pad_l, pad_r});
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Fail to set padding for op: " << name_;
        return RET_ERROR;
      }
    }
  }
  act_type_ = conv_prim_->activation_type();
  return RET_OK;
}
}  // namespace mindspore::lite
