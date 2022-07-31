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

#include "src/litert/delegate/coreml/op/deconvolution_coreml.h"
#include "src/litert/delegate/delegate_utils.h"
namespace mindspore::lite {
int DeconvolutionCoreMLOp::IsSupport() {
  if (!in_tensors_[kWeightIndex].IsConst()) {
    MS_LOG(WARNING) << "CoreML deconvolution does not support dynamic weight.";
    return RET_NOT_SUPPORT;
  }
  deconv_prim_ = op_primitive_->value_as_Conv2dTransposeFusion();
  if (deconv_prim_ == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  if (static_cast<int>(deconv_prim_->group()) != 1) {
    MS_LOG(WARNING) << "Only support group equals 1 for npu deconvolution op";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int DeconvolutionCoreMLOp::SetConvParam() {
  conv_param_->set_isdeconvolution(true);
  CHECK_NULL_RETURN(deconv_prim_->stride());
  auto stride_h = static_cast<int>(*(deconv_prim_->stride()->begin()));
  auto stride_w = static_cast<int>(*(deconv_prim_->stride()->begin() + 1));
  conv_param_->add_stride(stride_h);
  conv_param_->add_stride(stride_w);
  CHECK_NULL_RETURN(deconv_prim_->dilation());
  auto dilation_h = static_cast<int>(*(deconv_prim_->dilation()->begin()));
  auto dilation_w = static_cast<int>(*(deconv_prim_->dilation()->begin() + 1));
  conv_param_->add_dilationfactor(dilation_h);
  conv_param_->add_dilationfactor(dilation_w);
  conv_param_->add_outputshape(output_h_);
  conv_param_->add_outputshape(output_w_);
  if (deconv_prim_->pad_mode() == schema::PadMode_SAME) {
    conv_param_->mutable_same();
  } else {
    conv_param_->mutable_valid();
    if (deconv_prim_->pad_list() != nullptr) {
      auto pad_u = static_cast<int>(*(deconv_prim_->pad_list()->begin() + PAD_UP));
      auto pad_d = static_cast<int>(*(deconv_prim_->pad_list()->begin() + PAD_DOWN));
      auto pad_l = static_cast<int>(*(deconv_prim_->pad_list()->begin() + PAD_LEFT));
      auto pad_r = static_cast<int>(*(deconv_prim_->pad_list()->begin() + PAD_RIGHT));
      auto ret = SetPadding({pad_u, pad_d, pad_l, pad_r});
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Fail to set padding for op: " << name_;
        return RET_ERROR;
      }
    }
  }
  act_type_ = deconv_prim_->activation_type();
  return RET_OK;
}
}  // namespace mindspore::lite
