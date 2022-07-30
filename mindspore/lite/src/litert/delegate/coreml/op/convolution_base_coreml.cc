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

#include "src/litert/delegate/coreml/op/convolution_base_coreml.h"
#include "src/litert/delegate/delegate_utils.h"
namespace mindspore::lite {
int ConvolutionBaseCoreMLOp::SetConvWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto weight_shape = weight_tensor.Shape();
  conv_param_->set_kernelchannels(weight_shape.at(MS_WT_CIN));
  conv_param_->set_outputchannels(weight_shape.at(MS_WT_COUT));
  conv_param_->add_kernelsize(weight_shape.at(MS_WT_H));
  conv_param_->add_kernelsize(weight_shape.at(MS_WT_W));

  // transpose the weight, (c_out, h, w, c_in) -> (c_out, c_in, h, w)
  auto org_weight = weight_tensor.Data().get();
  MS_ASSERT(org_weight != nullptr);
  if (weight_tensor.DataType() == DataType::kNumberTypeFloat32) {
    auto *ml_weight_container = conv_param_->mutable_weights()->mutable_floatvalue();
    ml_weight_container->Resize(weight_tensor.ElementNum(), 0);
    auto *ml_weight = reinterpret_cast<void *>(ml_weight_container->mutable_data());
    PackNHWCToNCHWFp32(org_weight, ml_weight, weight_shape[MS_WT_COUT], weight_shape[MS_WT_H] * weight_shape[MS_WT_W],
                       weight_shape[MS_WT_CIN]);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of weight tensor for CoreML convolution.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionBaseCoreMLOp::SetConvBias() {
  if (in_tensors_.size() >= kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    auto org_bias = bias_tensor.Data().get();
    conv_param_->set_hasbias(true);
    if (bias_tensor.DataType() == DataType::kNumberTypeFloat32) {
      auto *ml_bias_container = conv_param_->mutable_bias()->mutable_floatvalue();
      ml_bias_container->Resize(bias_tensor.ElementNum(), 0);
      auto *ml_bias = reinterpret_cast<void *>(ml_bias_container->mutable_data());
      memcpy(ml_bias, org_bias, bias_tensor.DataSize());
    } else {
      MS_LOG(ERROR) << "Unsupported data type of bias tensor for CoreML convolution.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ConvolutionBaseCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr);
  conv_param_ = op_->mutable_convolution();
  auto ret = SetConvParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set conv param failed for op: " << name_;
    return RET_ERROR;
  }
  ret = SetConvWeight();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set conv weight failed for op: " << name_;
    return RET_ERROR;
  }
  ret = SetConvBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set conv bias failed for op: " << name_;
    return RET_ERROR;
  }
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    ret = SetActivation(act_type_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set conv activation failed for op: " << name_;
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
