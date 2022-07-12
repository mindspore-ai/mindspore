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

#include "src/litert/delegate/nnapi/op/conv_nnapi.h"
#include <algorithm>
#include <vector>
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "src/common/utils.h"
#include "nnacl/op_base.h"
#include "nnacl/fp32/transpose_fp32.h"

namespace mindspore {
namespace lite {
bool NNAPIConv::IsSupport() { return true; }

int NNAPIConv::InitParams() {
  auto conv = op_primitive_->value_as_Conv2DFusion();
  MS_ASSERT(conv != nullptr);
  group_ = static_cast<int>(conv->group());
  in_channel_ = static_cast<int>(conv->in_channel());
  out_channel_ = static_cast<int>(conv->out_channel());
  is_dw_conv_ = group_ != 1 && group_ == in_channel_ && group_ == out_channel_;
  is_group_conv_ = group_ != 1 && !is_dw_conv_;

  pad_mode_ = static_cast<int>(conv->pad_mode());
  if (conv->pad_list() != nullptr && conv->pad_list()->size() == DIMENSION_4D) {
    pad_list_.push_back(static_cast<int>(*(conv->pad_list()->begin() + PAD_LEFT)));
    pad_list_.push_back(static_cast<int>(*(conv->pad_list()->begin() + PAD_RIGHT)));
    pad_list_.push_back(static_cast<int>(*(conv->pad_list()->begin() + PAD_UP)));
    pad_list_.push_back(static_cast<int>(*(conv->pad_list()->begin() + PAD_DOWN)));
  }
  MS_CHECK_TRUE_RET(conv->stride() != nullptr && conv->stride()->size() == DIMENSION_2D, RET_ERROR);
  strides_.push_back(static_cast<int>(*(conv->stride()->begin() + 1)));
  strides_.push_back(static_cast<int>(*(conv->stride()->begin())));

  MS_CHECK_TRUE_RET(conv->dilation() != nullptr && conv->dilation()->size() == DIMENSION_2D, RET_ERROR);
  dilations_.push_back(static_cast<int>(*(conv->dilation()->begin() + 1)));
  dilations_.push_back(static_cast<int>(*(conv->dilation()->begin())));
  act_type_ = conv->activation_type();
  return RET_OK;
}

int NNAPIConv::AddAttributesForConv(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) {
  for (auto pad : pad_list_) {
    if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "pad", DataType::kNumberTypeInt32, pad) != RET_OK) {
      MS_LOG(ERROR) << "Add paddings for conv to NNAPI model failed.";
      return RET_ERROR;
    }
  }
  if (pad_list_.empty()) {
    // Use the implicit pad mode for NNAPI model.
    MS_CHECK_TRUE_RET(pad_mode_ != PadMode::Pad_pad, RET_ERROR);
    auto pad_mode = pad_mode_ - 1;  // the enum pad scheme of NNAPI is PAD_SAME and PAD_VALID.
    if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "pad_mode", DataType::kNumberTypeInt32, pad_mode) !=
        RET_OK) {
      MS_LOG(ERROR) << "Add pad mode for conv to NNAPI model failed.";
      return RET_ERROR;
    }
  }

  // set strides to NNAPI model.
  for (auto stride : strides_) {
    if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "stride", DataType::kNumberTypeInt32, stride) != RET_OK) {
      MS_LOG(ERROR) << "Add strides for conv to NNAPI model failed.";
      return RET_ERROR;
    }
  }
  // set group for grouped conv or multiplier for depthwise conv.
  if (group_ != 1) {
    int value = is_dw_conv_ ? 1 : group_;
    if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "group", DataType::kNumberTypeInt32, value) != RET_OK) {
      MS_LOG(ERROR) << "Add activation type for conv to NNAPI model failed.";
      return RET_ERROR;
    }
  }
  // convert act_type to an input of nnapi node.
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "act_type", DataType::kNumberTypeInt32, act_type_) !=
      RET_OK) {
    MS_LOG(ERROR) << "Add activation type for conv to NNAPI model failed.";
    return RET_ERROR;
  }
  // set nchw to an input of nnapi node.
  if (AddScalarToNNAPIModel<bool>(nnapi_model, all_tensors, "nchw", DataType::kNumberTypeBool, false) != RET_OK) {
    MS_LOG(ERROR) << "set nchw format for conv to NNAPI model failed.";
    return RET_ERROR;
  }
  // grouped conv has no dilations.
  if (!is_group_conv_) {
    for (auto dilation : dilations_) {
      if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "dilation", DataType::kNumberTypeInt32, dilation) !=
          RET_OK) {
        MS_LOG(ERROR) << "Add activation type for conv to NNAPI model failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int NNAPIConv::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model, std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  if (is_dw_conv_ && TransConvWeightFromKHWCToCHWK(nnapi_model, all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "Adjust weight for depthwise conv failed.";
    return RET_ERROR;
  }
  if (InitNNAPIOpInOut(*all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "InitNNAPINodeInfo failed.";
    return RET_ERROR;
  }
  if (in_tensors_.size() == kInputSize1) {
    // has no bias, new a bias tensor with zero value.
    auto weight_type = in_tensors_.at(1).DataType();
    auto bias_type = (weight_type != DataType::kNumberTypeInt8 && weight_type != DataType::kNumberTypeUInt8)
                       ? weight_type
                       : DataType::kNumberTypeInt32;
    MSTensorInfo bias_info{op_name_ + "_bias", bias_type, {out_channel_}, nullptr, 0};
    if (AddTensorToNNAPIModel(nnapi_model, all_tensors, bias_info) != RET_OK) {
      MS_LOG(ERROR) << "NNAPI does not support convolution without bias, and create zero-value tenosr failed.";
      return RET_ERROR;
    }
  }

  if (AddAttributesForConv(nnapi_model, all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "Add attributes for convolution failed.";
    return RET_ERROR;
  }

  OperationCode node_type = is_dw_conv_ ? ANEURALNETWORKS_DEPTHWISE_CONV_2D
                                        : (is_group_conv_ ? ANEURALNETWORKS_GROUPED_CONV_2D : ANEURALNETWORKS_CONV_2D);
  if (nnapi_->ANeuralNetworksModel_addOperation(nnapi_model, node_type, input_indices_.size(), input_indices_.data(),
                                                output_indices_.size(),
                                                output_indices_.data()) != ANEURALNETWORKS_NO_ERROR) {
    MS_LOG(ERROR) << "Add operation to NNAPI model failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int NNAPIConv::TransConvWeightFromKHWCToCHWK(ANeuralNetworksModel *nnapi_model,
                                             std::vector<mindspore::MSTensor> *all_tensors) {
  auto weight = inputs().at(1);
  if (!weight.IsConst()) {
    MS_LOG(ERROR) << "The weight of conv must be constant.";
    return RET_ERROR;
  }
  auto shape = weight.Shape();
  auto new_weight = weight.Clone();
  MS_CHECK_TRUE_RET(new_weight != nullptr, RET_ERROR);
  new_weight->SetQuantParams(weight.QuantParams());
  // transpose weight from KHWC to CHWK.
  std::vector<int64_t> new_shape = {shape.at(kNHWC_C), shape.at(kNHWC_H), shape.at(kNHWC_W), shape.at(kNHWC_N)};
  new_weight->SetShape(new_shape);

  TransposeParameter param;
  param.strides_[DIMENSION_4D - 1] = 1;
  param.out_strides_[DIMENSION_4D - 1] = 1;
  for (int i = DIMENSION_4D - 2; i >= 0; i--) {
    param.strides_[i] = shape[i + 1] * param.strides_[i + 1];
    param.out_strides_[i] = new_shape[i + 1] * param.out_strides_[i + 1];
  }
  param.num_axes_ = DIMENSION_4D;
  param.data_num_ = weight.ElementNum();
  std::vector<int> perm = {3, 1, 2, 0};
  memcpy(param.perm_, perm.data(), sizeof(int) * DIMENSION_4D);
  std::vector<int> out_shape;
  (void)std::transform(new_shape.begin(), new_shape.end(), std::back_inserter(out_shape),
                       [](int64_t x) { return static_cast<int>(x); });
  int ret = 0;
  if (weight.DataType() == DataType::kNumberTypeFloat32) {
    ret = DoTransposeFp32(reinterpret_cast<float *>(weight.MutableData()),
                          reinterpret_cast<float *>(new_weight->MutableData()),
                          reinterpret_cast<const int *>(out_shape.data()), &param);
  } else {
    MS_LOG(ERROR) << "Unsupported to pack depthwise conv weight";
    delete new_weight;
    return RET_ERROR;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Transpose conv weight from KHWC to CHWK failed.";
    delete new_weight;
    return RET_ERROR;
  }
  if (AddNNAPIOperand(nnapi_model, *new_weight, static_cast<int>(all_tensors->size()), kNHWC_C) != RET_OK) {
    MS_LOG(ERROR) << "Add depthwise conv weight to NNAPI model failed: " << op_name_;
    delete new_weight;
    return RET_ERROR;
  }
  in_tensors_.at(1) = *new_weight;
  all_tensors->push_back(*new_weight);
  op_attribute_tensors_.push_back(new_weight);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
