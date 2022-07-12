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

#include "src/litert/delegate/nnapi/op/conv_transpose_nnapi.h"
#include <algorithm>
#include <vector>
#include <unordered_map>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "src/common/utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
int NNAPIConvTranspose::InitParams() {
  auto conv_transpose = op_primitive_->value_as_Conv2dTransposeFusion();
  MS_ASSERT(conv_transpose != nullptr);
  group_ = static_cast<int>(conv_transpose->group());
  in_channel_ = static_cast<int>(conv_transpose->in_channel());
  out_channel_ = static_cast<int>(conv_transpose->out_channel());

  pad_mode_ = static_cast<int>(conv_transpose->pad_mode());
  if (conv_transpose->pad_list() != nullptr && conv_transpose->pad_list()->size() == DIMENSION_4D) {
    pad_list_.push_back(static_cast<int>(*(conv_transpose->pad_list()->begin() + PAD_LEFT)));
    pad_list_.push_back(static_cast<int>(*(conv_transpose->pad_list()->begin() + PAD_RIGHT)));
    pad_list_.push_back(static_cast<int>(*(conv_transpose->pad_list()->begin() + PAD_UP)));
    pad_list_.push_back(static_cast<int>(*(conv_transpose->pad_list()->begin() + PAD_DOWN)));
  }
  MS_CHECK_TRUE_RET(conv_transpose->stride() != nullptr && conv_transpose->stride()->size() == DIMENSION_2D, RET_ERROR);
  strides_.push_back(static_cast<int>(*(conv_transpose->stride()->begin() + 1)));
  strides_.push_back(static_cast<int>(*(conv_transpose->stride()->begin())));

  MS_CHECK_TRUE_RET(conv_transpose->dilation() != nullptr && conv_transpose->dilation()->size() == DIMENSION_2D,
                    RET_ERROR);
  dilations_.push_back(static_cast<int>(*(conv_transpose->dilation()->begin() + 1)));
  dilations_.push_back(static_cast<int>(*(conv_transpose->dilation()->begin())));
  act_type_ = conv_transpose->activation_type();
  return RET_OK;
}

int NNAPIConvTranspose::AddOpToNNAPIModel(ANeuralNetworksModel *nnapi_model,
                                          std::vector<mindspore::MSTensor> *all_tensors) {
  MS_ASSERT(nnapi_model != nullptr && all_tensors != nullptr);
  OperationCode node_type = ANEURALNETWORKS_TRANSPOSE_CONV_2D;
  if (TransConvWeightFromKHWCToCHWK(nnapi_model, all_tensors) != RET_OK) {
    MS_LOG(ERROR) << "Transpose weight of deconv failed.";
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

  if (pad_mode_ == PadMode::Pad_pad) {
    MS_CHECK_TRUE_RET(pad_list_.size() == DIMENSION_4D, RET_ERROR);
    for (auto pad : pad_list_) {
      if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "pad", DataType::kNumberTypeInt32, pad) != RET_OK) {
        MS_LOG(ERROR) << "Add paddings for conv transpose to NNAPI model failed.";
        return RET_ERROR;
      }
    }
  } else {
    auto output_shape = out_tensors_.front().Shape();
    if (std::find(output_shape.begin(), output_shape.end(), -1) != output_shape.end()) {
      MS_LOG(ERROR) << "The output shape of convolution transpose is invalid.";
      return RET_ERROR;
    }
    std::vector<int> shape;
    (void)std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(shape),
                         [](int64_t x) { return static_cast<int>(x); });
    MSTensorInfo tensor_info{op_name_ + "_shape",
                             DataType::kNumberTypeInt32,
                             {static_cast<int64_t>(shape.size())},
                             shape.data(),
                             shape.size() * sizeof(int)};
    if (AddTensorToNNAPIModel(nnapi_model, all_tensors, tensor_info) != RET_OK) {
      MS_LOG(ERROR) << "Add output shape tensor for convolution transpose failed.";
      return RET_ERROR;
    }
    // Use the implicit pad mode for NNAPI model.
    auto pad_mode = pad_mode_ - 1;  // the enum pad scheme of NNAPI is PAD_SAME and PAD_VALID.
    if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "pad_mode", DataType::kNumberTypeInt32, pad_mode) !=
        RET_OK) {
      MS_LOG(ERROR) << "Add pad mode for conv transpose to NNAPI model failed.";
      return RET_ERROR;
    }
  }

  // set strides to NNAPI model.
  for (auto stride : strides_) {
    if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "stride", DataType::kNumberTypeInt32, stride) != RET_OK) {
      MS_LOG(ERROR) << "Add strides for conv transpose to NNAPI model failed.";
      return RET_ERROR;
    }
  }

  // convert act_type to an input of nnapi node.
  if (AddScalarToNNAPIModel<int>(nnapi_model, all_tensors, "act_type", DataType::kNumberTypeInt32, act_type_) !=
      RET_OK) {
    MS_LOG(ERROR) << "Add activation type for conv transpose to NNAPI model failed.";
    return RET_ERROR;
  }
  // set nchw to an input of nnapi node.
  if (AddScalarToNNAPIModel<bool>(nnapi_model, all_tensors, "nchw", DataType::kNumberTypeBool, false) != RET_OK) {
    MS_LOG(ERROR) << "set nchw format for conv transpose to NNAPI model failed.";
    return RET_ERROR;
  }

  if (nnapi_->ANeuralNetworksModel_addOperation(nnapi_model, node_type, input_indices_.size(), input_indices_.data(),
                                                output_indices_.size(),
                                                output_indices_.data()) != ANEURALNETWORKS_NO_ERROR) {
    MS_LOG(ERROR) << "Add operation to NNAPI model failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
