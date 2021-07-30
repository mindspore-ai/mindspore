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

#include <numeric>
#include <functional>
#include "src/delegate/tensorrt/op/scale_tensorrt.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
constexpr int SCALE_INDEX = 1;
constexpr int SHIFT_INDEX = 2;
constexpr int POWER_INDEX = 3;

int ScaleTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 2 && in_tensors.size() != 3 && in_tensors.size() != 4) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is: " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  auto scale_op = op_primitive_->value_as_ScaleFusion();
  if (scale_op == nullptr) {
    MS_LOG(ERROR) << "convert failed";
    return RET_ERROR;
  }

  schema::ActivationType activation_type = scale_op->activation_type();
  nvinfer1::ITensor *scale_in_tensor = tensorrt_in_tensors_[0];
  // unsqueeze input Itensor to 4 dims
  if (in_tensors_[0].Shape().size() < 4) {
    scale_in_tensor = AddUnsqueezeOp(network);
    if (scale_in_tensor == nullptr) {
      MS_LOG(ERROR) << "AddUnsqueezeOp failed";
      return RET_ERROR;
    }
  }
  // mode of scale
  size_t axis = scale_op->axis();
  nvinfer1::ScaleMode mode;
  auto input_data_shape = in_tensors_[0].Shape();
  auto input_weight_shape = in_tensors_[1].Shape();
  int total = std::accumulate(input_data_shape.begin(), input_data_shape.end(), 1, std::multiplies<int>());
  MS_LOG(INFO) << "input tensor element cnt: " << total;
  if (input_weight_shape.size() == 0 || (input_weight_shape.size() == 1 && input_weight_shape[0] == 1)) {
    mode = nvinfer1::ScaleMode::kUNIFORM;
  } else if (axis < input_data_shape.size() && input_weight_shape.size() == 1 &&
             input_data_shape[axis] == input_weight_shape[0]) {
    mode = nvinfer1::ScaleMode::kCHANNEL;
  } else if (input_weight_shape.size() == 1 && input_weight_shape[0] == total) {
    mode = nvinfer1::ScaleMode::kELEMENTWISE;
  } else {
    MS_LOG(ERROR) << "ScaleMode create failed";
    return RET_ERROR;
  }
  bool nd = false;
  // (input * scale + shift) ^ power
  nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};
  nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, 0};
  nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, nullptr, 0};
  if (in_tensors_.size() > SCALE_INDEX) {
    scale.values = in_tensors_[SCALE_INDEX].MutableData();
    scale.count = in_tensors_[SCALE_INDEX].ElementNum();
    nd = input_weight_shape.size() == 1 ? false : true;
  }
  if (in_tensors_.size() > SHIFT_INDEX) {
    shift.values = in_tensors_[SHIFT_INDEX].MutableData();
    shift.count = in_tensors_[SHIFT_INDEX].ElementNum();
  }
  if (in_tensors_.size() > POWER_INDEX) {
    power.values = in_tensors_[POWER_INDEX].MutableData();
    power.count = in_tensors_[POWER_INDEX].ElementNum();
  }
  nvinfer1::IScaleLayer *cal_layer = nullptr;
  if (nd) {
    MS_LOG(WARNING) << "multi dims ScaleMode enter";
    cal_layer = network->addScaleNd(*scale_in_tensor, mode, shift, scale, power, axis);
  } else {
    cal_layer = network->addScale(*scale_in_tensor, mode, shift, scale, power);
  }

  if (cal_layer == nullptr) {
    MS_LOG(ERROR) << "addScaleNd failed for: " << op_name_;
    return RET_ERROR;
  }
  cal_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *op_out_tensor = cal_layer->getOutput(0);

  // add activation
  if (activation_type != schema::ActivationType::ActivationType_NO_ACTIVATION) {
    MS_LOG(WARNING) << "need activation for: " << op_name_;
  }
  op_out_tensor->setName(out_tensors_[0].Name().c_str());
  this->AddInnerOutTensors(op_out_tensor);
  return RET_OK;
}

nvinfer1::ITensor *ScaleTensorRT::AddUnsqueezeOp(nvinfer1::INetworkDefinition *network) {
  nvinfer1::IShuffleLayer *unsqueeze_layer = network->addShuffle(*this->tensorrt_in_tensors_[0]);
  if (unsqueeze_layer == nullptr) {
    MS_LOG(ERROR) << "addShuffle failed for: " << op_name_;
    return nullptr;
  }
  unsqueeze_layer->setName((op_name_ + "_unsqueeze").c_str());
  auto unsqueeze_shape = in_tensors_[0].Shape();
  for (size_t i = 0; i < 4 - unsqueeze_shape.size(); i++) {
    unsqueeze_shape.push_back(1);
  }
  nvinfer1::Dims unsqueeze_dims = lite::ConvertCudaDims(unsqueeze_shape);
  unsqueeze_layer->setReshapeDimensions(unsqueeze_dims);
  return unsqueeze_layer->getOutput(0);
}
}  // namespace mindspore::lite
