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
#include "src/delegate/tensorrt/op/activation_tensorrt.h"
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
  if (in_tensors.size() != INPUT_SIZE2 && in_tensors.size() != INPUT_SIZE3 && in_tensors.size() != INPUT_SIZE4) {
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

  // mode of scale
  int64_t axis = scale_op->axis();
  if (axis == -1) {
    axis = static_cast<int64_t>(in_tensors_[0].Shape().size() - 1);
  }
  mode_ = GetScaleMode(axis);
  out_format_ = tensorrt_in_tensors_[0].format_;
  MS_LOG(DEBUG) << "before transpose " << GetTensorFormat(tensorrt_in_tensors_[0].trt_tensor_, out_format_);

  nvinfer1::ITensor *scale_in_tensor = PreProcessInputTensor(network);
  if (scale_in_tensor == nullptr) {
    MS_LOG(ERROR) << "PreProcessInputTensor failed: " << op_name_;
    return RET_ERROR;
  }

  MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(scale_in_tensor, out_format_);

  bool nd = false;
  // (input * scale + shift) ^ power
  nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};
  nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, 0};
  nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, nullptr, 0};
  if (in_tensors_.size() > SCALE_INDEX) {
    scale.values = in_tensors_[SCALE_INDEX].MutableData();
    MS_ASSERT(scale.values);
    scale.count = in_tensors_[SCALE_INDEX].ElementNum();
    scale.type = ConvertDataType(in_tensors_[SCALE_INDEX].DataType());
    shift.type = scale.type;
    power.type = scale.type;
    nd = in_tensors_[1].Shape().size() == 1 ? false : true;
  }
  if (in_tensors_.size() > SHIFT_INDEX) {
    shift.values = in_tensors_[SHIFT_INDEX].MutableData();
    MS_ASSERT(shift.values);
    shift.count = in_tensors_[SHIFT_INDEX].ElementNum();
  }
  if (in_tensors_.size() > POWER_INDEX) {
    power.values = in_tensors_[POWER_INDEX].MutableData();
    MS_ASSERT(power.values);
    power.count = in_tensors_[POWER_INDEX].ElementNum();
  }
  nvinfer1::IScaleLayer *cal_layer = nullptr;

  if (nd) {
    MS_LOG(WARNING) << "multi dims ScaleMode enter";
    cal_layer = network->addScaleNd(*scale_in_tensor, mode_, shift, scale, power, axis);
  } else {
    cal_layer = network->addScale(*scale_in_tensor, mode_, shift, scale, power);
  }

  if (cal_layer == nullptr) {
    MS_LOG(ERROR) << "addScaleNd failed for: " << op_name_;
    return RET_ERROR;
  }
  cal_layer->setName(op_name_.c_str());

  // add activation
  nvinfer1::ITensor *activation_tensor = cal_layer->getOutput(0);
  if (activation_type != schema::ActivationType::ActivationType_NO_ACTIVATION) {
    auto activation_layer = ActivationTensorRT::AddActivation(network, activation_type, 0, cal_layer->getOutput(0));
    if (activation_layer == nullptr) {
      MS_LOG(ERROR) << "addActivation for scale failed";
      return RET_ERROR;
    }
    activation_layer->setName((op_name_ + "_activation").c_str());
    activation_tensor = activation_layer->getOutput(0);
  }

  // squeeze to origin dim
  nvinfer1::ITensor *op_out_tensor = activation_tensor;
  if (activation_tensor->getDimensions().nbDims > static_cast<int>(out_tensors_[0].Shape().size())) {
    op_out_tensor = AddSqueezeOp(activation_tensor, network);
  }
  op_out_tensor->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{op_out_tensor, out_format_});
  MS_LOG(DEBUG) << "output " << GetTensorFormat(op_out_tensor, out_format_);
  return RET_OK;
}

nvinfer1::ITensor *ScaleTensorRT::PreProcessInputTensor(nvinfer1::INetworkDefinition *network) {
  nvinfer1::ITensor *scale_in_tensor = tensorrt_in_tensors_[0].trt_tensor_;
  if (in_tensors_[0].Shape().size() < INPUT_SIZE4) {
    // unsqueeze input Itensor to 4 dims
    scale_in_tensor = AddUnsqueezeOp(network);
    if (scale_in_tensor == nullptr) {
      MS_LOG(ERROR) << "AddUnsqueezeOp failed";
      return nullptr;
    }
  } else if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
             mode_ == nvinfer1::ScaleMode::kCHANNEL && tensorrt_in_tensors_[0].format_ == Format::NHWC) {
    // per channel input format should be nchw, otherwise should be same with scale nhwc
    // transpose: NHWC->NCHW
    nvinfer1::IShuffleLayer *transpose_layer_in = NHWC2NCHW(network, *tensorrt_in_tensors_[0].trt_tensor_);
    if (transpose_layer_in == nullptr) {
      MS_LOG(ERROR) << "op action convert failed";
      return nullptr;
    }
    transpose_layer_in->setName((op_name_ + "_transpose2NCHW").c_str());
    scale_in_tensor = transpose_layer_in->getOutput(0);
    out_format_ = Format::NCHW;
  } else if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
             tensorrt_in_tensors_[0].format_ == Format::NCHW && mode_ == nvinfer1::ScaleMode::kELEMENTWISE) {
    // transpose: NCHW->NHWC
    nvinfer1::IShuffleLayer *transpose_layer_in = NCHW2NHWC(network, *tensorrt_in_tensors_[0].trt_tensor_);
    if (transpose_layer_in == nullptr) {
      MS_LOG(ERROR) << "op action convert failed";
      return nullptr;
    }
    transpose_layer_in->setName((op_name_ + "_transpose2NHWC").c_str());
    scale_in_tensor = transpose_layer_in->getOutput(0);
    out_format_ = Format::NHWC;
  }
  return scale_in_tensor;
}

nvinfer1::ScaleMode ScaleTensorRT::GetScaleMode(int64_t axis) {
  nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kUNIFORM;
  auto input_data_shape = in_tensors_[0].Shape();
  auto input_weight_shape = in_tensors_[1].Shape();
  int total = std::accumulate(input_data_shape.begin(), input_data_shape.end(), 1, std::multiplies<int>());
  MS_LOG(DEBUG) << "input tensor element cnt: " << total;
  if (input_weight_shape.size() == 0 || (input_weight_shape.size() == 1 && input_weight_shape[0] == 1)) {
    mode = nvinfer1::ScaleMode::kUNIFORM;
  } else if ((axis < static_cast<int64_t>(input_data_shape.size()) && input_weight_shape.size() == 1 &&
              input_data_shape[axis] == input_weight_shape[0]) ||
             (input_data_shape.size() == DIMENSION_4D && axis == DIMENSION_3D)) {
    mode = nvinfer1::ScaleMode::kCHANNEL;
  } else if (input_weight_shape.size() == 1 && input_weight_shape[0] == total) {
    mode = nvinfer1::ScaleMode::kELEMENTWISE;
  } else {
    MS_LOG(ERROR) << "ScaleMode create failed: " << op_name_;
    return mode;
  }
  MS_LOG(DEBUG) << op_name_ << " ScaleMode(UNIFORM 0, CHANNEL 1, ELEMENTWISE 2): " << static_cast<int>(mode);
  return mode;
}

nvinfer1::ITensor *ScaleTensorRT::AddUnsqueezeOp(nvinfer1::INetworkDefinition *network) {
  nvinfer1::IShuffleLayer *unsqueeze_layer = network->addShuffle(*this->tensorrt_in_tensors_[0].trt_tensor_);
  if (unsqueeze_layer == nullptr) {
    MS_LOG(ERROR) << "addShuffle failed for: " << op_name_;
    return nullptr;
  }
  unsqueeze_layer->setName((op_name_ + "_unsqueeze").c_str());
  auto unsqueeze_shape = in_tensors_[0].Shape();
  size_t unsqueeze_size = DIMENSION_4D - unsqueeze_shape.size();
  for (size_t i = 0; i < unsqueeze_size; i++) {
    unsqueeze_shape.push_back(1);
  }
  nvinfer1::Dims unsqueeze_dims = lite::ConvertCudaDims(unsqueeze_shape);
  unsqueeze_layer->setReshapeDimensions(unsqueeze_dims);
  return unsqueeze_layer->getOutput(0);
}

nvinfer1::ITensor *ScaleTensorRT::AddSqueezeOp(nvinfer1::ITensor *in_tensor, nvinfer1::INetworkDefinition *network) {
  nvinfer1::IShuffleLayer *squeeze_layer = network->addShuffle(*in_tensor);
  if (squeeze_layer == nullptr) {
    MS_LOG(ERROR) << "addShuffle failed for: " << op_name_;
    return nullptr;
  }
  squeeze_layer->setName((op_name_ + "_squeeze").c_str());
  nvinfer1::Dims squeeze_dims = lite::ConvertCudaDims(out_tensors_[0].Shape());
  MS_LOG(DEBUG) << "squeeze_dims cnt for scale: " << squeeze_dims.nbDims;
  squeeze_layer->setReshapeDimensions(squeeze_dims);
  return squeeze_layer->getOutput(0);
}
}  // namespace mindspore::lite
