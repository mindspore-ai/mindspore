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

#include "src/delegate/tensorrt/op/gather_tensorrt.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
constexpr int AXIS_INDEX = 2;

int GatherTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "invalid input tensor size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensor size: " << out_tensors.size();
    return RET_ERROR;
  }
  if (in_tensors[1].DataType() != DataType::kNumberTypeInt32) {
    MS_LOG(ERROR) << "Gather indices only support Int32";
    return RET_ERROR;
  }
  if (in_tensors[AXIS_INDEX].ElementNum() == 1) {
    MS_ASSERT(in_tensors[AXIS_INDEX].Data().get());
    axis_ = static_cast<const int *>(in_tensors[AXIS_INDEX].Data().get())[0];
  } else {
    MS_LOG(ERROR) << "TensorRT axis is attribute.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }

  nvinfer1::ITensor *gather_input = this->tensorrt_in_tensors_[0].trt_tensor_;
  if (in_tensors_[0].IsConst()) {
    gather_input = lite::ConvertConstantTensor(network, this->in_tensors_[0]);
    MS_LOG(DEBUG) << "gather input is const tensor " << op_name_;
  }
  if (gather_input == nullptr) {
    MS_LOG(ERROR) << "get gather input failed for: " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::ITensor *indices_tensor = this->tensorrt_in_tensors_[tensorrt_in_tensors_.size() - 1].trt_tensor_;
  if (in_tensors_[1].IsConst()) {
    indices_tensor = lite::ConvertConstantTensor(network, this->in_tensors_[1]);
    MS_LOG(DEBUG) << "gather indices is const tensor " << op_name_;
  }
  if (indices_tensor == nullptr) {
    MS_LOG(ERROR) << "get gather indices failed for: " << op_name_;
    return RET_ERROR;
  }

  Format out_format = tensorrt_in_tensors_[0].format_;
  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      tensorrt_in_tensors_[0].format_ == Format::NCHW) {
    // transpose: NCHW->NHWC
    nvinfer1::IShuffleLayer *transpose_layer_in = NCHW2NHWC(network, *tensorrt_in_tensors_[0].trt_tensor_);
    if (transpose_layer_in == nullptr) {
      MS_LOG(ERROR) << "op action convert failed";
      return RET_ERROR;
    }
    transpose_layer_in->setName((op_name_ + "_transpose2NHWC").c_str());
    gather_input = transpose_layer_in->getOutput(0);
    out_format = Format::NHWC;
  }

  nvinfer1::IGatherLayer *gather_layer =
    network->addGather(*gather_input, *indices_tensor /* indices */, axis_ /* axis */);
  if (gather_layer == nullptr) {
    MS_LOG(ERROR) << "addGather failed for TensorRT.";
    return RET_ERROR;
  }
  gather_layer->setName(op_name_.c_str());
  gather_layer->getOutput(0)->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{gather_layer->getOutput(0), out_format});
  return RET_OK;
}
}  // namespace mindspore::lite
