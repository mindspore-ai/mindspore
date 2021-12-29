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
  if (tensorrt_in_tensors_.size() < INPUT_SIZE2 && in_tensors_.size() >= INPUT_SIZE2) {
    int const_ms_tensor_index = in_tensors_[0].IsConst() ? 0 : 1;
    auto const_input = ConvertConstantTensor(network, in_tensors_[const_ms_tensor_index], op_name_);
    if (const_input == nullptr) {
      MS_LOG(ERROR) << "add const input tensor failed for " << op_name_;
      return RET_ERROR;
    }
    tensorrt_in_tensors_.push_back(ITensorHelper{const_input});
  }

  int indices_tensor_index = tensorrt_in_tensors_[0].trt_tensor_->getType() == nvinfer1::DataType::kINT32 ? 0 : 1;
  ITensorHelper gather_input;
  int ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[1 - indices_tensor_index], &gather_input);
  if (ret != RET_OK || gather_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim gather failed for " << op_name_;
    return RET_ERROR;
  }
  ITensorHelper indices_tensor;
  ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[indices_tensor_index], &indices_tensor);
  if (ret != RET_OK || indices_tensor.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim indices failed for " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::IGatherLayer *gather_layer =
    network->addGather(*gather_input.trt_tensor_, *indices_tensor.trt_tensor_ /* indices */, axis_ /* axis */);
  if (gather_layer == nullptr) {
    MS_LOG(ERROR) << "addGather failed for TensorRT.";
    return RET_ERROR;
  }

  gather_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *op_output = gather_layer->getOutput(0);
  // keep shape
  if (in_tensors_[1].Shape().empty()) {
    auto squeeze = network->addShuffle(*op_output);
    if (squeeze == nullptr) {
      MS_LOG(ERROR) << "add output squeeze failed for " << op_name_;
      return RET_ERROR;
    }
    squeeze->setName((op_name_ + "_squeeze_out").c_str());
    auto old_shape = ConvertMSShape(op_output->getDimensions());
    old_shape.erase(old_shape.begin() + axis_);
    squeeze->setReshapeDimensions(ConvertCudaDims(old_shape));
    op_output = squeeze->getOutput(0);
  }
  op_output->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{op_output, gather_input.format_, gather_input.same_format_});
  return RET_OK;
}
}  // namespace mindspore::lite
