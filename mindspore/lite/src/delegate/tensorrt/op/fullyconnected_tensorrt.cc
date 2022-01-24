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

#include "src/delegate/tensorrt/op/fullyconnected_tensorrt.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"
#include "src/delegate/tensorrt/op/activation_tensorrt.h"
namespace mindspore::lite {
constexpr int BIAS_INDEX = 2;
int FullyConnectedTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                                      const std::vector<mindspore::MSTensor> &in_tensors,
                                      const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int FullyConnectedTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  int axis;
  if (type_ == schema::PrimitiveType_FullConnection) {
    auto primitive = this->GetPrimitive()->value_as_FullConnection();
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "convert to primitive FullConnection failed for " << op_name_;
      return RET_ERROR;
    }
    activation_ = primitive->activation_type();
    axis = primitive->axis();
  }
  ITensorHelper fc_input;
  auto ret = PreprocessInputs(network, &fc_input);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PreprocessInputs failed for " << op_name_;
    return ret;
  }
  auto kernel_weight = ConvertWeight(in_tensors_[1]);
  auto bias_weight = ConvertWeight(in_tensors_[BIAS_INDEX]);
  nvinfer1::IFullyConnectedLayer *fc_layer =
    network->addFullyConnected(*(fc_input.trt_tensor_), out_tensors_[0].Shape()[axis], kernel_weight, bias_weight);
  if (fc_layer == nullptr) {
    MS_LOG(ERROR) << "addFullyConnected failed for " << op_name_;
    return RET_ERROR;
  }
  fc_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = fc_layer->getOutput(0);

  if (out_tensor->getDimensions().nbDims != out_tensors_[0].Shape().size()) {
    std::vector<int64_t> squeeze_dim(out_tensors_[0].Shape());
    squeeze_dim[0] = out_tensor->getDimensions().d[0] == -1 ? -1 : squeeze_dim[0];
    out_tensor = Reshape(network, out_tensor, squeeze_dim);
  }
  // add activation
  if (activation_ != schema::ActivationType::ActivationType_NO_ACTIVATION) {
    nvinfer1::ILayer *activation_layer = ActivationTensorRT::AddActivation(network, activation_, 0, 0, 0, out_tensor);
    if (activation_layer == nullptr) {
      MS_LOG(ERROR) << "addActivation for matmul failed";
      return RET_ERROR;
    }
    activation_layer->setName((op_name_ + "_activation").c_str());
    out_tensor = activation_layer->getOutput(0);
  }

  out_tensor->setName((op_name_ + "_output").c_str());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(out_tensor);
  this->AddInnerOutTensors(ITensorHelper{out_tensor, fc_input.format_});
  return RET_OK;
}

int FullyConnectedTensorRT::PreprocessInputs(nvinfer1::INetworkDefinition *network, ITensorHelper *fc_input) {
  auto ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[0], fc_input);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim failed for " << op_name_;
    return ret;
  }
  auto origin_dims = fc_input->trt_tensor_->getDimensions();
  if (origin_dims.nbDims != DIMENSION_4D) {
    std::vector<int64_t> expand_dim(origin_dims.d, origin_dims.d + origin_dims.nbDims);
    for (int i = 0; i < DIMENSION_4D - origin_dims.nbDims; i++) {
      expand_dim.push_back(1);
    }
    fc_input->trt_tensor_ = Reshape(network, fc_input->trt_tensor_, expand_dim);
  }
  return RET_OK;
}
}  // namespace mindspore::lite
