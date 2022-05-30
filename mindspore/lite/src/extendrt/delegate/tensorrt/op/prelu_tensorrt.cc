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

#include <numeric>
#include "src/extendrt/delegate/tensorrt/op/prelu_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int PReluTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                             const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }

  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int PReluTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  ITensorHelper prelu_input;
  int ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[0], &prelu_input);
  if (ret != RET_OK || prelu_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim input tensor failed for " << op_name_;
    return ret;
  }
  int input_nbdims = prelu_input.trt_tensor_->getDimensions().nbDims;
  int slope_nbdims = in_tensors_[1].Shape().size();
  auto slope = tensorrt_in_tensors_[1].trt_tensor_;
  if (input_nbdims != slope_nbdims) {
    slope = ConvertTensorWithExpandDims(network, in_tensors_[1], input_nbdims, op_name_ + "_slope");
    tensorrt_in_tensors_[1].trt_tensor_ = slope;
  }
  if (slope == nullptr) {
    MS_LOG(ERROR) << "add const input tensor failed for " << op_name_;
    return RET_ERROR;
  }
  ITensorHelper slope_helper;
  ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[1], &slope_helper);
  if (ret != RET_OK || slope_helper.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim slope tensor failed for " << op_name_;
    return ret;
  }

  auto *prelu_layer = network->addParametricReLU(*prelu_input.trt_tensor_, *slope_helper.trt_tensor_);
  if (prelu_layer == nullptr) {
    MS_LOG(ERROR) << "addParameticReLU failed for TensorRT.";
    return RET_ERROR;
  }

  nvinfer1::ITensor *out_tensor = prelu_layer->getOutput(0);
  out_tensor->setName((op_name_ + "_" + std::to_string(0)).c_str());
  this->AddInnerOutTensors(ITensorHelper{out_tensor, prelu_input.format_, prelu_input.same_format_});
  this->layer_ = prelu_layer;
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_PReLUFusion, PReluTensorRT)
}  // namespace mindspore::lite
