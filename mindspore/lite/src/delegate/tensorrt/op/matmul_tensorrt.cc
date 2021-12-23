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

#include "src/delegate/tensorrt/op/matmul_tensorrt.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"
namespace mindspore::lite {
constexpr int BIAS_INDEX = 2;

int MatMulTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                              const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2 && in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int MatMulTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (type_ == schema::PrimitiveType_MatMulFusion) {
    auto primitive = this->GetPrimitive()->value_as_MatMulFusion();
    transpose_a_ = primitive->transpose_a() ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;
    transpose_b_ = primitive->transpose_b() ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;
  } else if (type_ == schema::PrimitiveType_FullConnection) {
    transpose_a_ = nvinfer1::MatrixOperation::kNONE;
    transpose_b_ = nvinfer1::MatrixOperation::kTRANSPOSE;
  }

  nvinfer1::ITensor *matmul_input = tensorrt_in_tensors_[0].trt_tensor_;
  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      tensorrt_in_tensors_[0].format_ == Format::NCHW) {
    // transpose: NCHW->NHWC
    nvinfer1::IShuffleLayer *transpose_layer_in = NCHW2NHWC(network, *tensorrt_in_tensors_[0].trt_tensor_);
    if (transpose_layer_in == nullptr) {
      MS_LOG(ERROR) << "op action convert failed";
      return RET_ERROR;
    }
    transpose_layer_in->setName((op_name_ + "_transpose2NHWC").c_str());
    matmul_input = transpose_layer_in->getOutput(0);
  }

  nvinfer1::ITensor *weight = nullptr;
  if (in_tensors_[1].Shape().size() < static_cast<size_t>(matmul_input->getDimensions().nbDims)) {
    weight = ConvertTensorWithExpandDims(network, in_tensors_[1], in_tensors_[0].Shape().size(), op_name_);
  } else if (in_tensors_[1].Shape().size() == static_cast<size_t>(matmul_input->getDimensions().nbDims)) {
    weight = ConvertConstantTensor(network, in_tensors_[1], op_name_);
  } else {
    MS_LOG(ERROR) << "input tensor shape is invalid for " << op_name_;
    return RET_ERROR;
  }
  if (weight == nullptr) {
    MS_LOG(ERROR) << "create constant weight tensor failed for " << op_name_;
    return RET_ERROR;
  }

  MS_LOG(DEBUG) << "matmul input a " << GetTensorFormat(matmul_input);
  MS_LOG(DEBUG) << "matmul input b " << GetTensorFormat(weight);

  auto matmul_layer = network->addMatrixMultiply(*matmul_input, transpose_a_, *weight, transpose_b_);
  matmul_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = matmul_layer->getOutput(0);

  if (in_tensors_.size() == BIAS_INDEX + 1) {
    nvinfer1::ITensor *bias = nullptr;
    if (in_tensors_[BIAS_INDEX].Shape().size() < static_cast<size_t>(out_tensor->getDimensions().nbDims)) {
      bias =
        ConvertTensorWithExpandDims(network, in_tensors_[BIAS_INDEX], out_tensor->getDimensions().nbDims, op_name_);
    } else if (in_tensors_[BIAS_INDEX].Shape().size() == static_cast<size_t>(out_tensor->getDimensions().nbDims)) {
      bias = ConvertConstantTensor(network, in_tensors_[BIAS_INDEX], op_name_);
    } else {
      MS_LOG(ERROR) << "input tensor shape is invalid for " << op_name_;
      return RET_ERROR;
    }
    if (bias == nullptr) {
      MS_LOG(ERROR) << "create constant bias tensor failed for " << op_name_;
      return RET_ERROR;
    }
    auto bias_layer = network->addElementWise(*matmul_layer->getOutput(0), *bias, nvinfer1::ElementWiseOperation::kSUM);
    auto bias_layer_name = op_name_ + "_bias";
    bias_layer->setName(bias_layer_name.c_str());
    out_tensor = bias_layer->getOutput(0);
  }
  out_tensor->setName((op_name_ + "_output").c_str());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(out_tensor);
  this->AddInnerOutTensors(ITensorHelper{out_tensor});
  return RET_OK;
}
}  // namespace mindspore::lite
