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
#include "src/delegate/tensorrt/op/activation_tensorrt.h"
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
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "convert to primitive matmul failed for " << op_name_;
      return RET_ERROR;
    }
    transpose_a_ = primitive->transpose_a() ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;
    transpose_b_ = primitive->transpose_b() ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE;
    activation_ = primitive->activation_type();
  } else if (type_ == schema::PrimitiveType_FullConnection) {
    transpose_a_ = nvinfer1::MatrixOperation::kNONE;
    transpose_b_ = nvinfer1::MatrixOperation::kTRANSPOSE;
  }

  ITensorHelper matmul_a;
  ITensorHelper matmul_b;

  int ret = PreprocessInputs(network, &matmul_a, &matmul_b);
  if (ret != RET_OK || matmul_a.trt_tensor_ == nullptr || matmul_b.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs matmul failed for " << op_name_;
    return RET_ERROR;
  }

  MS_LOG(DEBUG) << "matmul input a " << GetTensorFormat(matmul_a);
  MS_LOG(DEBUG) << "matmul input b " << GetTensorFormat(matmul_b);

  auto matmul_layer =
    network->addMatrixMultiply(*matmul_a.trt_tensor_, transpose_a_, *matmul_b.trt_tensor_, transpose_b_);
  if (matmul_layer == nullptr) {
    MS_LOG(ERROR) << "addMatrixMultiply failed for " << op_name_;
    return RET_ERROR;
  }
  matmul_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = matmul_layer->getOutput(0);
  tensor_name_map_[matmul_layer->getOutput(0)->getName()] = op_name_;

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
    if (bias_layer == nullptr) {
      MS_LOG(ERROR) << "add bias add layer failed for " << op_name_;
      return RET_ERROR;
    }
    auto bias_layer_name = op_name_ + "_bias";
    bias_layer->setName(bias_layer_name.c_str());
    out_tensor = bias_layer->getOutput(0);
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
  this->AddInnerOutTensors(ITensorHelper{out_tensor, out_format_});
  return RET_OK;
}

int MatMulTensorRT::PreprocessInputs(nvinfer1::INetworkDefinition *network, ITensorHelper *matmul_a,
                                     ITensorHelper *matmul_b) {
  int ret;
  if (tensorrt_in_tensors_.size() == INPUT_SIZE2) {
    int a_index =
      GetDimsVolume(tensorrt_in_tensors_[0].trt_tensor_->getDimensions()) == GetDimsVolume(in_tensors_[0].Shape()) ? 0
                                                                                                                   : 1;
    ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[a_index], matmul_a);
    if (ret != RET_OK || matmul_a->trt_tensor_ == nullptr) {
      MS_LOG(ERROR) << "PreprocessInputs2SameDim of matmul input a failed for " << op_name_;
      return RET_ERROR;
    }
    ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[1 - a_index], matmul_b);
    if (ret != RET_OK || matmul_b->trt_tensor_ == nullptr) {
      MS_LOG(ERROR) << "PreprocessInputs2SameDim of matmul input b failed for " << op_name_;
      return RET_ERROR;
    }
    out_format_ = matmul_a->format_;
    if (matmul_a->format_ != matmul_b->format_) {
      MS_LOG(WARNING) << "matmul input tensor has different format " << op_name_;
    }
  } else if (tensorrt_in_tensors_.size() == 1) {
    nvinfer1::ITensor *weight = nullptr;
    int weight_index = in_tensors_[1].Data() != nullptr ? 1 : 0;
    if (in_tensors_[weight_index].Shape().size() <
        static_cast<size_t>(tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims)) {
      weight = ConvertTensorWithExpandDims(network, in_tensors_[weight_index],
                                           in_tensors_[1 - weight_index].Shape().size(), op_name_);
    } else if (in_tensors_[weight_index].Shape().size() ==
               static_cast<size_t>(tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims)) {
      weight = ConvertConstantTensor(network, in_tensors_[weight_index], op_name_);
    } else {
      MS_LOG(ERROR) << "input tensor shape is invalid for " << op_name_;
      return RET_ERROR;
    }
    if (weight == nullptr) {
      MS_LOG(ERROR) << "create constant weight tensor failed for " << op_name_;
      return RET_ERROR;
    }
    if (weight_index == 1) {
      matmul_b->trt_tensor_ = weight;
      ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[0], matmul_a);
      if (ret != RET_OK || matmul_a->trt_tensor_ == nullptr) {
        MS_LOG(ERROR) << "PreprocessInputs2SameDim of matmul input a failed for " << op_name_;
        return RET_ERROR;
      }
      out_format_ = matmul_a->format_;
    } else {
      matmul_a->trt_tensor_ = weight;
      ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[0], matmul_b);
      if (ret != RET_OK || matmul_b->trt_tensor_ == nullptr) {
        MS_LOG(ERROR) << "PreprocessInputs2SameDim of matmul input b failed for " << op_name_;
        return RET_ERROR;
      }
      out_format_ = matmul_b->format_;
    }
  } else {
    MS_LOG(ERROR) << op_name_ << " tensorrt in tensor size is invalid " << tensorrt_in_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
