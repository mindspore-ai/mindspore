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

#include "src/delegate/tensorrt/op/elementwise_tensorrt.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int ElementWiseTensorRT::IsSupport(const schema::Primitive *primitive,
                                   const std::vector<mindspore::MSTensor> &in_tensors,
                                   const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  std::map<schema::PrimitiveType, nvinfer1::ElementWiseOperation> element_wise_ops = {
    {schema::PrimitiveType_AddFusion, nvinfer1::ElementWiseOperation::kSUM},
    {schema::PrimitiveType_PowFusion, nvinfer1::ElementWiseOperation::kPOW},
    {schema::PrimitiveType_DivFusion, nvinfer1::ElementWiseOperation::kDIV},
    {schema::PrimitiveType_SubFusion, nvinfer1::ElementWiseOperation::kSUB},
    {schema::PrimitiveType_MulFusion, nvinfer1::ElementWiseOperation::kPROD},
  };
  auto iter_op = element_wise_ops.find(this->type_);
  if (iter_op != element_wise_ops.end()) {
    element_wise_op_ = iter_op->second;
  } else {
    // PrimitiveType_Eltwise
    auto eltwise_op = op_primitive_->value_as_Eltwise();
    if (eltwise_op == nullptr) {
      MS_LOG(ERROR) << "convert to Eltwise failed: " << op_name_;
      return RET_ERROR;
    }
    schema::EltwiseMode eltwiseMode = eltwise_op->mode();
    std::map<schema::EltwiseMode, nvinfer1::ElementWiseOperation> eltwise_modes = {
      {schema::EltwiseMode::EltwiseMode_SUM, nvinfer1::ElementWiseOperation::kSUM},
      {schema::EltwiseMode::EltwiseMode_PROD, nvinfer1::ElementWiseOperation::kPROD},
      {schema::EltwiseMode::EltwiseMode_MAXIMUM, nvinfer1::ElementWiseOperation::kMAX},
    };
    auto iter_mode = eltwise_modes.find(eltwiseMode);
    if (iter_mode != eltwise_modes.end()) {
      element_wise_op_ = iter_mode->second;
    } else {
      MS_LOG(ERROR) << "unsupported type for ElementWise op" << op_name_;
      return RET_ERROR;
    }
  }

  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "invalid input tensort size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensort size: " << out_tensors.size();
    return RET_ERROR;
  }

  // if constant tensor is scalar, it needs to know another input tensor's shape to broadcast
  if (in_tensors[0].Shape()[0] == -1 && in_tensors[1].Shape().size() == 0) {
    MS_LOG(ERROR) << "invalid all input tensor shape unknown for: " << op_name_;
    return RET_ERROR;
  }

  return RET_OK;
}

int ElementWiseTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network or input tensor size is invalid";
    return RET_ERROR;
  }
  first_in_tensor_index_ =
    SameDims(tensorrt_in_tensors_[0].trt_tensor_->getDimensions(), in_tensors_[0].Shape()) ? 0 : 1;

  if (this->tensorrt_in_tensors_.size() != INPUT_SIZE2) {
    int ret = AddConstTensor(network);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "AddConstTensor failed for " << op_name_;
      return ret;
    }
  }
  MS_LOG(DEBUG) << "before transpose "
                << GetTensorFormat(tensorrt_in_tensors_[first_in_tensor_index_].trt_tensor_,
                                   tensorrt_in_tensors_[first_in_tensor_index_].format_);
  MS_LOG(DEBUG) << "before transpose "
                << GetTensorFormat(tensorrt_in_tensors_[1 - first_in_tensor_index_].trt_tensor_,
                                   tensorrt_in_tensors_[1 - first_in_tensor_index_].format_);

  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      tensorrt_in_tensors_[0].format_ != tensorrt_in_tensors_[1].format_) {
    // when inputs format are different, change to NHWC
    int transpose_input_tensor = tensorrt_in_tensors_[0].format_ == Format::NCHW ? 0 : 1;
    nvinfer1::IShuffleLayer *transpose_layer =
      NCHW2NHWC(network, *tensorrt_in_tensors_[transpose_input_tensor].trt_tensor_);
    if (transpose_layer == nullptr) {
      MS_LOG(ERROR) << "op action convert failed";
      return RET_ERROR;
    }
    transpose_layer->setName((op_name_ + "_input_transpose2NHWC").c_str());
    tensorrt_in_tensors_[transpose_input_tensor].trt_tensor_ = transpose_layer->getOutput(0);
    tensorrt_in_tensors_[transpose_input_tensor].format_ = Format::NHWC;
  } else if (tensorrt_in_tensors_[0].format_ != tensorrt_in_tensors_[1].format_) {
    MS_LOG(ERROR) << "elementwise op inputs are in different format: " << op_name_;
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << "after transpose "
                << GetTensorFormat(tensorrt_in_tensors_[first_in_tensor_index_].trt_tensor_,
                                   tensorrt_in_tensors_[first_in_tensor_index_].format_);
  MS_LOG(DEBUG) << "after transpose "
                << GetTensorFormat(tensorrt_in_tensors_[1 - first_in_tensor_index_].trt_tensor_,
                                   tensorrt_in_tensors_[1 - first_in_tensor_index_].format_);

  nvinfer1::IElementWiseLayer *cal_layer =
    network->addElementWise(*tensorrt_in_tensors_[first_in_tensor_index_].trt_tensor_,
                            *tensorrt_in_tensors_[1 - first_in_tensor_index_].trt_tensor_, element_wise_op_);

  if (cal_layer == nullptr) {
    MS_LOG(ERROR) << "addElementWise failed for TensorRT.";
    return RET_ERROR;
  }
  cal_layer->setName(op_name_.c_str());

  nvinfer1::ITensor *op_out_tensor = cal_layer->getOutput(0);
  if (op_out_tensor == nullptr) {
    MS_LOG(ERROR) << "addElementWise out tensor is nullptr.";
    return RET_ERROR;
  }
  // add activation
  nvinfer1::ITensor *activation_out_tensor = AddActivation(network, op_out_tensor);
  op_out_tensor = (activation_out_tensor == nullptr) ? op_out_tensor : activation_out_tensor;

  // scale and shift
  if (type_ == schema::PrimitiveType_PowFusion) {
    auto pow_op = op_primitive_->value_as_PowFusion();
    if (pow_op == nullptr) {
      MS_LOG(ERROR) << "PowFusion convert failed.";
      return RET_ERROR;
    }
    float scale = pow_op->scale();
    float shift = pow_op->shift();
    if (abs(scale - 1) >= 1.0e-05 || abs(shift - 0) >= 1.0e-05) {
      MS_LOG(WARNING) << "deal with scale and shift for pow op";
    }
  }
  op_out_tensor->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{op_out_tensor, tensorrt_in_tensors_[1].format_});
  MS_LOG(DEBUG) << "output " << GetTensorFormat(op_out_tensor, tensorrt_in_tensors_[1].format_);
  return RET_OK;
}

nvinfer1::ITensor *ElementWiseTensorRT::AddActivation(nvinfer1::INetworkDefinition *network,
                                                      nvinfer1::ITensor *in_tensor) {
  schema::ActivationType activation = schema::ActivationType::ActivationType_NO_ACTIVATION;
  switch (type_) {
    case schema::PrimitiveType_AddFusion: {
      auto sum_op = op_primitive_->value_as_AddFusion();
      if (sum_op == nullptr) {
        MS_LOG(ERROR) << "AddFusion convert failed.";
        return nullptr;
      }
      activation = sum_op->activation_type();
      break;
    }
    case schema::PrimitiveType_DivFusion: {
      auto div_op = op_primitive_->value_as_DivFusion();
      if (div_op == nullptr) {
        MS_LOG(ERROR) << "DivFusion convert failed.";
        return nullptr;
      }
      activation = div_op->activation_type();
      break;
    }
    case schema::PrimitiveType_SubFusion: {
      auto sub_op = op_primitive_->value_as_SubFusion();
      if (sub_op == nullptr) {
        MS_LOG(ERROR) << "SubFusion convert failed.";
        return nullptr;
      }
      activation = sub_op->activation_type();
      break;
    }
    case schema::PrimitiveType_MulFusion: {
      auto mul_op = op_primitive_->value_as_MulFusion();
      if (mul_op == nullptr) {
        MS_LOG(ERROR) << "MulFusion convert failed.";
        return nullptr;
      }
      activation = mul_op->activation_type();
      break;
    }
    default:
      MS_LOG(DEBUG) << "no activation need for: " << op_name_;
  }
  nvinfer1::ITensor *activation_out_tensor = nullptr;
  if (activation != schema::ActivationType::ActivationType_NO_ACTIVATION) {
    MS_LOG(WARNING) << "op: " << op_name_ << " has activation";
  }
  return activation_out_tensor;
}
int ElementWiseTensorRT::AddConstTensor(nvinfer1::INetworkDefinition *network) {
  // create ITensor from MS constant tensor of index 1 - first_in_tensor_index_
  nvinfer1::ITensor *constant_input = nullptr;
  if (this->in_tensors_[1 - first_in_tensor_index_].Shape().size() == 0 ||
      this->in_tensors_[1 - first_in_tensor_index_].ElementNum() == 1) {
    constant_input = lite::ConvertScalarToITensor(network, this->in_tensors_[first_in_tensor_index_].Shape().size(),
                                                  in_tensors_[1 - first_in_tensor_index_].Data().get(),
                                                  in_tensors_[1 - first_in_tensor_index_].DataType());
    if (constant_input == nullptr) {
      MS_LOG(ERROR) << "create Itensor from scalar tensor failed: " << op_name_;
      return RET_ERROR;
    }
    this->AddInnerInTensors(ITensorHelper{constant_input, tensorrt_in_tensors_[0].format_});
  } else {
    constant_input = lite::ConvertConstantTensor(network, in_tensors_[1 - first_in_tensor_index_]);
    if (constant_input == nullptr) {
      MS_LOG(ERROR) << "create Itensor from constant tensor failed: " << op_name_;
      return RET_ERROR;
    }
    this->AddInnerInTensors(ITensorHelper{constant_input, Format::NHWC});
  }
  return RET_OK;
}
}  // namespace mindspore::lite
