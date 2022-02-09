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

#include "src/delegate/tensorrt/op/topk_tensorrt.h"

namespace mindspore::lite {
int TopKTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int TopKTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr || this->tensorrt_in_tensors_.size() != 1) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }
  nvinfer1::TopKOperation red_op = nvinfer1::TopKOperation::kMAX;
  int axis_value = 0;
  int topk = 0;
  bool keep_dims = false;
  if (type_ == schema::PrimitiveType_ArgMaxFusion) {
    red_op = nvinfer1::TopKOperation::kMAX;
    auto max_prim = op_primitive_->value_as_ArgMaxFusion();
    if (max_prim == nullptr) {
      MS_LOG(ERROR) << "convert ArgMaxFusion failed: " << op_name_;
      return RET_ERROR;
    }
    axis_value = max_prim->axis();
    topk = max_prim->top_k();
    keep_dims = max_prim->keep_dims();
  } else if (type_ == schema::PrimitiveType_ArgMinFusion) {
    red_op = nvinfer1::TopKOperation::kMIN;
    auto mim_prim = op_primitive_->value_as_ArgMinFusion();
    if (mim_prim == nullptr) {
      MS_LOG(ERROR) << "convert ArgMinFusion failed: " << op_name_;
      return RET_ERROR;
    }
    axis_value = mim_prim->axis();
    topk = mim_prim->top_k();
    keep_dims = mim_prim->keep_dims();
  } else {
    MS_LOG(ERROR) << "invalid op primitive for " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::ITensor *topk_input = tensorrt_in_tensors_[0].trt_tensor_;
  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      tensorrt_in_tensors_[0].format_ == Format::NCHW) {
    nvinfer1::IShuffleLayer *transpose_layer = NCHW2NHWC(network, *topk_input);
    if (transpose_layer == nullptr) {
      MS_LOG(ERROR) << "create transpose layer failed for " << op_name_;
      return RET_ERROR;
    }
    transpose_layer->setName((op_name_ + "_transpose_in").c_str());
    topk_input = transpose_layer->getOutput(0);
    this->transpose_layer_ = transpose_layer;
  }
  uint32_t reduce_axes = 1 << axis_value;

  nvinfer1::ITopKLayer *topk_layer = network->addTopK(*topk_input, red_op, topk, reduce_axes);
  if (topk_layer == nullptr) {
    MS_LOG(ERROR) << "addTopK failed for: " << op_name_;
    return RET_ERROR;
  }
  this->layer_ = topk_layer;
  topk_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *op_out_tensor = topk_layer->getOutput(1);
  // output 0 is data value, output 1 is index

  if (!keep_dims) {
    MS_LOG(DEBUG) << op_name_ << "add squeeze for not keep dims at index " << axis_value;
    if (op_out_tensor->getDimensions().d[axis_value] != 1) {
      MS_LOG(ERROR) << "output dims is invalid for squeeze: " << op_name_;
      return RET_ERROR;
    }
    nvinfer1::IShuffleLayer *squeeze_layer = network->addShuffle(*op_out_tensor);
    if (squeeze_layer == nullptr) {
      MS_LOG(ERROR) << "add squeeze layer failed for: " << op_name_;
      return RET_ERROR;
    }
    nvinfer1::Dims squeeze_dims{};
    squeeze_dims.nbDims = op_out_tensor->getDimensions().nbDims - 1;
    if (axis_value != squeeze_dims.nbDims) {
      MS_LOG(ERROR) << op_name_ << " reduce squeeze dims need check for axis: " << axis_value;
      return RET_ERROR;
    }
    for (int i = 0; i < squeeze_dims.nbDims; i++) {
      squeeze_dims.d[i] = 0;
      // same with input
    }
    squeeze_layer->setReshapeDimensions(squeeze_dims);
    squeeze_layer->setName((op_name_ + "_squeeze").c_str());
    op_out_tensor = squeeze_layer->getOutput(0);
  }

  op_out_tensor->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{op_out_tensor, Format::NHWC, true});
  return RET_OK;
}
}  // namespace mindspore::lite
