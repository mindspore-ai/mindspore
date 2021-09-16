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

#include <valarray>
#include "src/delegate/tensorrt/op/reduce_tensorrt.h"

namespace mindspore::lite {
int ReduceTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  auto reduce_op = primitive->value_as_ReduceFusion();
  if (reduce_op == nullptr) {
    MS_LOG(ERROR) << "convert failed";
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
  }
  auto it = reduce_ops_.find(reduce_op->mode());
  if (it != reduce_ops_.end()) {
    reduce_op_ = it->second;
  } else {
    MS_LOG(ERROR) << "unsupported ReduceMode: " << reduce_op->mode();
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  auto reduce_op = op_primitive_->value_as_ReduceFusion();
  if (reduce_op == nullptr) {
    MS_LOG(ERROR) << "convert failed";
    return RET_ERROR;
  }
  bool keep_dims = reduce_op->keep_dims();

  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      tensorrt_in_tensors_[0].format_ == Format::NCHW) {
    out_format_ = Format::NHWC;
  } else {
    out_format_ = tensorrt_in_tensors_[0].format_;
  }

  nvinfer1::ITensor *reduce_input = tensorrt_in_tensors_[0].trt_tensor_;
  // 4 dims support reduce at each axis
  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims != DIMENSION_4D) {
    nvinfer1::IShuffleLayer *unsqueeze_layer = network->addShuffle(*tensorrt_in_tensors_[0].trt_tensor_);
    if (unsqueeze_layer == nullptr) {
      MS_LOG(ERROR) << "add Shuffle op failed for TensorRT.";
      return RET_ERROR;
    }
    unsqueeze_layer->setName((op_name_ + "_unsqueeze4dims").c_str());
    nvinfer1::Dims unsqueeze_dims = tensorrt_in_tensors_[0].trt_tensor_->getDimensions();
    for (int i = unsqueeze_dims.nbDims; i < 4; i++) {
      unsqueeze_dims.d[i] = 1;
    }
    unsqueeze_dims.nbDims = 4;

    unsqueeze_layer->setReshapeDimensions(unsqueeze_dims);
    reduce_input = unsqueeze_layer->getOutput(0);
  }

  uint32_t reduceAxis = GetAxis();
  nvinfer1::IReduceLayer *layer = network->addReduce(*reduce_input, reduce_op_, reduceAxis, keep_dims);
  if (layer == nullptr) {
    MS_LOG(ERROR) << "addReduce failed for TensorRT.";
    return RET_ERROR;
  }
  layer->setName(op_name_.c_str());

  nvinfer1::ITensor *out_tensor = layer->getOutput(0);
  if (in_tensors_[0].Shape().size() != DIMENSION_4D) {
    // queeze to origin dim
    nvinfer1::IShuffleLayer *squeeze_layer = network->addShuffle(*layer->getOutput(0));
    if (squeeze_layer == nullptr) {
      MS_LOG(ERROR) << "add Shuffle op failed for TensorRT.";
      return RET_ERROR;
    }
    squeeze_layer->setName((op_name_ + "_squeeze").c_str());
    nvinfer1::Dims squeeze_dims = ConvertCudaDims(out_tensors_[0].Shape());
    squeeze_layer->setReshapeDimensions(squeeze_dims);
    out_tensor = squeeze_layer->getOutput(0);
  }
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "addReduce output tensor create failed for TensorRT.";
    return RET_ERROR;
  }
  out_tensor->setName(out_tensors_[0].Name().c_str());
  this->AddInnerOutTensors(ITensorHelper{out_tensor, out_format_});
  return RET_OK;
}
uint32_t ReduceTensorRT::GetAxis() {
  // axis
  uint32_t reduceAxis = 0;
  mindspore::MSTensor axis_tensor = this->in_tensors_[1];
  if (axis_tensor.Data() == nullptr) {
    MS_LOG(ERROR) << "invalid axis_tensor";
    return reduceAxis;
  }
  if (axis_tensor.DataType() != DataType::kNumberTypeInt32) {
    MS_LOG(WARNING) << "not int data type";
  }
  int *axis_data = reinterpret_cast<int *>(axis_tensor.MutableData());
  bool need_transpose_axis =
    (out_format_ == Format::NCHW) && (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D);
  uint32_t base = std::pow(2, in_tensors_[0].Shape().size());
  for (int i = 0; i < axis_tensor.ElementNum(); i++) {
    int format_axis_data = need_transpose_axis ? ConvertAxisFromNHWC2NCHW(*axis_data) : *axis_data;
    reduceAxis |= (base - (1u << format_axis_data));
    axis_data++;
  }
  MS_LOG(DEBUG) << "reduceAxis: " << reduceAxis;
  return reduceAxis;
}
}  // namespace mindspore::lite
