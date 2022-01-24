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
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
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
  out_format_ = tensorrt_in_tensors_[0].format_;
  nvinfer1::ITensor *reduce_input = tensorrt_in_tensors_[0].trt_tensor_;
  MS_LOG(DEBUG) << "origin input " << GetTensorFormat(tensorrt_in_tensors_[0]);
  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      !SameDims(tensorrt_in_tensors_[0].trt_tensor_->getDimensions(), in_tensors_[0].Shape())) {
    if (tensorrt_in_tensors_[0].format_ == Format::NCHW) {
      // NCHW->NHWC
      nvinfer1::IShuffleLayer *transpose_layer = NCHW2NHWC(network, *tensorrt_in_tensors_[0].trt_tensor_);
      if (transpose_layer == nullptr) {
        MS_LOG(ERROR) << "create transpose layer failed for " << op_name_;
        return RET_ERROR;
      }
      transpose_layer->setName((op_name_ + "_transpose_in").c_str());
      reduce_input = transpose_layer->getOutput(0);
      out_format_ = Format::NHWC;
    } else if (tensorrt_in_tensors_[0].format_ == Format::NHWC) {
      // NHWC->NCHW
      nvinfer1::IShuffleLayer *transpose_layer = NHWC2NCHW(network, *tensorrt_in_tensors_[0].trt_tensor_);
      if (transpose_layer == nullptr) {
        MS_LOG(ERROR) << "create transpose layer failed for " << op_name_;
        return RET_ERROR;
      }
      transpose_layer->setName((op_name_ + "_transpose_in").c_str());
      reduce_input = transpose_layer->getOutput(0);
      out_format_ = Format::NCHW;
    } else {
      MS_LOG(WARNING) << "input tensor format needs check: " << op_name_;
    }
  }
  MS_LOG(DEBUG) << "after transpose input " << GetTensorFormat(reduce_input, out_format_, true);

  uint32_t reduceAxis = GetAxis();
  nvinfer1::IReduceLayer *layer =
    network->addReduce(*reduce_input, ConvertTRTReduceMode(reduce_op->mode()), reduceAxis, keep_dims);
  if (layer == nullptr) {
    MS_LOG(ERROR) << "addReduce failed for TensorRT.";
    return RET_ERROR;
  }
  layer->setName(op_name_.c_str());

  nvinfer1::ITensor *out_tensor = layer->getOutput(0);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "addReduce output tensor create failed for TensorRT.";
    return RET_ERROR;
  }
  out_tensor->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{out_tensor, out_format_, true});
  MS_LOG(DEBUG) << "output " << GetTensorFormat(tensorrt_out_tensors_[0]);
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
  CHECK_NULL_RETURN(axis_data);
  for (int i = 0; i < axis_tensor.ElementNum(); i++) {
    int format_axis_data = (*axis_data == -1) ? in_tensors_[0].Shape().size() - 1 : *axis_data;
    MS_LOG(DEBUG) << op_name_ << " reduceAxis at index : " << *axis_data;
    reduceAxis |= 1u << format_axis_data;
    axis_data++;
  }
  return reduceAxis;
}
}  // namespace mindspore::lite
