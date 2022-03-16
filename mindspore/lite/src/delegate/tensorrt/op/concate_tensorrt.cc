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

#include "src/delegate/tensorrt/op/concate_tensorrt.h"
#include <algorithm>

namespace mindspore::lite {
int ConcateTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() < INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}
int ConcateTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  // Concat
  auto concate_op = this->op_primitive_->value_as_Concat();
  if (concate_op == nullptr) {
    MS_LOG(ERROR) << "concate_op convert failed";
    return RET_ERROR;
  }
  if (tensorrt_in_tensors_.size() != in_tensors_.size()) {
    MS_LOG(ERROR) << "concate_op in tensor is invalid, trt tensor has " << tensorrt_in_tensors_.size()
                  << ", but origin ms tensor has " << in_tensors_.size();
    return RET_ERROR;
  }

  nvinfer1::ITensor *trt_input_tensors[tensorrt_in_tensors_.size()];
  int ret = PreProcessInputs(network, trt_input_tensors);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PreProcessInputs failed for " << op_name_;
    return ret;
  }

  int axis = concate_op->axis();
  if (axis == -1) {
    axis = tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims - 1;
  }
  if (!same_format_) {
    if (trt_input_tensors[0]->getDimensions().nbDims == DIMENSION_4D && out_format_ == Format::NCHW) {
      // when inputs all NCHW, change axis
      axis = ConvertAxisFromNHWC2NCHW(axis);
      MS_LOG(DEBUG) << "concate axis change to " << axis << " when using NCHW format.";
    } else {
      MS_LOG(WARNING) << "input tensor format needs check, convert concat axis failed for " << op_name_;
    }
  }

  nvinfer1::IConcatenationLayer *concate_layer =
    network->addConcatenation(trt_input_tensors, static_cast<int>(tensorrt_in_tensors_.size()));
  if (concate_layer == nullptr) {
    MS_LOG(ERROR) << "addConcatenation failed for TensorRT.";
    return RET_ERROR;
  }

  if (axis != RET_INVALID_OP_ATTR) {
    concate_layer->setAxis(axis);
  }
  concate_layer->setName(op_name_.c_str());
  concate_layer->getOutput(0)->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{concate_layer->getOutput(0), out_format_, same_format_});
  this->layer_ = concate_layer;
  return RET_OK;
}

int ConcateTensorRT::PreProcessInputs(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *trt_input_tensors[]) {
  int input_nbDims = tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims;
  out_format_ = tensorrt_in_tensors_[0].format_;
  same_format_ = tensorrt_in_tensors_[0].same_format_;

  for (size_t i = 0; i < tensorrt_in_tensors_.size(); i++) {
    if (tensorrt_in_tensors_[i].trt_tensor_->getDimensions().nbDims != input_nbDims) {
      MS_LOG(ERROR) << "dims of inputs is invalid for " << op_name_;
      return RET_ERROR;
    }
    // keep origin format if all input format are the same
    if (input_nbDims == DIMENSION_4D && tensorrt_in_tensors_[i].format_ != out_format_) {
      out_format_ = Format::NHWC;
    }
  }

  // make sure all inputs are same format
  if (input_nbDims == DIMENSION_4D) {
    for (size_t i = 0; i < tensorrt_in_tensors_.size(); i++) {
      if (tensorrt_in_tensors_[i].format_ == out_format_) {
        trt_input_tensors[i] = tensorrt_in_tensors_[i].trt_tensor_;
        MS_LOG(DEBUG) << "concate input " << GetTensorFormat(tensorrt_in_tensors_[i]);
      } else {
        nvinfer1::IShuffleLayer *transpose_layer = NCHW2NHWC(network, *tensorrt_in_tensors_[i].trt_tensor_);
        if (transpose_layer == nullptr) {
          MS_LOG(ERROR) << "op action convert failed";
          return RET_ERROR;
        }
        trt_input_tensors[i] = transpose_layer->getOutput(0);
        this->transpose_layer_ = transpose_layer;
        same_format_ = true;
        MS_LOG(DEBUG) << "concate input " << GetTensorFormat(trt_input_tensors[i], Format::NHWC, true);
      }
    }
  } else {
    for (size_t i = 0; i < tensorrt_in_tensors_.size(); i++) {
      trt_input_tensors[i] = tensorrt_in_tensors_[i].trt_tensor_;
      MS_LOG(DEBUG) << "concate input " << GetTensorFormat(tensorrt_in_tensors_[i]);
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
