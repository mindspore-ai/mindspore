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

#include "src/delegate/tensorrt/op/convolution_tensorrt.h"
#include "src/delegate/tensorrt/op/activation_tensorrt.h"

namespace mindspore::lite {
constexpr int BIAS_INDEX = 2;

int ConvolutionTensorRT::IsSupport(const schema::Primitive *primitive,
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
  if (in_tensors[0].format() != Format::NHWC && in_tensors[0].format() != Format::NCHW) {
    MS_LOG(ERROR) << "Unsupported input tensor format of " << in_tensors[0].format();
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  const schema::Conv2DFusion *conv_op = this->op_primitive_->value_as_Conv2DFusion();
  if (conv_op == nullptr) {
    MS_LOG(ERROR) << "op action convert failed";
    return RET_ERROR;
  }

  nvinfer1::ITensor *conv_input = tensorrt_in_tensors_[0].trt_tensor_;
  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      tensorrt_in_tensors_[0].format_ == Format::NHWC) {
    // transpose: NHWC->NCHW
    nvinfer1::IShuffleLayer *transpose_layer_in = NHWC2NCHW(network, *tensorrt_in_tensors_[0].trt_tensor_);
    if (transpose_layer_in == nullptr) {
      MS_LOG(ERROR) << "transpose: NHWC->NCHW failed";
      return RET_ERROR;
    }
    transpose_layer_in->setName((op_name_ + "_transpose2NCHW").c_str());
    conv_input = transpose_layer_in->getOutput(0);
  }

  // transpose weight
  const mindspore::MSTensor &weight_tensor = in_tensors_[1];
  nvinfer1::Weights kernelWeights = lite::TransposeWeight(weight_tensor, &pack_weight_);

  // conv
  int nbOutputMaps = weight_tensor.Shape()[0];
  if (nbOutputMaps <= 0) {
    MS_LOG(ERROR) << "out_channel is invalid";
    return RET_ERROR;
  }

  auto kernel_size = conv_op->kernel_size();
  if (kernel_size == nullptr) {
    MS_LOG(ERROR) << "kernel_size is null";
    return RET_ERROR;
  }
  nvinfer1::Dims kernelSize = lite::ConvertCudaDims(std::vector<int64_t>(kernel_size->begin(), kernel_size->end()));

  // bias
  nvinfer1::Weights biasWeights{};
  if (in_tensors_.size() >= INPUT_SIZE3) {
    biasWeights = lite::ConvertWeight(in_tensors_[BIAS_INDEX]);
  } else {
    biasWeights.type = ConvertDataType(weight_tensor.DataType());
    biasWeights.count = 0;
    biasWeights.values = nullptr;
  }

  nvinfer1::IConvolutionLayer *conv_layer =
    network->addConvolutionNd(*conv_input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);

  if (conv_layer == nullptr) {
    MS_LOG(ERROR) << "ConvolutionLayer failed";
    return RET_ERROR;
  }
  conv_layer->setName((op_name_ + "_conv").c_str());

  // add params
  SetAttributes(conv_op, conv_layer);

  // add activation
  nvinfer1::ILayer *activation_layer = nullptr;
  if (conv_op->activation_type() == schema::ActivationType::ActivationType_NO_ACTIVATION) {
    activation_layer = conv_layer;
  } else {
    activation_layer =
      ActivationTensorRT::AddActivation(network, conv_op->activation_type(), 0, conv_layer->getOutput(0));
    if (activation_layer == nullptr) {
      MS_LOG(ERROR) << "addActivation for conv failed";
      return RET_ERROR;
    }
    activation_layer->setName((op_name_ + "_activation").c_str());
  }
  activation_layer->getOutput(0)->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{activation_layer->getOutput(0), Format::NCHW});
  return RET_OK;
}

void ConvolutionTensorRT::SetAttributes(const schema::Conv2DFusion *conv_op, nvinfer1::IConvolutionLayer *conv_layer) {
  auto stride = conv_op->stride();
  if (stride != nullptr) {
    auto stride_val = std::vector<int64_t>(stride->begin(), stride->end());
    auto dims = ConvertCudaDims(stride_val);
    conv_layer->setStrideNd(dims);
  }

  auto dilation = conv_op->dilation();
  if (dilation != nullptr) {
    auto dilation_val = std::vector<int64_t>(dilation->begin(), dilation->end());
    auto dims = ConvertCudaDims(dilation_val);
    conv_layer->setDilationNd(dims);
  }
  int nbGroups = conv_op->group();
  if (nbGroups > 0) {
    conv_layer->setNbGroups(nbGroups);
  }

  schema::PadMode pad_mode = conv_op->pad_mode();
  if (pad_mode == schema::PadMode::PadMode_SAME) {
    conv_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  } else {
    auto padding = conv_op->pad_list();
    if (padding != nullptr) {
      auto padding_val = std::vector<int64_t>(padding->begin(), padding->end());
      nvinfer1::Dims dims{};
      dims.nbDims = 2;
      dims.d[0] = padding_val[0];
      dims.d[1] = padding_val[2];
      conv_layer->setPaddingNd(dims);
    }
  }
}

ConvolutionTensorRT::~ConvolutionTensorRT() {
  if (pack_weight_ != nullptr) {
    free(pack_weight_);
    pack_weight_ = nullptr;
  }
}
}  // namespace mindspore::lite
