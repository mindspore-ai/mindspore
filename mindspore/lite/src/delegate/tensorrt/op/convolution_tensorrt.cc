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
#include "src/delegate/tensorrt/tensorrt_utils.h"
#include "nnacl/pack.h"

namespace mindspore::lite {
constexpr int BIAS_INDEX = 2;

int ConvolutionTensorRT::IsSupport(const schema::Primitive *primitive,
                                   const std::vector<mindspore::MSTensor> &in_tensors,
                                   const std::vector<mindspore::MSTensor> &out_tensors) {
  if (in_tensors.size() != 2 && in_tensors.size() != 3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
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
  // transpose: NHWC->NCHW
  nvinfer1::IShuffleLayer *transpose_layer_in = NHWC2NCHW(network, *tensorrt_in_tensors_[0]);
  if (transpose_layer_in == nullptr) {
    MS_LOG(ERROR) << "transpose: NHWC->NCHW failed";
    return RET_ERROR;
  }
  transpose_layer_in->setName((op_name_ + "_transpose2NCHW").c_str());

  // conv
  int nbOutputMaps = conv_op->out_channel();
  if (nbOutputMaps <= 0) {
    MS_LOG(ERROR) << "out_channel is invalid";
    return RET_ERROR;
  }

  nvinfer1::Dims kernelSize{};
  auto kernel_size = conv_op->kernel_size();
  if (kernel_size == nullptr) {
    MS_LOG(ERROR) << "kernel_size is null";
    return RET_ERROR;
  }
  kernelSize.nbDims = static_cast<int>(kernel_size->size());
  for (int i = 0; i < kernelSize.nbDims; i++) {
    kernelSize.d[i] = kernel_size->Get(i);
  }

  // transpose weight
  auto weight_tensor = in_tensors_[1];
  nvinfer1::Weights kernelWeights{};
  kernelWeights.count = weight_tensor.ElementNum();
  if (lite::ConvertDataType(weight_tensor.DataType()) != nvinfer1::DataType::kFLOAT) {
    MS_LOG(WARNING) << "kernelWeights data type is not float";
  }
  kernelWeights.type = nvinfer1::DataType::kFLOAT;
  auto weight_shape = weight_tensor.Shape();
  float *src_val = reinterpret_cast<float *>(weight_tensor.MutableData());
  pack_weight_ = reinterpret_cast<float *>(malloc(weight_tensor.ElementNum() * sizeof(float)));
  if (pack_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  PackNHWCToNCHWFp32(src_val, pack_weight_, weight_shape[0], weight_shape[1] * weight_shape[2], weight_shape[3], 0, 0);
  kernelWeights.values = pack_weight_;

  // bias
  nvinfer1::Weights biasWeights{};
  if (in_tensors_.size() >= INPUT_SIZE3) {
    auto bias_tensor = in_tensors_[BIAS_INDEX];
    biasWeights.type = ConvertDataType(bias_tensor.DataType());
    biasWeights.values = bias_tensor.MutableData();
    biasWeights.count = bias_tensor.ElementNum();
  } else {
    biasWeights.type = nvinfer1::DataType::kFLOAT;
    biasWeights.count = 0;
    biasWeights.values = nullptr;
  }

  nvinfer1::IConvolutionLayer *conv_layer =
    network->addConvolutionNd(*transpose_layer_in->getOutput(0), nbOutputMaps, kernelSize, kernelWeights, biasWeights);

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
  } else if (conv_op->activation_type() == schema::ActivationType::ActivationType_RELU) {
    activation_layer = network->addActivation(*conv_layer->getOutput(0), nvinfer1::ActivationType::kRELU);
    if (activation_layer == nullptr) {
      MS_LOG(ERROR) << "addActivation for conv failed";
      return RET_ERROR;
    }
    activation_layer->setName((op_name_ + "_relu").c_str());
  } else if (conv_op->activation_type() == schema::ActivationType::ActivationType_RELU6) {
    auto activation = network->addActivation(*conv_layer->getOutput(0), nvinfer1::ActivationType::kCLIP);
    if (activation == nullptr) {
      MS_LOG(ERROR) << "addActivation for conv failed";
      return RET_ERROR;
    }
    activation_layer->setName((op_name_ + "_relu6").c_str());
    activation->setAlpha(0);
    activation->setBeta(6);
    activation_layer = activation;
  } else {
    MS_LOG(DEBUG) << "Unsupported op action type for conv TensorRT: " << conv_op->activation_type();
    return RET_ERROR;
  }

  // transpose: NCHW->NHWC
  nvinfer1::IShuffleLayer *transpose_layer_out = NCHW2NHWC(network, *activation_layer->getOutput(0));
  if (transpose_layer_out == nullptr) {
    MS_LOG(ERROR) << "op action convert failed";
    return RET_ERROR;
  }
  transpose_layer_out->setName((op_name_ + "_transpose2NHWC").c_str());

  this->AddInnerOutTensors(transpose_layer_out->getOutput(0));
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
