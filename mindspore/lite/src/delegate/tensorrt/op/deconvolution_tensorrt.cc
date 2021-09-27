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

#include "src/delegate/tensorrt/op/deconvolution_tensorrt.h"
#include "src/delegate/tensorrt/op/activation_tensorrt.h"
#include "nnacl/pack.h"

namespace mindspore::lite {
int DeconvolutionTensorRT::IsSupport(const schema::Primitive *primitive,
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
int DeconvolutionTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  const schema::Conv2dTransposeFusion *deconv_op = this->op_primitive_->value_as_Conv2dTransposeFusion();
  if (deconv_op == nullptr) {
    MS_LOG(ERROR) << "op action convert failed";
    return RET_ERROR;
  }
  nvinfer1::ITensor *deconv_input = tensorrt_in_tensors_[0].trt_tensor_;
  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      tensorrt_in_tensors_[0].format_ == Format::NHWC) {
    // transpose: NHWC->NCHW
    nvinfer1::IShuffleLayer *transpose_layer_in = NHWC2NCHW(network, *tensorrt_in_tensors_[0].trt_tensor_);
    if (transpose_layer_in == nullptr) {
      MS_LOG(ERROR) << "transpose: NHWC->NCHW failed";
      return RET_ERROR;
    }
    transpose_layer_in->setName((op_name_ + "_transpose2NCHW").c_str());
    deconv_input = transpose_layer_in->getOutput(0);
  }

  // transpose weight
  const mindspore::MSTensor &weight_tensor = in_tensors_[1];
  nvinfer1::Weights kernelWeights = lite::TransposeWeight(weight_tensor, &pack_weight_);

  // deconv basic params
  int nbOutputMaps = weight_tensor.Shape()[0];
  if (nbOutputMaps <= 0) {
    MS_LOG(ERROR) << "out_channel is invalid";
    return RET_ERROR;
  }

  auto kernel_size = deconv_op->kernel_size();
  if (kernel_size == nullptr) {
    MS_LOG(ERROR) << "kernel_size is null";
    return RET_ERROR;
  }
  nvinfer1::Dims kernelSize = lite::ConvertCudaDims(std::vector<int64_t>(kernel_size->begin(), kernel_size->end()));

  // bias
  nvinfer1::Weights biasWeights{};
  if (in_tensors_.size() >= INPUT_SIZE3) {
    biasWeights = lite::ConvertWeight(in_tensors_[INPUT_SIZE3 - 1]);
  } else {
    biasWeights.type = ConvertDataType(weight_tensor.DataType());
    biasWeights.count = 0;
    biasWeights.values = nullptr;
  }

  nvinfer1::IDeconvolutionLayer *deconv_layer =
    network->addDeconvolutionNd(*deconv_input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);

  if (deconv_layer == nullptr) {
    MS_LOG(ERROR) << "DeconvolutionLayer failed";
    return RET_ERROR;
  }
  deconv_layer->setName((op_name_ + "_deconv").c_str());

  // set extra params
  SetAttributes(deconv_op, deconv_layer);

  // add activation
  nvinfer1::ILayer *activation_layer = nullptr;
  if (deconv_op->activation_type() == schema::ActivationType::ActivationType_NO_ACTIVATION) {
    activation_layer = deconv_layer;
  } else {
    activation_layer =
      ActivationTensorRT::AddActivation(network, deconv_op->activation_type(), 0, deconv_layer->getOutput(0));
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

void DeconvolutionTensorRT::SetAttributes(const schema::Conv2dTransposeFusion *ms_op,
                                          nvinfer1::IDeconvolutionLayer *decon_layer) {
  // kernel_size
  auto kernel_size = ms_op->kernel_size();
  auto kernel_size_val = std::vector<int64_t>(kernel_size->begin(), kernel_size->end());
  nvinfer1::Dims kernel_size_dims = lite::ConvertCudaDims(kernel_size_val);
  decon_layer->setKernelSizeNd(kernel_size_dims);

  // nbOutputMaps
  int32_t nbOutputMaps = static_cast<int32_t>(ms_op->out_channel());
  decon_layer->setNbOutputMaps(nbOutputMaps);

  // stride
  auto stride = ms_op->stride();
  auto stride_val = std::vector<int64_t>(stride->begin(), stride->end());
  nvinfer1::Dims stride_dims = lite::ConvertCudaDims(stride_val);
  decon_layer->setStrideNd(stride_dims);

  // nbGroups
  int32_t nbGroups = static_cast<int32_t>(ms_op->group());
  decon_layer->setNbGroups(nbGroups);

  // padding
  schema::PadMode pad_mode = ms_op->pad_mode();
  if (pad_mode == schema::PadMode::PadMode_SAME) {
    decon_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  } else {
    auto padding = ms_op->pad_list();
    auto out_pad = ms_op->output_paddings();
    if (padding == nullptr || out_pad == nullptr) {
      MS_LOG(WARNING) << "on pad value of " << op_name_;
      return;
    }
    auto padding_val = std::vector<int64_t>(padding->begin(), padding->end());
    auto out_pad_val = std::vector<int64_t>(out_pad->begin(), out_pad->end());  // h, w
    if (out_pad_val.size() != DIMENSION_2D || padding_val.size() != DIMENSION_4D) {
      MS_LOG(ERROR) << "invalid size of pad " << op_name_;
      return;
    }
    nvinfer1::Dims dims_pre{};
    dims_pre.nbDims = 2;
    dims_pre.d[0] = padding_val[0];  // up
    dims_pre.d[1] = padding_val[2];  // left
    decon_layer->setPrePadding(dims_pre);
    nvinfer1::Dims dims_post{};
    dims_post.nbDims = 2;
    dims_post.d[0] = padding_val[1] - out_pad_val[0];  // down
    dims_post.d[1] = padding_val[3] - out_pad_val[1];  // right
    decon_layer->setPostPadding(dims_post);
  }
}

DeconvolutionTensorRT::~DeconvolutionTensorRT() {
  if (pack_weight_ != nullptr) {
    free(pack_weight_);
    pack_weight_ = nullptr;
  }
}
}  // namespace mindspore::lite
