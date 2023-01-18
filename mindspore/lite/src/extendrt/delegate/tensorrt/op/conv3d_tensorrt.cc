/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/op/conv3d_tensorrt.h"
#include <memory>

namespace mindspore::lite {
constexpr int BIAS_INDEX = 2;
int Conv3DTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                              const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int Conv3DTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  auto conv_op = AsOps<ops::Conv3D>();
  if (conv_op == nullptr) {
    MS_LOG(ERROR) << "op action convert failed";
    return RET_ERROR;
  }

  nvinfer1::ITensor *conv_input = input(ctx, 0).trt_tensor_;

  // transpose weight
  const auto &weight_tensor = in_tensors_[1];
  nvinfer1::Weights kernelWeights = lite::ConvertWeight(weight_tensor);

  // conv
  int nbOutputMaps = weight_tensor.Shape()[0];
  if (nbOutputMaps <= 0) {
    MS_LOG(ERROR) << "out_channel is invalid";
    return RET_ERROR;
  }

  auto kernel_size = conv_op->get_kernel_size();
  if (kernel_size.empty()) {
    MS_LOG(ERROR) << "kernel_size is null";
    return RET_ERROR;
  }
  nvinfer1::Dims kernelSize = lite::ConvertCudaDims(std::vector<int64_t>(kernel_size.begin(), kernel_size.end()));
  if (kernelSize.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
    return RET_ERROR;
  }
  // bias not support
  nvinfer1::Weights biasWeights{};

  nvinfer1::IConvolutionLayer *conv_layer =
    ctx->network()->addConvolutionNd(*conv_input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);

  if (conv_layer == nullptr) {
    MS_LOG(ERROR) << "ConvolutionLayer failed";
    return RET_ERROR;
  }
  conv_layer->setName((op_name_ + "_conv").c_str());
  this->layer_ = conv_layer;

  // add params
  SetAttributes(conv_op, conv_layer);

  // add activation
  auto out_tensor = conv_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{out_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  return RET_OK;
}

void Conv3DTensorRT::SetAttributes(const std::shared_ptr<ops::Conv3D> &conv_op,
                                   nvinfer1::IConvolutionLayer *conv_layer) {
  auto stride = conv_op->get_stride();
  if (!stride.empty()) {
    auto stride_val = std::vector<int64_t>(stride.begin() + 2, stride.end());
    auto dims = ConvertCudaDims(stride_val);
    if (dims.nbDims == -1) {
      MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
      return;
    }
    conv_layer->setStrideNd(dims);
  }

  auto dilation = conv_op->get_dilation();
  if (!dilation.empty()) {
    auto dilation_val = std::vector<int64_t>(dilation.begin() + 2, dilation.end());
    auto dims = ConvertCudaDims(dilation_val);
    if (dims.nbDims == -1) {
      MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
      return;
    }
    conv_layer->setDilationNd(dims);
  }
  int nbGroups = conv_op->get_group();
  if (nbGroups > 0) {
    conv_layer->setNbGroups(nbGroups);
  }

  PadMode pad_mode = conv_op->get_pad_mode();
  if (pad_mode == PadMode::SAME) {
    conv_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  } else {
    std::vector<int64_t> padding;
    padding = conv_op->get_pad();
    if (padding.size() == DIMENSION_6D) {
      auto padding_val = std::vector<int>(padding.begin(), padding.end());
      if (padding_val[0] != padding_val[1] || padding_val[DIMENSION_2D] != padding_val[DIMENSION_3D] ||
          padding_val[DIMENSION_4D] != padding_val[DIMENSION_5D]) {
        MS_LOG(WARNING) << op_name_ << " has different up and down padding value";
        nvinfer1::Dims pre_dims{INPUT_SIZE3, {padding_val[0], padding_val[DIMENSION_2D], padding_val[DIMENSION_4D]}};
        conv_layer->setPrePadding(pre_dims);
        nvinfer1::Dims post_dims{INPUT_SIZE3, {padding_val[1], padding_val[DIMENSION_3D], padding_val[DIMENSION_5D]}};
        conv_layer->setPostPadding(post_dims);
      } else {
        nvinfer1::Dims dims{INPUT_SIZE3, {padding_val[0], padding_val[DIMENSION_2D], padding_val[DIMENSION_4D]}};
        conv_layer->setPaddingNd(dims);
      }
    } else if (padding.empty()) {
      nvinfer1::Dims3 dims;
      conv_layer->setPaddingNd(dims);
    } else {
      MS_LOG(WARNING) << "pad list is invalid for " << op_name_;
    }
  }
}

Conv3DTensorRT::~Conv3DTensorRT() {}
REGISTER_TENSORRT_CREATOR(ops::kNameConv3D, Conv3DTensorRT)
}  // namespace mindspore::lite
