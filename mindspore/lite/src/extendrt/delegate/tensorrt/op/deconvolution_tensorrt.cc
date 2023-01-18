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

#include "src/extendrt/delegate/tensorrt/op/deconvolution_tensorrt.h"
#include <memory>
#include "src/extendrt/delegate/tensorrt/op/activation_tensorrt.h"
#include "nnacl/pack.h"
#include "ops/fusion/conv2d_transpose_fusion.h"

namespace mindspore::lite {
int DeconvolutionTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                     const std::vector<TensorInfo> &out_tensors) {
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
int DeconvolutionTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  auto deconv_op = AsOps<ops::Conv2dTransposeFusion>();
  if (deconv_op == nullptr) {
    MS_LOG(ERROR) << "op action convert failed";
    return RET_ERROR;
  }
  nvinfer1::ITensor *deconv_input = input(ctx, 0).trt_tensor_;

  // transpose weight
  const auto &weight_tensor = in_tensors_[1];
  nvinfer1::Weights kernelWeights = lite::ConvertWeight(weight_tensor);

  // deconv basic params
  int nbOutputMaps = weight_tensor.Shape()[0];
  if (nbOutputMaps <= 0) {
    MS_LOG(ERROR) << "out_channel is invalid";
    return RET_ERROR;
  }

  auto kernel_size = deconv_op->get_kernel_size();
  if (kernel_size.empty()) {
    MS_LOG(ERROR) << "kernel_size is null";
    return RET_ERROR;
  }
  nvinfer1::Dims kernelSize = lite::ConvertCudaDims(std::vector<int64_t>(kernel_size.begin(), kernel_size.end()));
  if (kernelSize.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
    return RET_ERROR;
  }
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
    ctx->network()->addDeconvolutionNd(*deconv_input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);

  if (deconv_layer == nullptr) {
    MS_LOG(ERROR) << "DeconvolutionLayer failed";
    return RET_ERROR;
  }
  deconv_layer->setName((op_name_ + "_deconv").c_str());
  this->layer_ = deconv_layer;
  // set extra params
  SetAttributes(deconv_op, deconv_layer);

  // add activation
  nvinfer1::ILayer *activation_layer = nullptr;
  ActivationType activation_type = ActivationType::NO_ACTIVATION;
  if (deconv_op->HasAttr(ops::kActivationType)) {
    activation_type = deconv_op->get_activation_type();
  }
  if (activation_type == ActivationType::NO_ACTIVATION) {
    activation_layer = deconv_layer;
  } else {
    activation_layer = ActivationTensorRT::AddActivation(ctx, activation_type, 0, 0, 0, deconv_layer->getOutput(0),
                                                         op_name_, device_id_);
    if (activation_layer == nullptr) {
      MS_LOG(ERROR) << "addActivation for conv failed";
      return RET_ERROR;
    }
    activation_layer->setName((op_name_ + "_activation").c_str());
  }
  nvinfer1::ITensor *out_tensor = activation_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{out_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  return RET_OK;
}

void DeconvolutionTensorRT::SetAttributes(const std::shared_ptr<ops::Conv2dTransposeFusion> &ms_op,
                                          nvinfer1::IDeconvolutionLayer *decon_layer) {
  // kernel_size
  auto kernel_size = ms_op->get_kernel_size();
  if (!kernel_size.empty()) {
    auto kernel_size_val = std::vector<int64_t>(kernel_size.begin(), kernel_size.end());
    nvinfer1::Dims kernel_size_dims = lite::ConvertCudaDims(kernel_size_val);
    if (kernel_size_dims.nbDims == -1) {
      MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
      return;
    }
    decon_layer->setKernelSizeNd(kernel_size_dims);
  }

  // nbOutputMaps
  int nbOutputMaps = in_tensors_[1].Shape()[1];
  decon_layer->setNbOutputMaps(nbOutputMaps);

  // stride
  auto stride = ms_op->get_stride();
  if (!stride.empty()) {
    auto stride_val = std::vector<int64_t>(stride.begin(), stride.end());
    nvinfer1::Dims stride_dims = lite::ConvertCudaDims(stride_val);
    if (stride_dims.nbDims == -1) {
      MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
      return;
    }
    decon_layer->setStrideNd(stride_dims);
  }

  // nbGroups
  int32_t nbGroups = static_cast<int32_t>(ms_op->get_group());
  decon_layer->setNbGroups(nbGroups);

  // padding
  PadMode pad_mode = ms_op->get_pad_mode();
  if (pad_mode == PadMode::SAME) {
    decon_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  } else {
    auto padding = ms_op->get_pad_list();
    std::vector<int64_t> out_pad(DIMENSION_2D, 0);
    if (ms_op->HasAttr(ops::kOutputPaddings)) {
      out_pad = ms_op->get_output_paddings();
    }
    if (padding.empty() || out_pad.empty()) {
      MS_LOG(WARNING) << "on pad value of " << op_name_;
      return;
    }
    auto padding_val = std::vector<int64_t>(padding.begin(), padding.end());
    auto out_pad_val = std::vector<int64_t>(out_pad.begin(), out_pad.end());  // h, w
    if (out_pad_val.size() != DIMENSION_2D || padding_val.size() != DIMENSION_4D) {
      MS_LOG(ERROR) << "invalid size of pad " << op_name_;
      return;
    }
    nvinfer1::Dims dims_pre{};
    dims_pre.nbDims = DIMENSION_2D;
    dims_pre.d[0] = padding_val[0];  // up
    dims_pre.d[1] = padding_val[2];  // left
    decon_layer->setPrePadding(dims_pre);
    nvinfer1::Dims dims_post{};
    dims_post.nbDims = DIMENSION_2D;
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
REGISTER_TENSORRT_CREATOR(ops::kNameConv2dTransposeFusion, DeconvolutionTensorRT)
}  // namespace mindspore::lite
