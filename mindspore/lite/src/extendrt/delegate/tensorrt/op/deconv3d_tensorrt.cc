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

#include "src/extendrt/delegate/tensorrt/op/deconv3d_tensorrt.h"
#include <memory>
#include "nnacl/pack.h"
#include "ops/fusion/conv2d_transpose_fusion.h"

namespace mindspore::lite {
int Deconv3dTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE2 && in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}
int Deconv3dTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  auto deconv_op = AsOps<ops::Conv3DTranspose>();
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

  nvinfer1::ITensor *out_tensor = deconv_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{out_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  return RET_OK;
}

void Deconv3dTensorRT::SetAttributes(const std::shared_ptr<ops::Conv3DTranspose> &ms_op,
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
    auto stride_val = std::vector<int64_t>(stride.begin() + INPUT_SIZE2, stride.end());
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
    std::vector<int64_t> padding;
    if (ms_op->HasAttr(ops::kPadList)) {
      auto pad_list_ptr = ms_op->GetAttr(ops::kPadList);
      padding = GetValue<std::vector<int64_t>>(pad_list_ptr);
    }
    std::vector<int64_t> out_pad;
    if (ms_op->HasAttr(ops::kOutputPadding)) {
      auto pad_list_ptr = ms_op->GetAttr(ops::kOutputPadding);
      out_pad = GetValue<std::vector<int64_t>>(pad_list_ptr);
    }
    if (padding.empty() || out_pad.empty()) {
      MS_LOG(WARNING) << "on pad value of " << op_name_;
      return;
    }
    auto padding_val = std::vector<int64_t>(padding.begin(), padding.end());
    auto out_pad_val = std::vector<int64_t>(out_pad.begin() + INPUT_SIZE2, out_pad.end());  // h, w
    if (out_pad_val.size() != DIMENSION_3D || padding_val.size() != DIMENSION_6D) {
      MS_LOG(ERROR) << "invalid size of pad " << op_name_;
      return;
    }
    nvinfer1::Dims dims_pre{};
    dims_pre.nbDims = DIMENSION_3D;
    dims_pre.d[0] = padding_val[0];
    dims_pre.d[1] = padding_val[INPUT_SIZE2];
    dims_pre.d[INPUT_SIZE2] = padding_val[INPUT_SIZE4];
    decon_layer->setPrePadding(dims_pre);
    nvinfer1::Dims dims_post{};
    dims_post.nbDims = DIMENSION_3D;
    dims_post.d[0] = padding_val[1] - out_pad_val[0];
    dims_post.d[1] = padding_val[INPUT_SIZE3] - out_pad_val[1];
    dims_post.d[INPUT_SIZE2] = padding_val[INPUT_SIZE5] - out_pad_val[INPUT_SIZE2];
    decon_layer->setPostPadding(dims_post);
  }
}

Deconv3dTensorRT::~Deconv3dTensorRT() {}
REGISTER_TENSORRT_CREATOR(ops::kNameConv3DTranspose, Deconv3dTensorRT)
}  // namespace mindspore::lite
