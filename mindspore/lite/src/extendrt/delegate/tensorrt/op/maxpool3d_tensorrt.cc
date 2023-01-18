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

#include "src/extendrt/delegate/tensorrt/op/maxpool3d_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/max_pool3d.h"

namespace mindspore::lite {
int MaxPool3DTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                 const std::vector<TensorInfo> &out_tensors) {
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

int MaxPool3DTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (in_tensors_.size() != 1) {
    MS_LOG(ERROR) << "invalid input tensor size: " << in_tensors_.size();
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << "before transpose " << GetTensorFormat(input(ctx, 0));
  int ret = ParseParams(ctx);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseParams failed for : " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::ITensor *pool_input = input(ctx, 0).trt_tensor_;

  // global version pooling
  if (kernel_size_.empty()) {
    int reduce_axes = ((1 << pool_input->getDimensions().nbDims) - 1) & ~0b11;
    auto *layer = ctx->network()->addReduce(*pool_input, nvinfer1::ReduceOperation::kAVG, reduce_axes, true);
    if (layer == nullptr) {
      MS_LOG(ERROR) << "addReduce for pool failed";
      return RET_ERROR;
    }
    layer->setName(op_name_.c_str());
    this->layer_ = layer;
  } else {
    // pooling layer
    nvinfer1::Dims windowSize = lite::ConvertCudaDims(kernel_size_);
    if (windowSize.nbDims == -1) {
      MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
      return RET_ERROR;
    }
    nvinfer1::IPoolingLayer *pooling_layer = ctx->network()->addPoolingNd(*pool_input, pooling_type_, windowSize);
    if (pooling_layer == nullptr) {
      MS_LOG(ERROR) << "addPoolingNd failed for TensorRT.";
      return RET_ERROR;
    }
    ret = AddParams(pooling_layer);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "AddParams failed for : " << op_name_;
      return RET_ERROR;
    }
    pooling_layer->setName(op_name_.c_str());
    this->layer_ = pooling_layer;
  }

  // add activation
  nvinfer1::ITensor *out_trt_tensor = layer_->getOutput(0);
  auto output_helper = ITensorHelper{out_trt_tensor, Format::NCHW, true};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(output_helper);
  return RET_OK;
}

int MaxPool3DTensorRT::ParseParams(TensorRTContext *ctx) {
  auto pool_primitive = AsOps<ops::MaxPool3D>();
  if (pool_primitive == nullptr) {
    MS_LOG(ERROR) << "convert PoolFusion failed: " << op_name_;
    return RET_ERROR;
  }
  pooling_type_ = nvinfer1::PoolingType::kMAX;

  auto kernel_size = pool_primitive->get_kernel_size();
  if (kernel_size.empty()) {
    MS_LOG(ERROR) << "get kernel size failed: " << op_name_;
    return RET_ERROR;
  }
  kernel_size_ = std::vector<int64_t>(kernel_size.begin() + INPUT_SIZE2, kernel_size.end());

  auto stride = pool_primitive->get_strides();
  if (stride.empty()) {
    MS_LOG(ERROR) << "get stride failed: " << op_name_;
    return RET_ERROR;
  }
  stride_ = std::vector<int64_t>(stride.begin() + INPUT_SIZE2, stride.end());
  auto padding = pool_primitive->get_pad();
  if (padding.empty()) {
    MS_LOG(INFO) << "get padding is null, set to default 0: " << op_name_;
    padding_ = {0, 0, 0, 0, 0, 0};
  } else {
    padding_ = std::vector<int64_t>(padding.begin(), padding.end());
  }

  pad_mode_ = pool_primitive->get_pad_mode();
  return RET_OK;
}

int MaxPool3DTensorRT::AddParams(nvinfer1::IPoolingLayer *pooling_layer) {
  nvinfer1::Dims stride_dims = ConvertCudaDims(stride_);
  if (stride_dims.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
    return RET_ERROR;
  }
  pooling_layer->setStrideNd(stride_dims);
  if (pad_mode_ == PadMode::SAME) {
    pooling_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  } else {
    if (padding_.size() != DIMENSION_6D) {
      MS_LOG(ERROR) << "Invalid padding " << padding_ << ", op: " << op_name_;
      return RET_ERROR;
    }
    nvinfer1::Dims pre_dims{};
    pre_dims.nbDims = DIMENSION_3D;
    pre_dims.d[0] = padding_[kDim0];
    pre_dims.d[1] = padding_[kDim2];
    pre_dims.d[INPUT_SIZE2] = padding_[kDim4];
    pooling_layer->setPrePadding(pre_dims);

    nvinfer1::Dims post_dims{};
    post_dims.nbDims = DIMENSION_3D;
    post_dims.d[0] = padding_[kDim1];
    post_dims.d[1] = padding_[kDim3];
    post_dims.d[INPUT_SIZE2] = padding_[kDim5];
    pooling_layer->setPostPadding(post_dims);
  }
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameMaxPool3D, MaxPool3DTensorRT)
}  // namespace mindspore::lite
