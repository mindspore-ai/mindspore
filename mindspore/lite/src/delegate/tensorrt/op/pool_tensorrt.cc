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

#include "src/delegate/tensorrt/op/pool_tensorrt.h"
#include "src/delegate/tensorrt/op/activation_tensorrt.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int PoolTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                            const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
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

int PoolTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  const schema::AvgPoolFusion *pool_primitive = this->GetPrimitive()->value_as_AvgPoolFusion();
  if (pool_primitive == nullptr) {
    MS_LOG(ERROR) << "convert PoolFusion failed: " << op_name_;
    return RET_ERROR;
  }
  if (tensorrt_in_tensors_.size() != 1) {
    MS_LOG(ERROR) << "invalid input tensor size: " << tensorrt_in_tensors_.size();
    return RET_ERROR;
  }
  // transpose: NHWC->NCHW
  nvinfer1::IShuffleLayer *transpose_layer_in = NHWC2NCHW(network, *tensorrt_in_tensors_[0]);
  if (transpose_layer_in == nullptr) {
    MS_LOG(ERROR) << "transpose: NHWC->NCHW failed";
    return RET_ERROR;
  }
  transpose_layer_in->setName((op_name_ + "_transpose2NCHW").c_str());

  // pooling layer
  nvinfer1::PoolingType pooling_type = nvinfer1::PoolingType::kAVERAGE;
  auto kernel_size = pool_primitive->kernel_size();
  if (kernel_size == nullptr) {
    MS_LOG(ERROR) << "get kernel size failed: " << op_name_;
    return RET_ERROR;
  }
  std::vector<int64_t> kernel_size_val = std::vector<int64_t>(kernel_size->begin(), kernel_size->end());
  nvinfer1::Dims windowSize = lite::ConvertCudaDims(kernel_size_val);
  nvinfer1::IPoolingLayer *pooling_layer =
    network->addPoolingNd(*transpose_layer_in->getOutput(0), pooling_type, windowSize);
  if (pooling_layer == nullptr) {
    MS_LOG(ERROR) << "addPoolingNd failed for TensorRT.";
    return RET_ERROR;
  }
  AddParams(pool_primitive, pooling_layer);
  pooling_layer->setName(op_name_.c_str());

  // add activation
  nvinfer1::ILayer *activation_layer = nullptr;
  if (pool_primitive->activation_type() == schema::ActivationType::ActivationType_NO_ACTIVATION) {
    activation_layer = pooling_layer;
  } else {
    activation_layer =
      ActivationTensorRT::AddActivation(network, pool_primitive->activation_type(), 0, pooling_layer->getOutput(0));
    if (activation_layer == nullptr) {
      MS_LOG(ERROR) << "addActivation for pool failed";
      return RET_ERROR;
    }
    activation_layer->setName((op_name_ + "_activation").c_str());
  }
  // transpose: NCHW->NHWC
  nvinfer1::IShuffleLayer *transpose_layer_out = NCHW2NHWC(network, *activation_layer->getOutput(0));
  if (transpose_layer_out == nullptr) {
    MS_LOG(ERROR) << "op action convert failed";
    return RET_ERROR;
  }
  transpose_layer_out->setName((op_name_ + "_transpose2NHWC").c_str());
  transpose_layer_out->getOutput(0)->setName(out_tensors_[0].Name().c_str());
  this->AddInnerOutTensors(transpose_layer_out->getOutput(0));
  return RET_OK;
}

void PoolTensorRT::AddParams(const schema::AvgPoolFusion *primitive, nvinfer1::IPoolingLayer *pooling_layer) {
  auto stride = primitive->strides();
  std::vector<int64_t> stride_val = std::vector<int64_t>(stride->begin(), stride->end());
  nvinfer1::Dims stride_dims = ConvertCudaDims(stride_val);
  pooling_layer->setStrideNd(stride_dims);

  schema::PadMode pad_mode = primitive->pad_mode();
  if (pad_mode == schema::PadMode::PadMode_SAME) {
    pooling_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  }
  auto padding = primitive->pad();
  if (padding != nullptr) {
    auto padding_val = std::vector<int64_t>(padding->begin(), padding->end());
    nvinfer1::Dims dims{};
    dims.nbDims = 2;
    dims.d[0] = padding_val[1];
    dims.d[1] = padding_val[2];
    pooling_layer->setPaddingNd(dims);
  }
}
}  // namespace mindspore::lite
