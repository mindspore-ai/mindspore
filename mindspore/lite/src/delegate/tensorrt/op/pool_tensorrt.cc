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
  if (in_tensors[0].format() != Format::NHWC && in_tensors[0].format() != Format::NCHW) {
    MS_LOG(ERROR) << "Unsupported input tensor format of " << in_tensors[0].format();
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (tensorrt_in_tensors_.size() != 1) {
    MS_LOG(ERROR) << "invalid input tensor size: " << tensorrt_in_tensors_.size();
    return RET_ERROR;
  }
  int ret = ParseParams();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseParams failed for : " << op_name_;
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << "before transpose "
                << GetTensorFormat(tensorrt_in_tensors_[0].trt_tensor_, tensorrt_in_tensors_[0].format_);

  nvinfer1::ITensor *pool_input = tensorrt_in_tensors_[0].trt_tensor_;
  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      tensorrt_in_tensors_[0].format_ == Format::NHWC) {
    // transpose: NHWC->NCHW
    nvinfer1::IShuffleLayer *transpose_layer_in = NHWC2NCHW(network, *tensorrt_in_tensors_[0].trt_tensor_);
    if (transpose_layer_in == nullptr) {
      MS_LOG(ERROR) << "transpose: NHWC->NCHW failed";
      return RET_ERROR;
    }
    transpose_layer_in->setName((op_name_ + "_transpose2NCHW").c_str());
    pool_input = transpose_layer_in->getOutput(0);
  }

  // pooling layer
  nvinfer1::Dims windowSize = lite::ConvertCudaDims(kernel_size_);
  nvinfer1::IPoolingLayer *pooling_layer = network->addPoolingNd(*pool_input, pooling_type_, windowSize);
  if (pooling_layer == nullptr) {
    MS_LOG(ERROR) << "addPoolingNd failed for TensorRT.";
    return RET_ERROR;
  }
  AddParams(pooling_layer);
  pooling_layer->setName(op_name_.c_str());

  // add activation
  nvinfer1::ILayer *activation_layer = nullptr;
  if (activation_type_ == schema::ActivationType::ActivationType_NO_ACTIVATION) {
    activation_layer = pooling_layer;
  } else {
    activation_layer = ActivationTensorRT::AddActivation(network, activation_type_, 0, pooling_layer->getOutput(0));
    if (activation_layer == nullptr) {
      MS_LOG(ERROR) << "addActivation for pool failed";
      return RET_ERROR;
    }
    activation_layer->setName((op_name_ + "_activation").c_str());
  }
  activation_layer->getOutput(0)->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{activation_layer->getOutput(0), Format::NCHW});
  MS_LOG(DEBUG) << "output " << GetTensorFormat(activation_layer->getOutput(0), Format::NCHW);
  return RET_OK;
}

int PoolTensorRT::ParseParams() {
  switch (type_) {
    case (schema::PrimitiveType_AvgPoolFusion): {
      const schema::AvgPoolFusion *pool_primitive = this->GetPrimitive()->value_as_AvgPoolFusion();
      if (pool_primitive == nullptr) {
        MS_LOG(ERROR) << "convert PoolFusion failed: " << op_name_;
        return RET_ERROR;
      }
      pooling_type_ = nvinfer1::PoolingType::kAVERAGE;

      auto kernel_size = pool_primitive->kernel_size();
      if (kernel_size == nullptr) {
        MS_LOG(ERROR) << "get kernel size failed: " << op_name_;
        return RET_ERROR;
      }
      kernel_size_ = std::vector<int64_t>(kernel_size->begin(), kernel_size->end());

      auto stride = pool_primitive->strides();
      if (stride == nullptr) {
        MS_LOG(ERROR) << "get stride failed: " << op_name_;
        return RET_ERROR;
      }
      stride_ = std::vector<int64_t>(stride->begin(), stride->end());

      auto padding = pool_primitive->pad();
      if (padding == nullptr) {
        MS_LOG(ERROR) << "get padding failed: " << op_name_;
        return RET_ERROR;
      }
      padding_ = std::vector<int64_t>(padding->begin(), padding->end());

      pad_mode_ = pool_primitive->pad_mode();
      activation_type_ = pool_primitive->activation_type();
      break;
    }
    case (schema::PrimitiveType_MaxPoolFusion): {
      const schema::MaxPoolFusion *pool_primitive = this->GetPrimitive()->value_as_MaxPoolFusion();
      if (pool_primitive == nullptr) {
        MS_LOG(ERROR) << "convert PoolFusion failed: " << op_name_;
        return RET_ERROR;
      }
      pooling_type_ = nvinfer1::PoolingType::kMAX;

      auto kernel_size = pool_primitive->kernel_size();
      if (kernel_size == nullptr) {
        MS_LOG(ERROR) << "get kernel size failed: " << op_name_;
        return RET_ERROR;
      }
      kernel_size_ = std::vector<int64_t>(kernel_size->begin(), kernel_size->end());

      auto stride = pool_primitive->strides();
      if (stride == nullptr) {
        MS_LOG(ERROR) << "get stride failed: " << op_name_;
        return RET_ERROR;
      }
      stride_ = std::vector<int64_t>(stride->begin(), stride->end());

      auto padding = pool_primitive->pad();
      if (padding == nullptr) {
        MS_LOG(ERROR) << "get padding failed: " << op_name_;
        return RET_ERROR;
      }
      padding_ = std::vector<int64_t>(padding->begin(), padding->end());

      pad_mode_ = pool_primitive->pad_mode();
      activation_type_ = pool_primitive->activation_type();
      break;
    }
    default: {
      MS_LOG(ERROR) << "unsupported primitive type of " << type_ << " for node: " << op_name_;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void PoolTensorRT::AddParams(nvinfer1::IPoolingLayer *pooling_layer) {
  nvinfer1::Dims stride_dims = ConvertCudaDims(stride_);
  pooling_layer->setStrideNd(stride_dims);
  if (pad_mode_ == schema::PadMode::PadMode_SAME) {
    pooling_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
  } else {
    nvinfer1::Dims dims{};
    dims.nbDims = DIMENSION_2D;
    dims.d[0] = padding_[0];
    dims.d[1] = padding_[DIMENSION_2D];
    pooling_layer->setPaddingNd(dims);
  }
}
}  // namespace mindspore::lite
