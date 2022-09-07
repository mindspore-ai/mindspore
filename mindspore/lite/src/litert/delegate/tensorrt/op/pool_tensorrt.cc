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

#include "src/litert/delegate/tensorrt/op/pool_tensorrt.h"
#include "src/litert/delegate/tensorrt/op/activation_tensorrt.h"
#include "src/litert/delegate/tensorrt/tensorrt_utils.h"

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

int PoolTensorRT::AddInnerOp(TensorRTContext *ctx) {
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
  if (input(ctx, 0).trt_tensor_->getDimensions().nbDims == DIMENSION_4D && input(ctx, 0).format_ == Format::NHWC) {
    // transpose: NHWC->NCHW
    nvinfer1::IShuffleLayer *transpose_layer_in = NHWC2NCHW(ctx, *input(ctx, 0).trt_tensor_);
    if (transpose_layer_in == nullptr) {
      MS_LOG(ERROR) << "transpose: NHWC->NCHW failed";
      return RET_ERROR;
    }
    transpose_layer_in->setName((op_name_ + "_transpose2NCHW").c_str());
    this->transpose_layer_ = transpose_layer_in;
    pool_input = transpose_layer_in->getOutput(0);
  }

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
    AddParams(pooling_layer);
    pooling_layer->setName(op_name_.c_str());
    this->layer_ = pooling_layer;
  }

  // add activation
  nvinfer1::ILayer *activation_layer = nullptr;
  if (activation_type_ == schema::ActivationType::ActivationType_NO_ACTIVATION) {
    activation_layer = this->layer_;
  } else {
    activation_layer =
      ActivationTensorRT::AddActivation(ctx, activation_type_, 0, 0, 0, this->layer_->getOutput(0), device_id_);
    if (activation_layer == nullptr) {
      MS_LOG(ERROR) << "addActivation for pool failed";
      return RET_ERROR;
    }
    activation_layer->setName((op_name_ + "_activation").c_str());
  }
  nvinfer1::ITensor *out_trt_tensor = activation_layer->getOutput(0);
  auto output_helper = ITensorHelper{out_trt_tensor, Format::NCHW, false};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(output_helper);
  return RET_OK;
}

int PoolTensorRT::ParseParams(TensorRTContext *ctx) {
  switch (type_) {
    case (schema::PrimitiveType_AvgPoolFusion): {
      const schema::AvgPoolFusion *pool_primitive = this->GetPrimitive()->value_as_AvgPoolFusion();
      if (pool_primitive == nullptr) {
        MS_LOG(ERROR) << "convert PoolFusion failed: " << op_name_;
        return RET_ERROR;
      }
      pooling_type_ = nvinfer1::PoolingType::kAVERAGE;

      auto stride = pool_primitive->strides();
      if (stride == nullptr) {
        MS_LOG(ERROR) << "get stride failed: " << op_name_;
        return RET_ERROR;
      }
      stride_ = std::vector<int64_t>(stride->begin(), stride->end());
      auto kernel_size = pool_primitive->kernel_size();
      if (kernel_size == nullptr) {
        MS_LOG(WARNING) << op_name_ << "don't has kernel size";
      } else {
        kernel_size_ = std::vector<int64_t>(kernel_size->begin(), kernel_size->end());
      }
      auto padding = pool_primitive->pad();
      if (padding != nullptr && padding->size() != DIMENSION_4D) {
        MS_LOG(ERROR) << op_name_ << "has invalid pad dims: " << padding->size();
        return RET_ERROR;
      } else if (padding == nullptr || padding->size() == 0) {
        padding_ = std::vector<int64_t>(DIMENSION_4D, 0);
      } else {
        padding_ = std::vector<int64_t>(padding->begin(), padding->end());
      }

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
        MS_LOG(INFO) << "get padding is null, set to default 0: " << op_name_;
        padding_ = {0, 0, 0, 0};
      } else {
        padding_ = std::vector<int64_t>(padding->begin(), padding->end());
      }

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
  if (stride_dims.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
    return;
  }
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
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_AvgPoolFusion, PoolTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_MaxPoolFusion, PoolTensorRT)
}  // namespace mindspore::lite
