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

#include "src/litert/delegate/tensorrt/op/topk_tensorrt.h"

namespace mindspore::lite {
int TopKTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1 && in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1 && in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int TopKTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx->network() == nullptr || ReadyInputsNumber(ctx) != 1) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }
  int ret = ParseParams(ctx);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseParams failed for " << op_name_;
    return ret;
  }

  ITensorHelper topk_input;
  ret = PreprocessInputs(ctx, &topk_input);
  if (ret != RET_OK || topk_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "preprocess input failed for " << op_name_;
    return ret;
  }
  axis_ = 1 << axis_value_;
  MS_LOG(DEBUG) << "addTopK input " << GetTensorFormat(topk_input);
  MS_LOG(DEBUG) << op_name_ << " has k: " << top_k_ << ", axis: " << axis_value_;

  nvinfer1::ITopKLayer *topk_layer = ctx->network()->addTopK(*topk_input.trt_tensor_, topk_op_, top_k_, axis_);
  CHECK_NULL_RETURN(topk_layer);
  this->layer_ = topk_layer;
  topk_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *value_out_tensor = topk_layer->getOutput(0);
  nvinfer1::ITensor *index_out_tensor = topk_layer->getOutput(1);
  // output 0 is data value, output 1 is index

  if (value_out_tensor->getDimensions().nbDims != out_tensors_[0].Shape().size()) {
    nvinfer1::Dims out_dims = ConvertCudaDims(out_tensors_[0].Shape());
    out_dims.d[0] = value_out_tensor->getDimensions().d[0];
    value_out_tensor = Reshape(ctx, value_out_tensor, out_dims);
    CHECK_NULL_RETURN(value_out_tensor);
    value_out_tensor->setName((op_name_ + "_value_output").c_str());
    index_out_tensor = Reshape(ctx, index_out_tensor, out_dims);
    CHECK_NULL_RETURN(index_out_tensor);
    index_out_tensor->setName((op_name_ + "_index_output").c_str());
  }
  if (out_tensors_.size() == INPUT_SIZE2) {
    ctx->RegisterTensor(ITensorHelper{value_out_tensor, topk_input.format_, true}, out_tensors_[0].Name());
  }
  ctx->RegisterTensor(ITensorHelper{index_out_tensor, topk_input.format_, true},
                      out_tensors_[out_tensors_.size() == INPUT_SIZE2].Name());
  return RET_OK;
}

int TopKTensorRT::ParseParams(TensorRTContext *ctx) {
  switch (type_) {
    case schema::PrimitiveType_ArgMaxFusion: {
      topk_op_ = nvinfer1::TopKOperation::kMAX;
      auto max_prim = op_primitive_->value_as_ArgMaxFusion();
      CHECK_NULL_RETURN(max_prim);
      axis_value_ = max_prim->axis();
      axis_value_ = axis_value_ > 0 ? axis_value_ : in_tensors_[0].Shape().size() + axis_value_;
      top_k_ = max_prim->top_k();
      break;
    }
    case schema::PrimitiveType_ArgMinFusion: {
      topk_op_ = nvinfer1::TopKOperation::kMIN;
      auto mim_prim = op_primitive_->value_as_ArgMinFusion();
      CHECK_NULL_RETURN(mim_prim);
      axis_value_ = mim_prim->axis();
      axis_value_ = axis_value_ > 0 ? axis_value_ : in_tensors_[0].Shape().size() + axis_value_;
      top_k_ = mim_prim->top_k();
      break;
    }
    case schema::PrimitiveType_TopKFusion: {
      auto topk_prim = op_primitive_->value_as_TopKFusion();
      CHECK_NULL_RETURN(topk_prim);
      topk_op_ = topk_prim->largest() == 1 ? nvinfer1::TopKOperation::kMAX : nvinfer1::TopKOperation::kMIN;
      axis_value_ = topk_prim->axis();
      axis_value_ = axis_value_ > 0 ? axis_value_ : in_tensors_[0].Shape().size() + axis_value_;
      if (in_tensors_.size() < INPUT_SIZE2) {
        MS_LOG(ERROR) << "invalid input size " << in_tensors_.size() << "for " << op_name_;
        return RET_ERROR;
      }
      std::vector<float> tmp(1);
      int ret_k = ParseData2Vector(in_tensors_[1], &tmp);
      if (ret_k != RET_OK) {
        return ret_k;
      }
      top_k_ = tmp[0];
      break;
    }
    default: {
      MS_LOG(ERROR) << op_name_ << " has more primitive type: " << schema::EnumNamePrimitiveType(type_);
      return RET_ERROR;
    }
  }
  // Currently reduceAxes must specify exactly one dimension, and it must be one of the last four dimensions.
  if (axis_value_ != in_tensors_[0].Shape().size() - 1) {
    MS_LOG(ERROR) << op_name_ << " has unsupported axis : " << axis_value_;
    return RET_ERROR;
  }
  return RET_OK;
}
int TopKTensorRT::PreprocessInputs(TensorRTContext *ctx, ITensorHelper *topk_input) {
  auto input_dim = input(ctx, 0).trt_tensor_->getDimensions();
  int ret = RET_ERROR;
  if (input_dim.nbDims == DIMENSION_4D) {
    ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), topk_input);
  } else if (input_dim.nbDims < DIMENSION_4D) {
    // only support 4d
    nvinfer1::Dims4 expect_dim;
    for (int i = 0; i < DIMENSION_4D; i++) {
      if (i < input_dim.nbDims) {
        expect_dim.d[DIMENSION_4D - 1 - i] = input_dim.d[input_dim.nbDims - 1 - i];
      } else {
        expect_dim.d[DIMENSION_4D - 1 - i] = 1;
      }
    }
    topk_input->trt_tensor_ = Reshape(ctx, input(ctx, 0).trt_tensor_, expect_dim);
    CHECK_NULL_RETURN(topk_input->trt_tensor_);
    axis_value_ += (DIMENSION_4D - input_dim.nbDims);
    return RET_OK;
  } else {
    MS_LOG(ERROR) << op_name_ << " has invalid input dims: " << input_dim.nbDims;
  }
  return ret;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_ArgMaxFusion, TopKTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_ArgMinFusion, TopKTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_TopKFusion, TopKTensorRT)
}  // namespace mindspore::lite
