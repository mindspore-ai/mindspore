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

#include <unordered_map>
#include "src/extendrt/delegate/tensorrt/op/topk_tensorrt.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/fusion/arg_max_fusion.h"
#include "ops/fusion/arg_min_fusion.h"
#include "ops/fusion/topk_fusion.h"

namespace mindspore::lite {
namespace {
nvinfer1::ITensor *TopkReshape(TensorRTContext *ctx, nvinfer1::ITensor *input, int axis) {
  auto squeeze = ctx->network()->addShuffle(*input);
  if (squeeze == nullptr) {
    return nullptr;
  }
  auto old_shape = ConvertMSShape(input->getDimensions());
  old_shape.erase(old_shape.begin() + axis);
  squeeze->setReshapeDimensions(ConvertCudaDims(old_shape));
  return squeeze->getOutput(0);
}
}  // namespace

int TopKTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                            const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != 1 && in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1 && out_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  if (type_ != ops::kNameTopKFusion) {
    // need reshape
    dynamic_shape_params_.support_hw_dynamic_ = false;
  }
  if ((type_ == ops::kNameTopKFusion || type_ == ops::kNameTopK) && in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "TopkFusion or Topk need 2 input tensors for " << op_name_;
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
  bool need_expand = (topk_input.trt_tensor_->getDimensions().nbDims == 1);
  if (need_expand == true) {
    topk_input.trt_tensor_ = ExpandDim(ctx, topk_input.trt_tensor_, 0);
    axis_ = INPUT_SIZE2;
  }

  nvinfer1::ITopKLayer *topk_layer;
  if (topk_input.trt_tensor_->getType() == nvinfer1::DataType::kINT32) {
    MS_LOG(INFO) << "trt op topk not support INT32 as input, cast to float.";
    auto cast_layer = ctx->network()->addIdentity(*topk_input.trt_tensor_);
    CHECK_NULL_RETURN(cast_layer);
    cast_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
    auto cast_output = cast_layer->getOutput(0);
    CHECK_NULL_RETURN(cast_output);
    cast_layer->setName((op_name_ + "_cast").c_str());
    topk_layer = ctx->network()->addTopK(*cast_output, topk_op_, top_k_, axis_);
  } else {
    topk_layer = ctx->network()->addTopK(*topk_input.trt_tensor_, topk_op_, top_k_, axis_);
  }

  CHECK_NULL_RETURN(topk_layer);
  this->layer_ = topk_layer;
  topk_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *value_out_tensor = topk_layer->getOutput(0);
  nvinfer1::ITensor *index_out_tensor = topk_layer->getOutput(1);
  if (need_expand == true) {
    auto shape = ConvertMSShape(value_out_tensor->getDimensions());
    shape.erase(shape.begin());
    value_out_tensor = Reshape(ctx, value_out_tensor, shape);
    index_out_tensor = Reshape(ctx, index_out_tensor, shape);
  }
  // output 0 is data value, output 1 is index

  if (top_k_ == 1 && type_ != ops::kNameTopKFusion && keep_dims_ == false) {
    value_out_tensor = TopkReshape(ctx, value_out_tensor, axis_value_);
    if (value_out_tensor == nullptr) {
      MS_LOG(ERROR) << "add output squeeze failed!";
      return RET_ERROR;
    }
    index_out_tensor = TopkReshape(ctx, index_out_tensor, axis_value_);
    if (index_out_tensor == nullptr) {
      MS_LOG(ERROR) << "add output squeeze failed!";
      return RET_ERROR;
    }
  }
  if (out_tensors_.size() == INPUT_SIZE2) {
    auto out_tensor = (out_tensors_[1].DataType() == DataType::kNumberTypeInt32) ? index_out_tensor : value_out_tensor;
    auto output_helper = ITensorHelper{out_tensor, topk_input.format_, true};
    ctx->RegisterTensor(output_helper, out_tensors_[1].Name());
  }
  auto out_tensor = (out_tensors_[0].DataType() == DataType::kNumberTypeInt32) ? index_out_tensor : value_out_tensor;
  auto output_helper = ITensorHelper{out_tensor, topk_input.format_, true};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  return RET_OK;
}

int TopKTensorRT::ParseParams(TensorRTContext *ctx) {
  int input_nbDims = input(ctx, 0).trt_tensor_->getDimensions().nbDims;
  if (type_ == ops::kNameArgMinFusion || type_ == ops::kNameArgMaxFusion) {
    std::unordered_map<std::string, nvinfer1::TopKOperation> type2op = {
      {ops::kNameArgMaxFusion, nvinfer1::TopKOperation::kMAX}, {ops::kNameArgMinFusion, nvinfer1::TopKOperation::kMIN}};
    topk_op_ = type2op[type_];
    auto prim = AsOps<ops::ArgMaxFusion>();
    CHECK_NULL_RETURN(prim);
    axis_value_ = prim->get_axis();
    axis_value_ = axis_value_ >= 0 ? axis_value_ : input_nbDims + axis_value_;
    if (prim->HasAttr(ops::kKeepDims)) {
      keep_dims_ = prim->get_keep_dims();
    }
    top_k_ = prim->HasAttr(ops::kTopK) ? prim->get_top_k() : 1;
  }
  if (type_ == ops::kNameTopKFusion) {
    auto topk_prim = AsOps<ops::TopKFusion>();
    CHECK_NULL_RETURN(topk_prim);
    if (topk_prim->HasAttr(ops::kLargest)) {
      topk_op_ = topk_prim->get_largest() == 1 ? nvinfer1::TopKOperation::kMAX : nvinfer1::TopKOperation::kMIN;
    } else {
      MS_LOG(INFO) << "No attribute Largest for TopKFusion, use Default: MAX";
      topk_op_ = nvinfer1::TopKOperation::kMAX;
    }

    if (topk_prim->HasAttr(ops::kAxis)) {
      axis_value_ = topk_prim->get_axis();
    } else {
      MS_LOG(INFO) << "No attribute Axis for TopKFusion, use Default: input dims - 1";
      axis_value_ = input_nbDims - 1;
    }
    axis_value_ = axis_value_ >= 0 ? axis_value_ : input_nbDims + axis_value_;
    std::vector<float> tmp(1);
    int ret_k = ParseData2Vector(in_tensors_[1], &tmp);
    if (ret_k != RET_OK) {
      return ret_k;
    }
    top_k_ = tmp[0];
  }
  if (type_ == ops::kNameTopK) {
    auto topk_prim = AsOps<ops::TopK>();
    CHECK_NULL_RETURN(topk_prim);
    topk_op_ = nvinfer1::TopKOperation::kMAX;

    axis_value_ = input_nbDims - 1;
    std::vector<float> tmp(1);
    int ret_k = ParseData2Vector(in_tensors_[1], &tmp);
    if (ret_k != RET_OK) {
      return ret_k;
    }
    top_k_ = tmp[0];
  }
  // Currently reduceAxes must specify exactly one dimension, and it must be one of the last four dimensions.
  if (axis_value_ != input_nbDims - 1) {
    MS_LOG(ERROR) << op_name_ << " has unsupported axis : " << axis_value_;
    return RET_ERROR;
  }
  return RET_OK;
}

int TopKTensorRT::PreprocessInputs(TensorRTContext *ctx, ITensorHelper *topk_input) {
  auto input_dim = input(ctx, 0).trt_tensor_->getDimensions();
  int ret = RET_OK;
  if (input_dim.nbDims == DIMENSION_4D) {
    ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), topk_input);
  } else {
    *topk_input = input(ctx, 0);
  }
  return ret;
}
REGISTER_TENSORRT_CREATOR(ops::kNameArgMaxFusion, TopKTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameArgMinFusion, TopKTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameTopKFusion, TopKTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameTopK, TopKTensorRT)
}  // namespace mindspore::lite
