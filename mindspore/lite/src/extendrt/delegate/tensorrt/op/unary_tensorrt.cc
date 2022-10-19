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

#include "src/extendrt/delegate/tensorrt/op/unary_tensorrt.h"

namespace mindspore::lite {
int UnaryTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                             const std::vector<TensorInfo> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
  }
  auto it = unary_ops_.find(type_);
  if (it != unary_ops_.end()) {
    unary_op_ = it->second;
  } else {
    MS_LOG(ERROR) << "unsupported unary ops type: " << type_;
    return RET_ERROR;
  }
  return RET_OK;
}

int UnaryTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }
  if (type_ == ops::kNameNeg && input(ctx, 0).trt_tensor_->getType() == nvinfer1::DataType::kINT32) {
    auto trt_in_tensor = input(ctx, 0).trt_tensor_;
    size_t dims_size = mindspore::IntToSize(trt_in_tensor->getDimensions().nbDims);
    static const float neg1_const = -1;
    auto prod_input1 =
      ConvertScalarToITensor(ctx, dims_size, &neg1_const, DataType::kNumberTypeInt32, op_name_ + "_neg1");
    CHECK_NULL_RETURN(prod_input1);
    auto prod_layer =
      ctx->network()->addElementWise(*trt_in_tensor, *prod_input1, nvinfer1::ElementWiseOperation::kPROD);
    CHECK_NULL_RETURN(prod_layer);
    auto out_tensor = prod_layer->getOutput(0);
    CHECK_NULL_RETURN(out_tensor);
    ctx->RegisterTensor(ITensorHelper{out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                        out_tensors_[0].Name());
    prod_layer->setName(op_name_.c_str());
    this->layer_ = prod_layer;
    ctx->RegisterLayer(prod_layer, op_name_);
    return RET_OK;
  }
  nvinfer1::IUnaryLayer *cal_layer = ctx->network()->addUnary(*input(ctx, 0).trt_tensor_, unary_op_);
  if (cal_layer == nullptr) {
    MS_LOG(ERROR) << "addUnary failed for: " << op_name_;
    return RET_ERROR;
  }
  cal_layer->setName(op_name_.c_str());
  this->layer_ = cal_layer;
  if (type_ == ops::kNameExpFusion) {
    auto exp_op = AsOps<ops::ExpFusion>();
    CHECK_NULL_RETURN(exp_op);
    float scale = exp_op->get_scale();
    float shift = exp_op->get_shift();
    float base = exp_op->get_base();
    if (scale != 1.0f || shift != 0.0f || base != -1.0f) {
      MS_LOG(ERROR) << op_name_ << " has fusion to calculate.";
      return RET_ERROR;
    }
  }
  nvinfer1::ITensor *op_out_tensor = cal_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{op_out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameSqrt, UnaryTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameAbs, UnaryTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameNeg, UnaryTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameLog, UnaryTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameSin, UnaryTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameCos, UnaryTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameCeil, UnaryTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameFloor, UnaryTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameExpFusion, UnaryTensorRT)
#if TRT_VERSION_GE(7, 2)
REGISTER_TENSORRT_CREATOR(ops::kNameLogicalNot, UnaryTensorRT)
#endif
}  // namespace mindspore::lite
