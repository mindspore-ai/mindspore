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

#include "src/delegate/tensorrt/op/topk_tensorrt.h"

namespace mindspore::lite {
int TopKTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors) {
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
  return RET_OK;
}

int TopKTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr || this->tensorrt_in_tensors_.size() != 1) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }
  nvinfer1::TopKOperation red_op = nvinfer1::TopKOperation::kMAX;
  int axis_value = 0;
  int topk = 0;
  bool keep_dims = false;
  if (type_ == schema::PrimitiveType_ArgMaxFusion) {
    red_op = nvinfer1::TopKOperation::kMAX;
    auto max_prim = op_primitive_->value_as_ArgMaxFusion();
    if (max_prim == nullptr) {
      MS_LOG(ERROR) << "convert ArgMaxFusion failed: " << op_name_;
      return RET_ERROR;
    }
    axis_value = max_prim->axis();
    topk = max_prim->top_k();
    keep_dims = max_prim->keep_dims();
  } else if (type_ == schema::PrimitiveType_ArgMinFusion) {
    red_op = nvinfer1::TopKOperation::kMIN;
    auto mim_prim = op_primitive_->value_as_ArgMinFusion();
    if (mim_prim == nullptr) {
      MS_LOG(ERROR) << "convert ArgMinFusion failed: " << op_name_;
      return RET_ERROR;
    }
    axis_value = mim_prim->axis();
    topk = mim_prim->top_k();
    keep_dims = mim_prim->keep_dims();
  } else {
    MS_LOG(ERROR) << "invalid op primitive for " << op_name_;
  }
  if (keep_dims) {
    MS_LOG(WARNING) << "keep dims is unsupported for " << op_name_;
  }

  if (tensorrt_in_tensors_[0].format_ == Format::NCHW) {
    axis_value = ConvertAxisFromNHWC2NCHW(axis_value);
  }
  uint32_t reduce_axes = 1 << axis_value;

  nvinfer1::ITopKLayer *topk_layer = network->addTopK(*tensorrt_in_tensors_[0].trt_tensor_, red_op, topk, reduce_axes);
  if (topk_layer == nullptr) {
    MS_LOG(ERROR) << "addTopK failed for: " << op_name_;
    return RET_ERROR;
  }
  topk_layer->setName(op_name_.c_str());

  nvinfer1::ITensor *op_out_tensor = topk_layer->getOutput(0);
  op_out_tensor->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{op_out_tensor, tensorrt_in_tensors_[0].format_});
  return RET_OK;
}
}  // namespace mindspore::lite
