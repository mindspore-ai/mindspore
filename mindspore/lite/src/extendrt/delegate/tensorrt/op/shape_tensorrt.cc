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

#include "src/extendrt/delegate/tensorrt/op/shape_tensorrt.h"

namespace mindspore::lite {
int ShapeTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
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
int ShapeTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  nvinfer1::ITensor *shape_input = input(ctx, 0).trt_tensor_;
  if (input(ctx, 0).trt_tensor_->getDimensions().nbDims == DIMENSION_4D && !input(ctx, 0).same_format_) {
    if (input(ctx, 0).format_ == Format::NCHW) {
      // transpose: NCHW->NHWC
      nvinfer1::IShuffleLayer *transpose_layer_in = NCHW2NHWC(ctx, *input(ctx, 0).trt_tensor_);
      if (transpose_layer_in == nullptr) {
        MS_LOG(ERROR) << "transpose: NCHW->NHWC failed for " << op_name_;
        return RET_ERROR;
      }
      transpose_layer_in->setName((op_name_ + "_transpose2NHWC").c_str());
      shape_input = transpose_layer_in->getOutput(0);
      this->transpose_layer_ = transpose_layer_in;
    }
  }
  nvinfer1::IShapeLayer *shape_layer = ctx->network()->addShape(*shape_input);

  if (shape_layer == nullptr) {
    MS_LOG(ERROR) << "add shape op failed for TensorRT.";
    return RET_ERROR;
  }
  shape_layer->setName(op_name_.c_str());
  ctx->RegisterTensor(ITensorHelper{shape_layer->getOutput(0), Format::NHWC, true}, out_tensors_[0].Name());
  this->layer_ = shape_layer;
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Shape, ShapeTensorRT)
}  // namespace mindspore::lite
