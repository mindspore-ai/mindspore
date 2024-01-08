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

#include "src/extendrt/delegate/tensorrt/op/resize_bilinearv2_tensorrt.h"
#include <vector>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include "ops/resize_bilinear_v2.h"

namespace mindspore::lite {
int ResizeBilinearV2TensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                        const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
  }
  resize_op_ = AsOps<ops::ResizeBilinearV2>();
  if (resize_op_ == nullptr) {
    MS_LOG(ERROR) << "convert failed " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeBilinearV2TensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }

  nvinfer1::ITensor *resize_in_tensor = input(ctx, 0).trt_tensor_;
  MS_LOG(DEBUG) << "origin input " << GetTensorFormat(input(ctx, 0));

  auto shape_value_tensor = in_tensors_[1];
  std::vector<float> out_shape;
  ParseValueFromShapeTensor(ctx, shape_value_tensor, &out_shape);
  auto in_dims = resize_in_tensor->getDimensions();
  if (out_shape.size() == DIMENSION_2D && in_dims.nbDims == DIMENSION_4D) {
    // out_shape: origin_n, out_shape[0], out_shape[1], origin_c
    out_shape.insert(out_shape.begin(), in_dims.d[0]);            // batch size is dynamic
    out_shape.insert(out_shape.begin() + 1, in_dims.d[kNCHW_C]);  // channel is const
  }
  nvinfer1::Dims dims;
  dims.nbDims = DIMENSION_4D;
  dims.d[0] = out_shape[0];
  dims.d[1] = out_shape[1];
  dims.d[2] = out_shape[2];
  dims.d[3] = out_shape[3];
  nvinfer1::IResizeLayer *resize_layer = ctx->network()->addResize(*resize_in_tensor);
  if (resize_layer == nullptr) {
    MS_LOG(ERROR) << "create resize layer failed for " << op_name_;
    return RET_ERROR;
  }
  resize_layer->setOutputDimensions(dims);
  resize_layer->setResizeMode(nvinfer1::ResizeMode::kLINEAR);
  if (resize_op_->get_align_corners()) {
    resize_layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kALIGN_CORNERS);
  } else if (resize_op_->get_half_pixel_centers()) {
    resize_layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL);
  } else {
    resize_layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC);
  }

  nvinfer1::ITensor *output_tensor = resize_layer->getOutput(0);
  if (output_tensor == nullptr) {
    return RET_ERROR;
  }
  this->layer_ = resize_layer;
  auto output_helper = ITensorHelper{output_tensor, Format::NCHW, true};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(output_helper);
  return RET_OK;
}

REGISTER_TENSORRT_CREATOR(ops::kNameResizeBilinearV2, ResizeBilinearV2TensorRT)
}  // namespace mindspore::lite
