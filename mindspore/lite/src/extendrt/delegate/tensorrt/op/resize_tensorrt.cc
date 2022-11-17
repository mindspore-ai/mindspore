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

#include "src/extendrt/delegate/tensorrt/op/resize_tensorrt.h"
#include <vector>
#include <algorithm>
#include <memory>
#include "nnacl/nnacl_common.h"
#include "ops/resize.h"

namespace mindspore::lite {
int ResizeTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                              const std::vector<TensorInfo> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1 && in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
  }
  resize_op_ = AsOps<ops::Resize>();
  if (resize_op_ == nullptr) {
    MS_LOG(ERROR) << "convert failed " << op_name_;
    return RET_ERROR;
  }
  if (resize_op_->get_coordinate_transform_mode() == CoordinateTransformMode::ALIGN_CORNERS &&
      resize_op_->get_method() == ResizeMethod::LINEAR) {
    MS_LOG(ERROR) << "Resize op do not support coordinate_transform_mode == ALIGN_CORNERS when method == LINEAR "
                  << op_name_;
    return RET_ERROR;
  }
  dynamic_shape_params_.support_hw_dynamic_ =
    (resize_op_->get_new_height() > 0 && resize_op_->get_new_width() > 0) ? false : true;
  dynamic_shape_params_.support_hw_dynamic_ &= resize_op_->get_method() != ResizeMethod::LINEAR;

  return RET_OK;
}

int ResizeTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }

  nvinfer1::ITensor *resize_in_tensor = input(ctx, 0).trt_tensor_;
  MS_LOG(DEBUG) << "origin input " << GetTensorFormat(input(ctx, 0));

  nvinfer1::ITensor *output_tensor = RunTensorRT(ctx, resize_in_tensor);
  if (output_tensor == nullptr) {
    return RET_ERROR;
  }
  auto output_helper = ITensorHelper{output_tensor, Format::NCHW, true};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(output_helper);
  return RET_OK;
}

nvinfer1::ITensor *ResizeTensorRT::RunTensorRT(TensorRTContext *ctx, nvinfer1::ITensor *resize_in_tensor) {
  nvinfer1::IResizeLayer *resize_layer = ctx->network()->addResize(*resize_in_tensor);
  if (resize_layer == nullptr) {
    MS_LOG(ERROR) << "create resize layer failed for " << op_name_;
    return nullptr;
  }
  int ret = SetOutputDims(ctx, resize_in_tensor, resize_layer);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetOutputDims failed for " << op_name_;
    return nullptr;
  }

  ret = SetParams(resize_layer);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetParams failed for " << op_name_;
    return nullptr;
  }
  this->layer_ = resize_layer;
  return resize_layer->getOutput(0);
}

int ResizeTensorRT::SetOutputDims(TensorRTContext *ctx, nvinfer1::ITensor *resize_in_tensor,
                                  nvinfer1::IResizeLayer *resize_layer) {
  nvinfer1::Dims in_dims = resize_in_tensor->getDimensions();
  if (in_tensors_.size() == 1 && in_dims.nbDims == DIMENSION_4D) {
    nvinfer1::Dims4 new_dims(in_dims.d[0], in_dims.d[1], resize_op_->get_new_height(),
                             resize_op_->get_new_width());  // nchw
    resize_layer->setOutputDimensions(new_dims);            // static shape
  } else if (resize_op_->HasAttr(kAttrScales)) {
    auto scales = resize_op_->GetAttr(kAttrScales);
    if (!scales) {
      return RET_ERROR;
    }
    auto scales_val = GetValue<std::vector<float>>(scales);
    if (SizeToInt(scales_val.size()) != in_dims.nbDims) {
      MS_LOG(ERROR) << "Size " << scales_val.size() << " of scales get from attr != input dims count " << in_dims.nbDims
                    << ", op: " << op_name_;
      return RET_ERROR;
    }
    resize_layer->setScales(scales_val.data(), scales_val.size());
  } else {
    auto shape_value_tensor = in_tensors_[1];
    if (!shape_value_tensor.IsConst() && in_tensors_.size() >= INPUT_SIZE2) {
      // dynamic output shape
      auto shape_tensor = input(ctx, 1).trt_tensor_;
      if (shape_tensor->getDimensions().d[0] == INPUT_SIZE4) {
        resize_layer->setInput(1, *shape_tensor);
      } else {
        auto in_tensor_shape = ctx->network()->addShape(*resize_in_tensor)->getOutput(0);
        CHECK_NULL_RETURN(in_tensor_shape);
        nvinfer1::Dims start_dims{1, {0}};
        nvinfer1::Dims size_dims{1, {2}};
        nvinfer1::Dims stride_dims{1, {1}};
        auto nc = ctx->network()->addSlice(*in_tensor_shape, start_dims, size_dims, stride_dims)->getOutput(0);
        CHECK_NULL_RETURN(nc);

        nvinfer1::ITensor *trt_input_tensors[INPUT_SIZE2];
        trt_input_tensors[0] = nc;
        trt_input_tensors[1] = shape_tensor;

        auto concat_layer = ctx->network()->addConcatenation(trt_input_tensors, INPUT_SIZE2);
        concat_layer->setAxis(0);
        auto nchw = concat_layer->getOutput(0);
        CHECK_NULL_RETURN(nchw);
        nchw = TRTTensorCast(ctx, nchw, nvinfer1::DataType::kINT32, op_name_ + "_input_nchw_to_int32");
        resize_layer->setInput(1, *nchw);
      }
    } else {
      std::vector<float> out_shape;
      ParseValueFromShapeTensor(ctx, shape_value_tensor, &out_shape);
      if (out_shape.size() == DIMENSION_2D && in_dims.nbDims == DIMENSION_4D) {
        // out_shape: origin_n, out_shape[0], out_shape[1], origin_c
        out_shape.insert(out_shape.begin(), in_dims.d[0]);            // batch size is dynamic
        out_shape.insert(out_shape.begin() + 1, in_dims.d[kNCHW_C]);  // channel is const
      }
      if (shape_value_tensor.DataType() == DataType::kNumberTypeInt32) {
        if (resize_in_tensor->getDimensions().d[0] == -1) {
          nvinfer1::IShapeLayer *shape_layer = ctx->network()->addShape(*resize_in_tensor);
          auto in_shape = shape_layer->getOutput(0);
          mask2_[2] = out_shape[kNCHW_H];
          mask2_[3] = out_shape[kNCHW_W];
          auto mask1 = ConvertConstantTensor1D(ctx, mask1_, nvinfer1::DataType::kINT32);
          auto mask2 = ConvertConstantTensor1D(ctx, mask2_, nvinfer1::DataType::kINT32);
          in_shape =
            ctx->network()->addElementWise(*in_shape, *mask1, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
          in_shape =
            ctx->network()->addElementWise(*in_shape, *mask2, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
          resize_layer->setInput(1, *in_shape);
        } else {
          nvinfer1::Dims dims;
          dims.nbDims = DIMENSION_4D;
          dims.d[0] = out_shape[0];
          dims.d[1] = out_shape[1];
          dims.d[2] = out_shape[2];
          dims.d[3] = out_shape[3];
          resize_layer->setOutputDimensions(dims);
        }
      } else {
        float scales[DIMENSION_4D]{1, 1, 1, 1};
        scales[kNCHW_H] = out_shape[kNCHW_H];
        scales[kNCHW_W] = out_shape[kNCHW_W];
        resize_layer->setScales(scales, DIMENSION_4D);
      }
    }
  }
  return RET_OK;
}

void ResizeTensorRT::ParseValueFromShapeTensor(TensorRTContext *ctx, const TensorInfo &shape_value_tensor,
                                               std::vector<float> *out_shape) {
  switch (shape_value_tensor.DataType()) {
    case DataType::kNumberTypeFloat32: {
      const float *shape_data_fp32 = static_cast<const float *>(shape_value_tensor.Data());
      for (int64_t i = 0; i < shape_value_tensor.ElementNum(); i++) {
        out_shape->push_back(*(shape_data_fp32 + i));
      }
      break;
    }
    case DataType::kNumberTypeFloat16: {
      const uint16_t *shape_data_fp16 = static_cast<const uint16_t *>(shape_value_tensor.Data());
      for (int64_t i = 0; i < shape_value_tensor.ElementNum(); i++) {
        out_shape->push_back(ShortToFloat32(*(shape_data_fp16 + i)));
      }
      break;
    }
    case DataType::kNumberTypeInt32: {
      const int *shape_data_int32 = static_cast<const int *>(shape_value_tensor.Data());
      for (int64_t i = 0; i < shape_value_tensor.ElementNum(); i++) {
        out_shape->push_back(*(shape_data_int32 + i));
      }
      break;
    }
    case DataType::kNumberTypeInt64: {
      auto shape_data_int = static_cast<const int64_t *>(shape_value_tensor.Data());
      for (int64_t i = 0; i < shape_value_tensor.ElementNum(); i++) {
        out_shape->push_back(LongToFloat(shape_data_int[i]));
      }
      break;
    }
    default:
      MS_LOG(WARNING) << op_name_
                      << " more datatype need to check: " << static_cast<int>(shape_value_tensor.DataType());
      break;
  }
}

int ResizeTensorRT::SetParams(nvinfer1::IResizeLayer *resize_layer) {
  auto method = resize_op_->get_method();
  std::map<ResizeMethod, nvinfer1::ResizeMode> method_map = {{ResizeMethod::LINEAR, nvinfer1::ResizeMode::kLINEAR},
                                                             {ResizeMethod::NEAREST, nvinfer1::ResizeMode::kNEAREST}};
  if (method_map.find(method) == method_map.end()) {
    MS_LOG(ERROR) << op_name_ << " unsupported resize mode " << static_cast<int>(method);
    return RET_ERROR;
  }
  resize_layer->setResizeMode(method_map.at(method));

  auto coordinate_transform_mode = resize_op_->get_coordinate_transform_mode();
// unsupported for trt6, but support setCoordinateTransformation() in version8
#if TRT_VERSION_GE(8, 0)
  std::map<CoordinateTransformMode, nvinfer1::ResizeCoordinateTransformation> transform_map = {
    {CoordinateTransformMode::ASYMMETRIC, nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC},
    {CoordinateTransformMode::ALIGN_CORNERS, nvinfer1::ResizeCoordinateTransformation::kALIGN_CORNERS},
    {CoordinateTransformMode::HALF_PIXEL, nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL}};
  auto transform_it = transform_map.find(coordinate_transform_mode);
  if (transform_it == transform_map.end()) {
    MS_LOG(ERROR) << op_name_ << " not support resize coordinate transform mode " << coordinate_transform_mode;
    return RET_ERROR;
  }
  resize_layer->setCoordinateTransformation(transform_it->second);
#else
  if (coordinate_transform_mode != CoordinateTransformMode::ASYMMETRIC) {
    MS_LOG(WARNING) << op_name_ << " has coordinate_transform_mode may not supported: "
                    << static_cast<int>(coordinate_transform_mode);
  }
#endif
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameResize, ResizeTensorRT)
}  // namespace mindspore::lite
