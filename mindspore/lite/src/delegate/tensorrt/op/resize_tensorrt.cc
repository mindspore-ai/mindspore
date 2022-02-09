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

#include <algorithm>
#include "src/delegate/tensorrt/op/resize_tensorrt.h"
#include "nnacl/nnacl_common.h"

namespace mindspore::lite {
int ResizeTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
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
  resize_op_ = op_primitive_->value_as_Resize();
  if (resize_op_ == nullptr) {
    MS_LOG(ERROR) << "convert failed " << op_name_;
    return RET_ERROR;
  }
  if (resize_op_->method() == schema::ResizeMethod_LINEAR) {
    MS_LOG(WARNING) << "TensorRT linear resize has precision issue, using cpu instead for " << op_name_;
    return RET_ERROR;
  }
  dynamic_shape_params_.support_hw_dynamic_ =
    (resize_op_->new_height() > 0 && resize_op_->new_width() > 0) ? false : true;
  // constant new hw op don't support hw resize
  return RET_OK;
}

int ResizeTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }

  nvinfer1::ITensor *resize_in_tensor = tensorrt_in_tensors_[0].trt_tensor_;
  MS_LOG(DEBUG) << "origin input " << GetTensorFormat(tensorrt_in_tensors_[0]);

  if (resize_in_tensor->getDimensions().nbDims == DIMENSION_4D && tensorrt_in_tensors_[0].format_ == Format::NHWC) {
    // NHWC->NCHW
    nvinfer1::IShuffleLayer *transpose_layer = NHWC2NCHW(network, *tensorrt_in_tensors_[0].trt_tensor_);
    if (transpose_layer == nullptr) {
      MS_LOG(ERROR) << "create transpose layer failed for " << op_name_;
      return RET_ERROR;
    }
    transpose_layer->setName((op_name_ + "_transpose_in").c_str());
    resize_in_tensor = transpose_layer->getOutput(0);
    this->transpose_layer_ = transpose_layer;
  }
  MS_LOG(DEBUG) << "after transpose input " << GetTensorFormat(resize_in_tensor, Format::NCHW, false);

  nvinfer1::IResizeLayer *resize_layer = network->addResize(*resize_in_tensor);
  if (resize_layer == nullptr) {
    MS_LOG(ERROR) << "create resize layer failed for " << op_name_;
    return RET_ERROR;
  }
  int ret = SetOutputDims(resize_in_tensor, resize_layer);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetOutputDims failed for " << op_name_;
    return RET_ERROR;
  }

  ret = SetParams(resize_layer);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetParams failed for " << op_name_;
    return RET_ERROR;
  }

  resize_layer->getOutput(0)->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{resize_layer->getOutput(0), Format::NCHW, false});
  MS_LOG(DEBUG) << "output " << GetTensorFormat(tensorrt_out_tensors_[0]);
  this->layer_ = resize_layer;
  return RET_OK;
}

int ResizeTensorRT::SetOutputDims(nvinfer1::ITensor *resize_in_tensor, nvinfer1::IResizeLayer *resize_layer) {
  if (in_tensors_.size() == 1 && !dynamic_shape_params_.support_dynamic_) {
    nvinfer1::Dims new_dims = resize_in_tensor->getDimensions();  // nchw
    if (new_dims.nbDims != DIMENSION_4D) {
      MS_LOG(ERROR) << op_name_ << " resize has new height and width value, but input dim is " << new_dims.nbDims;
      return RET_ERROR;
    }
    new_dims.d[kNCHW_H] = resize_op_->new_height();
    new_dims.d[kNCHW_W] = resize_op_->new_width();
    resize_layer->setOutputDimensions(new_dims);  // static shape
  } else if (in_tensors_.size() == 1 && !dynamic_shape_params_.support_hw_dynamic_ &&
             dynamic_shape_params_.support_dynamic_ && resize_in_tensor->getDimensions().nbDims == DIMENSION_4D) {
    // hw is static, but has dynamic batch size
    float scales[DIMENSION_4D]{1, 1, 1, 1};
    scales[kNCHW_H] =
      static_cast<float>(resize_op_->new_height()) / static_cast<float>(resize_in_tensor->getDimensions().d[kNCHW_H]);
    scales[kNCHW_W] =
      static_cast<float>(resize_op_->new_width()) / static_cast<float>(resize_in_tensor->getDimensions().d[kNCHW_W]);
    resize_layer->setScales(scales, DIMENSION_4D);
  } else {
    auto shape_value_tensor = in_tensors_[1];
    if (shape_value_tensor.Data() == nullptr) {
      // dynamic output shape
      if (tensorrt_in_tensors_.size() < INPUT_SIZE2) {
        MS_LOG(ERROR) << "no output shape tensor found for " << op_name_;
        return RET_ERROR;
      }
      resize_layer->setInput(1, *tensorrt_in_tensors_[1].trt_tensor_);
    } else {
      std::vector<float> out_shape;
      ParseValueFromShapeTensor(shape_value_tensor, &out_shape);
      if (SameDims(out_shape, out_tensors_[0].Shape())) {
        // static dims
        if (out_shape.size() == DIMENSION_4D) {
          // convert nhwc to nchw
          auto channel = out_shape[out_shape.size() - 1];
          out_shape.insert(out_shape.begin() + 1, channel);
          out_shape.erase(out_shape.begin() + out_shape.size() - 1);
        }
        resize_layer->setOutputDimensions(ConvertCudaDims(out_shape));
      } else if (IsScaleOutputDim(in_tensors_[0].Shape(), out_tensors_[0].Shape(), out_shape)) {
        // scale dims
        if (out_shape.size() != DIMENSION_4D) {
          MS_LOG(ERROR) << "dims count needs check for " << op_name_;
          return RET_ERROR;
        }
        float scales[DIMENSION_4D]{1, 1, 1, 1};
        scales[kNCHW_H] =
          static_cast<float>(out_tensors_[0].Shape()[kNHWC_H]) / static_cast<float>(in_tensors_[0].Shape()[kNHWC_H]);
        scales[kNCHW_W] =
          static_cast<float>(out_tensors_[0].Shape()[kNHWC_W]) / static_cast<float>(in_tensors_[0].Shape()[kNHWC_W]);
        resize_layer->setScales(scales, DIMENSION_4D);
      } else {
        MS_LOG(DEBUG) << op_name_ << " output shape tensor value is const, but set to scales for dynamic input shape.";
        float scales[out_tensors_[0].Shape().size()];
        for (size_t i = 0; i < out_tensors_[0].Shape().size(); i++) {
          scales[i] = static_cast<float>(out_tensors_[0].Shape()[i]) / static_cast<float>(in_tensors_[0].Shape()[i]);
        }
        if (out_tensors_[0].Shape().size() == DIMENSION_4D) {
          scales[kNCHW_W] = scales[kNHWC_W];
          scales[kNCHW_H] = scales[kNHWC_H];
          scales[kNCHW_C] = 1;
        }
        for (size_t i = 0; i < out_tensors_[0].Shape().size(); i++) {
          MS_LOG(DEBUG) << op_name_ << "scale at " << i << ": " << scales[i];
        }
        resize_layer->setScales(scales, out_tensors_[0].Shape().size());
      }
    }
  }
  return RET_OK;
}

void ResizeTensorRT::ParseValueFromShapeTensor(const mindspore::MSTensor &shape_value_tensor,
                                               std::vector<float> *out_shape) {
  switch (shape_value_tensor.DataType()) {
    case DataType::kNumberTypeFloat32: {
      const float *shape_data_fp32 = static_cast<const float *>(shape_value_tensor.Data().get());
      for (int i = 0; i < shape_value_tensor.ElementNum(); i++) {
        out_shape->push_back(*(shape_data_fp32 + i));
      }
      break;
    }
    case DataType::kNumberTypeFloat16: {
      const uint16_t *shape_data_fp16 = static_cast<const uint16_t *>(shape_value_tensor.Data().get());
      for (int i = 0; i < shape_value_tensor.ElementNum(); i++) {
        out_shape->push_back(ShortToFloat32(*(shape_data_fp16 + i)));
      }
      break;
    }
    case DataType::kNumberTypeInt32: {
      const int *shape_data_fp16 = static_cast<const int *>(shape_value_tensor.Data().get());
      for (int i = 0; i < shape_value_tensor.ElementNum(); i++) {
        out_shape->push_back(*(shape_data_fp16 + i));
      }
      break;
    }
    default:
      MS_LOG(WARNING) << op_name_
                      << " more datatype need to check: " << static_cast<int>(shape_value_tensor.DataType());
      break;
  }
  if (out_shape->size() == DIMENSION_2D &&
      tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D) {
    // out_shape: origin_n, out_shape[0], out_shape[1], origin_c
    out_shape->insert(out_shape->begin(),
                      tensorrt_in_tensors_[0].trt_tensor_->getDimensions().d[0]);  // batch size is dynamic
    out_shape->push_back(in_tensors_[0].Shape()[kNHWC_C]);                         // channel is const
  }
}

bool ResizeTensorRT::IsScaleOutputDim(const std::vector<int64_t> &in_shape, const std::vector<int64_t> &out_shape,
                                      const std::vector<float> &shape_tensor_val) {
  if (in_shape.size() != out_shape.size() || shape_tensor_val.size() != in_shape.size()) {
    MS_LOG(WARNING) << "tensor shape is not same for " << op_name_;
    return false;
  }
  for (size_t i = 0; i < in_shape.size(); i++) {
    if (std::abs(in_shape[i] * shape_tensor_val[i] - out_shape[i]) > 1e-6) {
      return false;
    }
  }
  return true;
}

int ResizeTensorRT::SetParams(nvinfer1::IResizeLayer *resize_layer) {
  auto method = resize_op_->method();
  std::map<schema::ResizeMethod, nvinfer1::ResizeMode> method_map = {
    {schema::ResizeMethod_LINEAR, nvinfer1::ResizeMode::kLINEAR},
    {schema::ResizeMethod_NEAREST, nvinfer1::ResizeMode::kNEAREST}};
  if (method_map.find(method) == method_map.end()) {
    MS_LOG(ERROR) << op_name_ << " unsupported resize mode " << EnumNameResizeMethod(method);
    return RET_ERROR;
  }
  resize_layer->setResizeMode(method_map.at(method));

  // unsupported for trt6, but support setCoordinateTransformation() in version8
  auto coordinate_transform_mode = resize_op_->coordinate_transform_mode();
  if (coordinate_transform_mode != schema::CoordinateTransformMode_ASYMMETRIC) {
    MS_LOG(WARNING) << op_name_ << " has coordinate_transform_mode may not supported: "
                    << EnumNameCoordinateTransformMode(coordinate_transform_mode);
  }
  return RET_OK;
}
}  // namespace mindspore::lite
