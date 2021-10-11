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
  return RET_OK;
}

int ResizeTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }

  nvinfer1::ITensor *resize_in_tensor = tensorrt_in_tensors_[0].trt_tensor_;
  MS_LOG(DEBUG) << "origin input " << GetTensorFormat(resize_in_tensor, tensorrt_in_tensors_[0].format_);

  if (resize_in_tensor->getDimensions().nbDims == DIMENSION_4D && tensorrt_in_tensors_[0].format_ == Format::NCHW) {
    // NCHW->NHWC
    nvinfer1::IShuffleLayer *transpose_layer = NCHW2NHWC(network, *tensorrt_in_tensors_[0].trt_tensor_);
    if (transpose_layer == nullptr) {
      MS_LOG(ERROR) << "create transpose layer failed for " << op_name_;
      return RET_ERROR;
    }
    transpose_layer->setName((op_name_ + "_transpose_in").c_str());
    resize_in_tensor = transpose_layer->getOutput(0);
  }
  MS_LOG(DEBUG) << "after transpose input " << GetTensorFormat(resize_in_tensor, Format::NHWC);

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
  this->AddInnerOutTensors(ITensorHelper{resize_layer->getOutput(0), Format::NHWC});
  MS_LOG(DEBUG) << "output " << GetTensorFormat(resize_layer->getOutput(0), Format::NHWC);
  return RET_OK;
}

int ResizeTensorRT::SetOutputDims(nvinfer1::ITensor *resize_in_tensor, nvinfer1::IResizeLayer *resize_layer) {
  auto resize_op = op_primitive_->value_as_Resize();
  if (resize_op == nullptr) {
    MS_LOG(ERROR) << "convert failed " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors_.size() == 1) {
    nvinfer1::Dims new_dims = resize_in_tensor->getDimensions();  // nhwc
    new_dims.d[1] = resize_op->new_height();
    new_dims.d[2] = resize_op->new_width();
    resize_layer->setOutputDimensions(new_dims);
  } else {
    std::vector<float> out_shape;
    const void *shape_data = in_tensors_[1].Data().get();
    if (shape_data == nullptr) {
      // dynamic output shape
      if (tensorrt_in_tensors_.size() < INPUT_SIZE2) {
        MS_LOG(ERROR) << "no output shape tensor found for " << op_name_;
        return RET_ERROR;
      }
      resize_layer->setInput(1, *tensorrt_in_tensors_[1].trt_tensor_);
    } else {
      if (in_tensors_[1].ElementNum() != resize_in_tensor->getDimensions().nbDims) {
        MS_LOG(ERROR) << "output shape tensor value is invalid for " << op_name_;
        return RET_ERROR;
      }
      switch (in_tensors_[1].DataType()) {
        case DataType::kNumberTypeFloat32: {
          const float *shape_data_fp32 = static_cast<const float *>(shape_data);
          for (int i = 0; i < in_tensors_[1].ElementNum(); i++) {
            out_shape.push_back(*(shape_data_fp32 + i));
          }
          break;
        }
        case DataType::kNumberTypeFloat16: {
          const uint16_t *shape_data_fp16 = static_cast<const uint16_t *>(shape_data);
          for (int i = 0; i < in_tensors_[1].ElementNum(); i++) {
            out_shape.push_back(ShortToFloat32(*(shape_data_fp16 + i)));
          }
          break;
        }
        default:
          MS_LOG(WARNING) << op_name_
                          << " more datatype need to check: " << static_cast<int>(in_tensors_[1].DataType());
          break;
      }
      if (SameDims(out_shape, out_tensors_[0].Shape())) {
        // static dims
        resize_layer->setOutputDimensions(ConvertCudaDims(out_shape));
      } else if (IsScaleOutputDim(in_tensors_[0].Shape(), out_tensors_[0].Shape(), out_shape)) {
        // scale dims
        if (out_shape.size() > DIMENSION_4D) {
          MS_LOG(ERROR) << "dims count needs check for " << op_name_;
          return RET_ERROR;
        }
        float scales[DIMENSION_4D];
        std::copy(out_shape.begin(), out_shape.end(), scales);
        resize_layer->setScales(scales, out_shape.size());
      } else {
        MS_LOG(ERROR) << "output shape tensor value is invalid for " << op_name_;
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
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
  auto resize_op = op_primitive_->value_as_Resize();
  if (resize_op == nullptr) {
    MS_LOG(ERROR) << "convert failed " << op_name_;
    return RET_ERROR;
  }

  auto method = resize_op->method();
  std::map<schema::ResizeMethod, nvinfer1::ResizeMode> method_map = {
    {schema::ResizeMethod_LINEAR, nvinfer1::ResizeMode::kLINEAR},
    {schema::ResizeMethod_NEAREST, nvinfer1::ResizeMode::kNEAREST}};
  if (method_map.find(method) == method_map.end()) {
    MS_LOG(ERROR) << op_name_ << " unsupported resize mode " << EnumNameResizeMethod(method);
    return RET_ERROR;
  }
  resize_layer->setResizeMode(method_map.at(method));

  // unsupported for trt6, but support in higher version
  auto coordinate_transform_mode = resize_op->coordinate_transform_mode();
  if (coordinate_transform_mode != schema::CoordinateTransformMode_ASYMMETRIC) {
    MS_LOG(WARNING) << op_name_ << " has coordinate_transform_mode not supported: "
                    << EnumNameCoordinateTransformMode(coordinate_transform_mode);
  }
  return RET_OK;
}
}  // namespace mindspore::lite
