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

#include <vector>
#include <algorithm>
#include <memory>
#include "src/litert/delegate/tensorrt/op/resize_tensorrt.h"
#include "nnacl/nnacl_common.h"
#include "resize_bilinear_impl.cuh"

namespace mindspore::lite {
namespace {
nvinfer1::ITensor *ConvertConstantTensor1D(TensorRTContext *ctx, int *weights_vec, nvinfer1::DataType data_type) {
  nvinfer1::Weights weights{data_type, weights_vec, INPUT_SIZE4};
  nvinfer1::Dims dims;
  dims.nbDims = 1;
  dims.d[0] = INPUT_SIZE4;
  nvinfer1::IConstantLayer *constant_tensor = ctx->network()->addConstant(dims, weights);
  if (constant_tensor == nullptr) {
    MS_LOG(ERROR) << "create constant_tensor failed.";
    return nullptr;
  }
  return constant_tensor->getOutput(0);
}
}  // namespace
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
  if (resize_op_->coordinate_transform_mode() == schema::CoordinateTransformMode_ALIGN_CORNERS &&
      resize_op_->method() == schema::ResizeMethod_LINEAR) {
    MS_LOG(ERROR) << "Resize op do not support coordinate_transform_mode == ALIGN_CORNERS when method == LINEAR "
                  << op_name_;
    return RET_ERROR;
  }
  dynamic_shape_params_.support_hw_dynamic_ =
    (resize_op_->new_height() > 0 && resize_op_->new_width() > 0) ? false : true;
  dynamic_shape_params_.support_hw_dynamic_ &= resize_op_->method() != schema::ResizeMethod_LINEAR;

  return RET_OK;
}

int ResizeTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }

  nvinfer1::ITensor *resize_in_tensor = input(ctx, 0).trt_tensor_;
  MS_LOG(DEBUG) << "origin input " << GetTensorFormat(input(ctx, 0));

  if (resize_in_tensor->getDimensions().nbDims == DIMENSION_4D && input(ctx, 0).format_ == Format::NHWC) {
    // NHWC->NCHW
    nvinfer1::IShuffleLayer *transpose_layer = NHWC2NCHW(ctx, *input(ctx, 0).trt_tensor_);
    if (transpose_layer == nullptr) {
      MS_LOG(ERROR) << "create transpose layer failed for " << op_name_;
      return RET_ERROR;
    }
    transpose_layer->setName((op_name_ + "_transpose_in").c_str());
    resize_in_tensor = transpose_layer->getOutput(0);
    this->transpose_layer_ = transpose_layer;
  }
  MS_LOG(DEBUG) << "after transpose input " << GetTensorFormat(resize_in_tensor, Format::NCHW, false);

  auto method = resize_op_->method();
  nvinfer1::ITensor *output_tensor = nullptr;
  if (method == schema::ResizeMethod_LINEAR) {
    MS_LOG(INFO) << "using plugin for resize";
    output_tensor = RunPlugin(ctx, resize_in_tensor);
  } else {
    output_tensor = RunTensorRT(ctx, resize_in_tensor);
  }
  if (output_tensor == nullptr) {
    return RET_ERROR;
  }
  auto output_helper = ITensorHelper{output_tensor, Format::NCHW, false};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(output_helper);
  return RET_OK;
}

nvinfer1::ITensor *ResizeTensorRT::RunPlugin(TensorRTContext *ctx, nvinfer1::ITensor *resize_in_tensor) {
  std::vector<int> resize_shape;
  if (ReadyInputsNumber(ctx) == INPUT_SIZE2) {
    MS_LOG(ERROR) << "dynamic input is not support!";
    return nullptr;
  } else {
    if (in_tensors_.size() == 1) {
      resize_shape.push_back(static_cast<float>(resize_op_->new_height()));
      resize_shape.push_back(static_cast<float>(resize_op_->new_width()));
    } else {
      const int *resize_ptr = reinterpret_cast<const int *>(in_tensors_[1].Data().get());
      for (int i = 0; i != in_tensors_[1].ElementNum(); ++i) {
        resize_shape.push_back(*(resize_ptr + i));
      }
      if (resize_shape.size() != INPUT_SIZE2) {
        MS_LOG(ERROR) << "Do not support resize number more than 2";
        return nullptr;
      }
    }
    bool using_half_pixel = (resize_op_->coordinate_transform_mode() == schema::CoordinateTransformMode_HALF_PIXEL);
    auto plugin = std::make_shared<ResizeLinear2DPlugin>(resize_in_tensor->getName(), resize_shape[0], resize_shape[1],
                                                         using_half_pixel, device_id_);
    if (plugin == nullptr) {
      MS_LOG(ERROR) << "add ResizeLinear2D plugin failed for " << op_name_;
      return nullptr;
    }
    nvinfer1::ITensor *inputTensors[] = {resize_in_tensor};
    nvinfer1::IPluginV2Layer *resize_layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
    if (resize_layer == nullptr) {
      MS_LOG(ERROR) << "add resize op failed for TensorRT.";
      return nullptr;
    }
    resize_layer->setName(op_name_.c_str());
    this->layer_ = resize_layer;
    return resize_layer->getOutput(0);
  }
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
    nvinfer1::Dims4 new_dims(in_dims.d[0], in_dims.d[1], resize_op_->new_height(), resize_op_->new_width());  // nchw
    resize_layer->setOutputDimensions(new_dims);  // static shape
  } else {
    auto shape_value_tensor = in_tensors_[1];
    if (shape_value_tensor.Data() == nullptr && in_tensors_.size() >= INPUT_SIZE2) {
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
        out_shape.insert(out_shape.begin(), in_dims.d[0]);  // batch size is dynamic
        out_shape.push_back(in_dims.d[kNCHW_C]);            // channel is const
      }
      if (shape_value_tensor.DataType() == DataType::kNumberTypeInt32) {
        if (resize_in_tensor->getDimensions().d[0] == -1) {
          nvinfer1::IShapeLayer *shape_layer = ctx->network()->addShape(*resize_in_tensor);
          auto in_shape = shape_layer->getOutput(0);
          mask2_[INPUT_SIZE2] = out_shape[kNHWC_H];
          mask2_[INPUT_SIZE3] = out_shape[kNHWC_W];
          auto mask1 = ConvertConstantTensor1D(ctx, mask1_, nvinfer1::DataType::kINT32);
          auto mask2 = ConvertConstantTensor1D(ctx, mask2_, nvinfer1::DataType::kINT32);
          in_shape =
            ctx->network()->addElementWise(*in_shape, *mask1, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
          in_shape =
            ctx->network()->addElementWise(*in_shape, *mask2, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
          resize_layer->setInput(1, *in_shape);
        } else {
          auto h = out_shape[kNHWC_H];
          auto w = out_shape[kNHWC_W];
          auto c = out_shape[kNHWC_C];
          out_shape[kNCHW_H] = h;
          out_shape[kNCHW_W] = w;
          out_shape[kNCHW_C] = c;
          nvinfer1::Dims dims;
          dims.nbDims = DIMENSION_4D;
          dims.d[0] = out_shape[0];
          dims.d[1] = out_shape[1];
          dims.d[INPUT_SIZE2] = out_shape[INPUT_SIZE2];
          dims.d[INPUT_SIZE3] = out_shape[INPUT_SIZE3];
          resize_layer->setOutputDimensions(dims);
        }
      } else {
        auto h = out_shape[kNHWC_H];
        auto w = out_shape[kNHWC_W];
        out_shape[kNCHW_H] = h;
        out_shape[kNCHW_W] = w;
        float scales[DIMENSION_4D]{1, 1, 1, 1};
        scales[kNCHW_H] = out_shape[kNCHW_H];
        scales[kNCHW_W] = out_shape[kNCHW_W];
        resize_layer->setScales(scales, DIMENSION_4D);
      }
    }
  }
  return RET_OK;
}

void ResizeTensorRT::ParseValueFromShapeTensor(TensorRTContext *ctx, const mindspore::MSTensor &shape_value_tensor,
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
      const int *shape_data_int32 = static_cast<const int *>(shape_value_tensor.Data().get());
      for (int i = 0; i < shape_value_tensor.ElementNum(); i++) {
        out_shape->push_back(*(shape_data_int32 + i));
      }
      break;
    }
    default:
      MS_LOG(WARNING) << op_name_
                      << " more datatype need to check: " << static_cast<int>(shape_value_tensor.DataType());
      break;
  }
}

bool ResizeTensorRT::IsScaleOutputDim(const std::vector<int64_t> &in_shape, const std::vector<int64_t> &out_shape,
                                      const std::vector<float> &shape_tensor_val) {
  if (out_shape.size() != DIMENSION_4D) {
    MS_LOG(WARNING) << "dims count needs check for " << op_name_;
    return false;
  }
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

REGISTER_TENSORRT_PLUGIN(ResizeLinear2DPluginCreater);
template class TensorRTPluginCreater<ResizeLinear2DPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int ResizeLinear2DPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                  const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                  void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  return RunCudaResizeLinear2D(inputDesc, inputs, outputs, stream);
}

int ResizeLinear2DPlugin::RunCudaResizeLinear2D(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                                                void *const *outputs, cudaStream_t stream) {
  auto &dims = inputDesc[0].dims;
  float h_scale = dims.d[kNCHW_H] / static_cast<float>(resize_h_);
  float w_scale = dims.d[kNCHW_W] / static_cast<float>(resize_w_);
  CalResizeBilinear(static_cast<const float *>(inputs[0]), dims.d[0], dims.d[1], dims.d[kNCHW_H], dims.d[kNCHW_W],
                    resize_h_, resize_w_, h_scale, w_scale, using_half_pixel_, static_cast<float *>(outputs[0]),
                    device_id_, stream);
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *ResizeLinear2DPlugin::clone() const noexcept {
  auto *plugin = new (std::nothrow) ResizeLinear2DPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "new plugin failed!";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t ResizeLinear2DPlugin::getSerializationSize() const noexcept { return sizeof(int) * INPUT_SIZE2 + sizeof(bool); }

nvinfer1::DimsExprs ResizeLinear2DPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs,
                                                              int nbInputDims,
                                                              nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  dims.nbDims = inputs[0].nbDims;
  dims.d[0] = inputs[0].d[0];
  dims.d[1] = inputs[0].d[1];
  auto nh = exprBuilder.constant(resize_h_);
  dims.d[INPUT_SIZE2] = nh;
  auto nw = exprBuilder.constant(resize_w_);
  dims.d[INPUT_SIZE3] = nw;
  return dims;
}

void ResizeLinear2DPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &resize_h_, sizeof(int));
  SerializeValue(&buffer, &resize_w_, sizeof(int));
  SerializeValue(&buffer, &using_half_pixel_, sizeof(bool));
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Resize, ResizeTensorRT)
}  // namespace mindspore::lite
