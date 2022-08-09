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
#include "src/extendrt/delegate/tensorrt/op/resize_tensorrt.h"
#include "nnacl/nnacl_common.h"
#include "resize_bilinear_impl.cuh"
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
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
}

int ResizeTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }

  nvinfer1::ITensor *resize_in_tensor = input(ctx, 0).trt_tensor_;
  MS_LOG(DEBUG) << "origin input " << GetTensorFormat(input(ctx, 0));

  MS_LOG(DEBUG) << "after transpose input " << GetTensorFormat(resize_in_tensor, Format::NCHW, true);

  auto method = resize_op_->get_method();
  nvinfer1::ITensor *output_tensor = nullptr;
  if (method == ResizeMethod::LINEAR) {
    MS_LOG(INFO) << "using plugin for resize";
    output_tensor = RunPlugin(ctx, resize_in_tensor);
  } else {
    output_tensor = RunTensorRT(ctx, resize_in_tensor);
  }
  if (output_tensor == nullptr) {
    return RET_ERROR;
  }
  auto output_helper = ITensorHelper{output_tensor, Format::NCHW, true};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(output_helper);
  return RET_OK;
}

nvinfer1::ITensor *ResizeTensorRT::RunPlugin(TensorRTContext *ctx, nvinfer1::ITensor *resize_in_tensor) {
  constexpr int non_const_resize_input_count = 2;
  if (ReadyInputsNumber(ctx) == non_const_resize_input_count) {
    return nullptr;
  } else {
    const int *resize_ptr = reinterpret_cast<const int *>(in_tensors_[1].Data());
    std::vector<int> resize_shape;
    for (int i = 0; i != in_tensors_[1].ElementNum(); ++i) {
      resize_shape.push_back(*(resize_ptr + i));
    }
    constexpr int resize_hw_dims_count = 2;
    if (resize_shape.size() != resize_hw_dims_count) {
      MS_LOG(ERROR) << "Do not support resize number more than 2";
      return nullptr;
    }
    auto plugin =
      std::make_shared<ResizeLinear2DPlugin>(resize_in_tensor->getName(), resize_shape[0], resize_shape[1], device_id_);
    nvinfer1::ITensor *inputTensors[] = {resize_in_tensor};
    nvinfer1::IPluginV2Layer *resize_layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
    if (resize_layer == nullptr) {
      MS_LOG(ERROR) << "add spacetobatch op failed for TensorRT.";
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
  auto in_dims = resize_in_tensor->getDimensions();
  if (in_tensors_.size() == 1 && in_dims.nbDims == DIMENSION_4D) {
    // hw is static, but has dynamic batch size
    float scales[DIMENSION_4D]{1, 1, 1, 1};
    scales[kNCHW_H] = static_cast<float>(resize_op_->get_new_height()) / static_cast<float>(in_dims.d[kNCHW_H]);
    scales[kNCHW_W] = static_cast<float>(resize_op_->get_new_width()) / static_cast<float>(in_dims.d[kNCHW_W]);
    resize_layer->setScales(scales, DIMENSION_4D);
  } else {
    auto shape_value_tensor = in_tensors_[1];
    if (!shape_value_tensor.IsConst() && in_tensors_.size() >= INPUT_SIZE2) {
      // dynamic output shape
      resize_layer->setInput(1, *input(ctx, 1).trt_tensor_);
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
        MS_LOG(WARNING) << scales[0] << " " << scales[1] << " " << scales[2] << " " << scales[3];
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
      for (int i = 0; i < shape_value_tensor.ElementNum(); i++) {
        out_shape->push_back(*(shape_data_fp32 + i));
      }
      break;
    }
    case DataType::kNumberTypeFloat16: {
      const uint16_t *shape_data_fp16 = static_cast<const uint16_t *>(shape_value_tensor.Data());
      for (int i = 0; i < shape_value_tensor.ElementNum(); i++) {
        out_shape->push_back(ShortToFloat32(*(shape_data_fp16 + i)));
      }
      break;
    }
    case DataType::kNumberTypeInt32: {
      const int *shape_data_int32 = static_cast<const int *>(shape_value_tensor.Data());
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
  auto method = resize_op_->get_method();
  std::map<ResizeMethod, nvinfer1::ResizeMode> method_map = {{ResizeMethod::LINEAR, nvinfer1::ResizeMode::kLINEAR},
                                                             {ResizeMethod::NEAREST, nvinfer1::ResizeMode::kNEAREST}};
  if (method_map.find(method) == method_map.end()) {
    MS_LOG(ERROR) << op_name_ << " unsupported resize mode " << static_cast<int>(method);
    return RET_ERROR;
  }
  resize_layer->setResizeMode(method_map.at(method));

  // unsupported for trt6, but support setCoordinateTransformation() in version8
  auto coordinate_transform_mode = resize_op_->get_coordinate_transform_mode();
  if (coordinate_transform_mode != CoordinateTransformMode::ASYMMETRIC) {
    MS_LOG(WARNING) << op_name_ << " has coordinate_transform_mode may not supported: "
                    << static_cast<int>(coordinate_transform_mode);
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
                    resize_h_, resize_w_, h_scale, w_scale, false, static_cast<float *>(outputs[0]), device_id_,
                    stream);
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

size_t ResizeLinear2DPlugin::getSerializationSize() const noexcept { return sizeof(int) * 2; }

nvinfer1::DimsExprs ResizeLinear2DPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs,
                                                              int nbInputDims,
                                                              nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  dims.nbDims = inputs[0].nbDims;
  dims.d[0] = inputs[0].d[0];
  dims.d[1] = inputs[0].d[1];
  auto nh = exprBuilder.constant(resize_h_);
  dims.d[2] = nh;
  auto nw = exprBuilder.constant(resize_w_);
  dims.d[3] = nw;
  return dims;
}

void ResizeLinear2DPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &resize_h_, sizeof(int));
  SerializeValue(&buffer, &resize_w_, sizeof(int));
}
REGISTER_TENSORRT_CREATOR(ops::kNameResize, ResizeTensorRT)
}  // namespace mindspore::lite
