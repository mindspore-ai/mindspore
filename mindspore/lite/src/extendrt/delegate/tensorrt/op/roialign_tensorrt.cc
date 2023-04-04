/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "src/extendrt/delegate/tensorrt/op/roialign_tensorrt.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/roi_align_impl.cuh"
#include "ops/roi_align.h"

namespace mindspore::lite {
int ROIAlignTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size for ROIAlignTensorRT, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size for ROIAlignTensorRT, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int ROIAlignTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }
  auto op = AsOps<ops::ROIAlign>();
  int pooled_height = op->get_pooled_height();
  int pooled_width = op->get_pooled_width();
  float spatial_scale = op->get_spatial_scale();
  int sample_num = op->get_sample_num();
  int roi_end_mode = op->get_roi_end_mode();
  nvinfer1::ITensor *inputTensors[] = {input(ctx, 0).trt_tensor_, input(ctx, 1).trt_tensor_};
  int channel = inputTensors[0]->getDimensions().d[1];
  int height = inputTensors[0]->getDimensions().d[INPUT_SIZE2];
  int width = inputTensors[0]->getDimensions().d[INPUT_SIZE3];
  int roi_rows = inputTensors[1]->getDimensions().d[0];
  int roi_cols = inputTensors[1]->getDimensions().d[1];
  auto plugin = std::make_shared<ROIAlignPlugin>(op_name_, pooled_height, pooled_width, spatial_scale, sample_num,
                                                 roi_end_mode, channel, height, width, roi_rows, roi_cols);
  CHECK_NULL_RETURN(plugin);
  nvinfer1::IPluginV2Layer *roialign_layer = ctx->network()->addPluginV2(inputTensors, INPUT_SIZE2, *plugin);
  CHECK_NULL_RETURN(roialign_layer);
  this->layer_ = roialign_layer;
  nvinfer1::ITensor *op_out_tensor = roialign_layer->getOutput(0);
  CHECK_NULL_RETURN(op_out_tensor);
  ctx->RegisterTensor(ITensorHelper{op_out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(ROIAlignPluginCreater);
template class TensorRTPluginCreater<ROIAlignPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int ROIAlignPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                            const void *const *inputs, void *const *outputs, void *workspace,
                            cudaStream_t stream) noexcept {
  return RunCudaROIAlign(inputDesc, inputs, outputs, stream);
}

int ROIAlignPlugin::RunCudaROIAlign(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                                    void *const *outputs, cudaStream_t stream) {
  if (inputDesc[1].type == nvinfer1::DataType::kFLOAT) {
    ROIAlign(static_cast<const float *>(inputs[0]), static_cast<const float *>(inputs[1]), roi_rows_, roi_cols_,
             static_cast<float *>(outputs[0]), spatial_scale_, sample_num_, roi_end_mode_, channel_, height_, width_,
             pooled_height_, pooled_width_, device_id_, stream);
  } else {
    MS_LOG(ERROR) << "unsupported roialign data type";
    return RET_ERROR;
  }
  return RET_OK;
}

nvinfer1::DimsExprs ROIAlignPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs,
                                                        int nbInputDims, nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  dims.nbDims = INPUT_SIZE4;
  dims.d[0] = inputs[1].d[0];
  dims.d[1] = inputs[0].d[1];
  auto pooled_height = exprBuilder.constant(pooled_height_);
  dims.d[INPUT_SIZE2] = pooled_height;
  auto pooled_width = exprBuilder.constant(pooled_width_);
  dims.d[INPUT_SIZE3] = pooled_width;
  return dims;
}

bool ROIAlignPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                               int nbOutputs) noexcept {
  return tensorsDesc[pos].type == nvinfer1::DataType::kFLOAT &&
         tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

nvinfer1::IPluginV2DynamicExt *ROIAlignPlugin::clone() const noexcept {
  auto *plugin = new (std::nothrow) ROIAlignPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "malloc roialign plugin failed";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}
REGISTER_TENSORRT_CREATOR(ops::kNameROIAlign, ROIAlignTensorRT)
}  // namespace mindspore::lite
