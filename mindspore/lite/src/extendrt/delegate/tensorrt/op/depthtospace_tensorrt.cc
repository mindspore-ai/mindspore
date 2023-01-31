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

#include "src/extendrt/delegate/tensorrt/op/depthtospace_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/depthtospace_impl.cuh"
#include "ops/depth_to_space.h"

namespace mindspore::lite {
int DepthToSpaceTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                    const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }

  if (out_tensors.size() < 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int DepthToSpaceTensorRT::AddInnerOp(TensorRTContext *ctx) {
  nvinfer1::ITensor *input_tensor = input(ctx, 0).trt_tensor_;
  auto op = AsOps<ops::DepthToSpace>();
  int block_size = op->get_block_size();

  auto plugin = std::make_shared<DepthToSpacePlugin>(input_tensor->getName(), block_size, device_id_);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "add depthtospace plugin failed for" << op_name_;
    return RET_ERROR;
  }
  nvinfer1::ITensor *inputTensors[] = {input_tensor};
  nvinfer1::IPluginV2Layer *layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
  if (layer == nullptr) {
    MS_LOG(ERROR) << "add depthtospace op failed for TensorRT.";
    return RET_ERROR;
  }
  layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{out_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  this->layer_ = layer;
  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(DepthToSpacePluginCreater);
template class TensorRTPluginCreater<DepthToSpacePlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int DepthToSpacePlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  return RunCudaDepthToSpace(inputDesc, inputs, outputs, stream);
}

int DepthToSpacePlugin::RunCudaDepthToSpace(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                                            void *const *outputs, cudaStream_t stream) {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  int in = input_dims.d[0];
  int ic = input_dims.d[1];
  int ih = input_dims.d[2];
  int iw = input_dims.d[3];
  int on = in;
  int oc = ic / block_size_ / block_size_;
  int oh = ih * block_size_;
  int ow = iw * block_size_;

  int size = on * oc * oh * ow;

  CalDepthToSpace<float>(size, static_cast<const float *>(inputs[0]), in, ic, ih, iw, on, oc, oh, ow, block_size_,
                         static_cast<float *>(outputs[0]), device_id_, stream);
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *DepthToSpacePlugin::clone() const noexcept {
  auto *plugin = new (std::nothrow) DepthToSpacePlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "new plugin failed!";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t DepthToSpacePlugin::getSerializationSize() const noexcept { return sizeof(int); }

nvinfer1::DimsExprs DepthToSpacePlugin::getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs,
                                                            int nbInputDims,
                                                            nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  dims.nbDims = inputs[0].nbDims;
  dims.d[0] = inputs[0].d[0];
  dims.d[1] = inputs[0].d[1];
  auto block_size_sqrt = exprBuilder.constant(block_size_ * block_size_);
  dims.d[1] = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *inputs[0].d[1], *block_size_sqrt);
  auto block_size = exprBuilder.constant(block_size_);
  dims.d[INPUT_SIZE2] =
    exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *inputs[0].d[INPUT_SIZE2], *block_size);
  dims.d[INPUT_SIZE3] =
    exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *inputs[0].d[INPUT_SIZE3], *block_size);
  return dims;
}

void DepthToSpacePlugin::serialize(void *buffer) const noexcept { SerializeValue(&buffer, &block_size_, sizeof(int)); }
REGISTER_TENSORRT_CREATOR(ops::kNameDepthToSpace, DepthToSpaceTensorRT)
}  // namespace mindspore::lite
