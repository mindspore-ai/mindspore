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

#include "src/delegate/tensorrt/op/reducescatter_tensorrt.h"
#include <numeric>
#include <thread>
#include "NvInferRuntimeCommon.h"

namespace mindspore::lite {
const char *REDUCESCATTER_PLUGIN_VERSION{"1"};
const char *REDUCESCATTER_PLUGIN_NAME{"ReduceScatterPluginCreater"};
nvinfer1::PluginFieldCollection ReduceScatterPluginCreater::field_collection_{};
std::vector<nvinfer1::PluginField> ReduceScatterPluginCreater::fields_;
REGISTER_TENSORRT_PLUGIN(ReduceScatterPluginCreater);

int ReduceScatterTensorRT::IsSupport(const schema::Primitive *primitive,
                                     const std::vector<mindspore::MSTensor> &in_tensors,
                                     const std::vector<mindspore::MSTensor> &out_tensors) {
#ifndef LITE_CUDA_DISTRIBUTION
  MS_LOG(ERROR)
    << "Unsupported package for gpu distribution feature, please recompile with MS_ENABLE_CUDA_DISTRIBUTION set to on.";
  return RET_ERROR;
#else
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid input tensor size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensor size: " << out_tensors.size();
    return RET_ERROR;
  }
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
#endif
}

int ReduceScatterTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  nvinfer1::ITensor *inputTensors[] = {tensorrt_in_tensors_[0].trt_tensor_};
  auto reduce_op = op_primitive_->value_as_ReduceScatter();
  if (reduce_op == nullptr) {
    MS_LOG(ERROR) << "convert failed for " << op_name_;
    return RET_ERROR;
  }
  auto reduce_mode = reduce_op->mode();
  auto rank = GetGPUGroupSize();
  auto plugin = std::make_shared<ReduceScatterPlugin>(op_name_, reduce_mode, rank);
  MS_LOG(INFO) << op_name_ << " group size: " << rank << ", rank id: " << GetRankID();
  nvinfer1::IPluginV2Layer *reduce_scatter_layer = network->addPluginV2(inputTensors, 1, *plugin);
  if (reduce_scatter_layer == nullptr) {
    MS_LOG(ERROR) << "create ReduceScatter layer failed for: " << op_name_;
    return RET_ERROR;
  }
  nvinfer1::ITensor *reduce_scatter_out = reduce_scatter_layer->getOutput(0);
  reduce_scatter_layer->setName(op_name_.c_str());
  reduce_scatter_out->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(
    ITensorHelper{reduce_scatter_out, tensorrt_in_tensors_[0].format_, tensorrt_in_tensors_[0].same_format_});
  return RET_OK;
}

// ReduceScatterPluginCreater
ReduceScatterPluginCreater::ReduceScatterPluginCreater() {
  // Fill PluginFieldCollection with PluginField arguments metadata
  field_collection_.nbFields = fields_.size();
  field_collection_.fields = fields_.data();
}

const char *ReduceScatterPluginCreater::getPluginName() const noexcept { return REDUCESCATTER_PLUGIN_NAME; }

const char *ReduceScatterPluginCreater::getPluginVersion() const noexcept { return REDUCESCATTER_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection *ReduceScatterPluginCreater::getFieldNames() noexcept {
  return &field_collection_;
}

nvinfer1::IPluginV2 *ReduceScatterPluginCreater::createPlugin(const char *name,
                                                              const nvinfer1::PluginFieldCollection *fc) noexcept {
  const nvinfer1::PluginField *fields = fc->fields;
  schema::ReduceMode red_mode = static_cast<const schema::ReduceMode *>(fields[0].data)[0];
  int rank = static_cast<const int *>(fields[1].data)[0];
  MS_LOG(DEBUG) << "createPlugin: " << name << " of rank: " << rank;
  return new (std::nothrow) ReduceScatterPlugin(name, red_mode, rank);
}
nvinfer1::IPluginV2 *ReduceScatterPluginCreater::deserializePlugin(const char *name, const void *serialData,
                                                                   size_t serialLength) noexcept {
  int rank = GetGPUGroupSize();
  schema::ReduceMode red_mode;
  DeserializeValue(&serialData, &serialLength, &red_mode, sizeof(schema::ReduceMode));
  MS_LOG(DEBUG) << name << " is deserialize as rank: " << rank << ", red_mode: " << red_mode;
  return new (std::nothrow) ReduceScatterPlugin(name, red_mode, rank);
}
void ReduceScatterPluginCreater::setPluginNamespace(const char *libNamespace) noexcept { name_space_ = libNamespace; }

const char *ReduceScatterPluginCreater::getPluginNamespace() const noexcept { return name_space_.c_str(); }

// ReduceScatterPlugin
int ReduceScatterPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                 const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                 void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  nvinfer1::Dims output_dims = outputDesc[0].dims;
  int recieve_element_cnt =
    std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<int64_t>());
  const void *input = inputs[0];
  void *output = outputs[0];
  auto data_type = inputDesc->type;
  auto ret = DistributionCollective::instance().ReduceScatterWrapper(input, output, recieve_element_cnt, data_type,
                                                                     red_mode_, stream, NCCL_WORLD_GROUP);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ReduceScatter nccl run failed for " << layer_name_;
    return ret;
  }
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *ReduceScatterPlugin::clone() const noexcept {
  auto *plugin = new ReduceScatterPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

nvinfer1::DimsExprs ReduceScatterPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                                             int nbInputs,
                                                             nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs out_dims{};
  out_dims.nbDims = inputs->nbDims;
  auto rank_dim = exprBuilder.constant(rank_);
  out_dims.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kCEIL_DIV, *inputs->d[0], *rank_dim);
  for (int i = 1; i < inputs->nbDims; i++) {
    out_dims.d[i] = inputs->d[i];
  }
  return out_dims;
}

bool ReduceScatterPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc,
                                                    int nbInputs, int nbOutputs) noexcept {
  return true;
}

void ReduceScatterPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                          const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {}

size_t ReduceScatterPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                             const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  return 0;
}

nvinfer1::DataType ReduceScatterPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                          int nbInputs) const noexcept {
  return inputTypes[0];
}

const char *ReduceScatterPlugin::getPluginType() const noexcept { return REDUCESCATTER_PLUGIN_NAME; }

const char *ReduceScatterPlugin::getPluginVersion() const noexcept { return REDUCESCATTER_PLUGIN_VERSION; }

int ReduceScatterPlugin::getNbOutputs() const noexcept { return 1; }

int ReduceScatterPlugin::initialize() noexcept { return 0; }

void ReduceScatterPlugin::terminate() noexcept {}

size_t ReduceScatterPlugin::getSerializationSize() const noexcept { return sizeof(schema::ReduceMode); }

void ReduceScatterPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &red_mode_, sizeof(schema::ReduceMode));
}

void ReduceScatterPlugin::destroy() noexcept {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void ReduceScatterPlugin::setPluginNamespace(const char *libNamespace) noexcept { name_space_ = libNamespace; }

const char *ReduceScatterPlugin::getPluginNamespace() const noexcept { return name_space_.c_str(); }
}  // namespace mindspore::lite
