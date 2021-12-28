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

#include "src/delegate/tensorrt/op/equal_tensorrt.h"
#include <numeric>
#include <memory>
#include <functional>
#include "src/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"

namespace mindspore::lite {
const char *EQUAL_PLUGIN_VERSION{"1"};
const char *EQUAL_PLUGIN_NAME{"EqualPluginCreater"};
nvinfer1::PluginFieldCollection EqualPluginCreater::field_collection_{};
std::vector<nvinfer1::PluginField> EqualPluginCreater::fields_;
REGISTER_TENSORRT_PLUGIN(EqualPluginCreater);

int EqualTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "invalid input tensor size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensor size: " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int EqualTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  nvinfer1::ITensor *inputTensors[] = {tensorrt_in_tensors_[0].trt_tensor_, tensorrt_in_tensors_[1].trt_tensor_};
  auto plugin = std::make_shared<EqualPlugin>(op_name_);
  nvinfer1::IPluginV2Layer *equal_layer = network->addPluginV2(inputTensors, 2, *plugin);
  if (equal_layer == nullptr) {
    MS_LOG(ERROR) << "create equal layer failed for: " << op_name_;
    return RET_ERROR;
  }
  nvinfer1::ITensor *equal_out = equal_layer->getOutput(0);
  equal_layer->setName(op_name_.c_str());
  equal_out->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(
    ITensorHelper{equal_out, tensorrt_in_tensors_[0].format_, tensorrt_in_tensors_[0].same_format_});
  return RET_OK;
}

// EqualPluginCreater
EqualPluginCreater::EqualPluginCreater() {
  // Fill PluginFieldCollection with PluginField arguments metadata
  field_collection_.nbFields = fields_.size();
  field_collection_.fields = fields_.data();
}

const char *EqualPluginCreater::getPluginName() const noexcept { return EQUAL_PLUGIN_NAME; }

const char *EqualPluginCreater::getPluginVersion() const noexcept { return EQUAL_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection *EqualPluginCreater::getFieldNames() noexcept { return &field_collection_; }

nvinfer1::IPluginV2 *EqualPluginCreater::createPlugin(const char *name,
                                                      const nvinfer1::PluginFieldCollection *fc) noexcept {
  return new (std::nothrow) EqualPlugin(name);
}

nvinfer1::IPluginV2 *EqualPluginCreater::deserializePlugin(const char *name, const void *serialData,
                                                           size_t serialLength) noexcept {
  MS_LOG(DEBUG) << name << " deserialize";
  return new (std::nothrow) EqualPlugin(name);
}

void EqualPluginCreater::setPluginNamespace(const char *libNamespace) noexcept { name_space_ = libNamespace; }

const char *EqualPluginCreater::getPluginNamespace() const noexcept { return name_space_.c_str(); }

// EqualPlugin
int EqualPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                         const void *const *inputs, void *const *outputs, void *workspace,
                         cudaStream_t stream) noexcept {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  int element_cnt = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int64_t>());

  if (inputDesc->type == nvinfer1::DataType::kINT32) {
    const int *input1 = static_cast<const int *>(inputs[0]);
    const int *input2 = static_cast<const int *>(inputs[1]);
    int *output = static_cast<int *>(outputs[0]);
    Equal(input1, input2, output, element_cnt, stream);
  } else if (inputDesc->type == nvinfer1::DataType::kFLOAT) {
    const float *input1 = static_cast<const float *>(inputs[0]);
    const float *input2 = static_cast<const float *>(inputs[1]);
    float *output = static_cast<float *>(outputs[0]);
    Equal(input1, input2, output, element_cnt, stream);
  } else {
    MS_LOG(ERROR) << "unsupported equal data type";
  }
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *EqualPlugin::clone() const noexcept {
  auto *plugin = new EqualPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

nvinfer1::DimsExprs EqualPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                                     nvinfer1::IExprBuilder &exprBuilder) noexcept {
  return *inputs;
}

bool EqualPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                            int nbOutputs) noexcept {
  return true;
}

void EqualPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                  const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {}

size_t EqualPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                     const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  return 0;
}

nvinfer1::DataType EqualPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
  noexcept {
  return inputTypes[0];
}

const char *EqualPlugin::getPluginType() const noexcept { return EQUAL_PLUGIN_NAME; }

const char *EqualPlugin::getPluginVersion() const noexcept { return EQUAL_PLUGIN_VERSION; }

int EqualPlugin::getNbOutputs() const noexcept { return 1; }

int EqualPlugin::initialize() noexcept { return 0; }

void EqualPlugin::terminate() noexcept {}

size_t EqualPlugin::getSerializationSize() const noexcept { return 0; }

void EqualPlugin::serialize(void *buffer) const noexcept {}

void EqualPlugin::destroy() noexcept {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void EqualPlugin::setPluginNamespace(const char *libNamespace) noexcept { name_space_ = libNamespace; }

const char *EqualPlugin::getPluginNamespace() const noexcept { return name_space_.c_str(); }
}  // namespace mindspore::lite
