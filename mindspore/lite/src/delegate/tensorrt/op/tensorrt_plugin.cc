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

#include "src/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
nvinfer1::PluginFieldCollection TensorRTPluginCreater::field_collection_{};
std::vector<nvinfer1::PluginField> TensorRTPluginCreater::fields_;

nvinfer1::DimsExprs TensorRTPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                                        int nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept {
  return inputs[0];
}

bool TensorRTPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                               int nbOutputs) noexcept {
  return true;
}

void TensorRTPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                     const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {}

size_t TensorRTPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                        const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  return 0;
}

nvinfer1::DataType TensorRTPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                     int nbInputs) const noexcept {
  return inputTypes[0];
}

const char *TensorRTPlugin::getPluginType() const noexcept { return plugin_name_.c_str(); }

const char *TensorRTPlugin::getPluginVersion() const noexcept { return plugin_version_.c_str(); }

int TensorRTPlugin::getNbOutputs() const noexcept { return 1; }

int TensorRTPlugin::initialize() noexcept { return 0; }

void TensorRTPlugin::terminate() noexcept {}

size_t TensorRTPlugin::getSerializationSize() const noexcept { return 0; }

void TensorRTPlugin::serialize(void *buffer) const noexcept {}

void TensorRTPlugin::destroy() noexcept {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void TensorRTPlugin::setPluginNamespace(const char *libNamespace) noexcept { name_space_ = libNamespace; }

const char *TensorRTPlugin::getPluginNamespace() const noexcept { return name_space_.c_str(); }

TensorRTPluginCreater::TensorRTPluginCreater(const std::string &plugin_name) {
  // Fill PluginFieldCollection with PluginField arguments metadata
  field_collection_.nbFields = fields_.size();
  field_collection_.fields = fields_.data();
  plugin_name_ = plugin_name;
}

const char *TensorRTPluginCreater::getPluginName() const noexcept { return plugin_name_.c_str(); }

const char *TensorRTPluginCreater::getPluginVersion() const noexcept { return plugin_version_.c_str(); }

const nvinfer1::PluginFieldCollection *TensorRTPluginCreater::getFieldNames() noexcept { return &field_collection_; }

void TensorRTPluginCreater::setPluginNamespace(const char *libNamespace) noexcept { name_space_ = libNamespace; }

const char *TensorRTPluginCreater::getPluginNamespace() const noexcept { return name_space_.c_str(); }
}  // namespace mindspore::lite
