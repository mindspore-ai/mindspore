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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_TENSORRT_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_TENSORRT_PLUGIN_H_
#include <string>
#include <vector>
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "NvInferRuntimeCommon.h"
#include <NvInfer.h>

namespace mindspore::lite {
void SerializeValue(void **buffer, const void *value, size_t cpy_size);
void DeserializeValue(void const **buffer, size_t *buffer_size, void *value, size_t cpy_size);
class TensorRTPlugin : public nvinfer1::IPluginV2DynamicExt {
 public:
  TensorRTPlugin(const std::string &layer_name, const std::string &plugin_name, uint32_t device_id = 0)
      : layer_name_(layer_name), plugin_name_(plugin_name), device_id_(device_id) {}

  // It doesn't make sense to make GeluPluginDynamic without arguments, so we delete
  // default constructor.
  TensorRTPlugin() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
    noexcept override;

  // IPluginV2 Methods
  const char *getPluginType() const noexcept override;
  const char *getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char *pluginNamespace) noexcept override;
  const char *getPluginNamespace() const noexcept override;

 protected:
  std::string layer_name_;
  std::string name_space_;
  std::string plugin_version_{"1"};
  std::string plugin_name_;
  uint32_t device_id_{0};
};

template <class T>
class TensorRTPluginCreater : public nvinfer1::IPluginCreator {
 public:
  explicit TensorRTPluginCreater(const std::string &plugin_name) : plugin_name_(plugin_name) {
    // Fill PluginFieldCollection with PluginField arguments metadata
    field_collection_.nbFields = fields_.size();
    field_collection_.fields = fields_.data();
  }

  const char *getPluginName() const noexcept override { return plugin_name_.c_str(); }

  const char *getPluginVersion() const noexcept override { return plugin_version_.c_str(); }

  const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override { return &field_collection_; }

  void setPluginNamespace(const char *pluginNamespace) noexcept override { name_space_ = std::string(pluginNamespace); }

  const char *getPluginNamespace() const noexcept override { return name_space_.c_str(); }

  nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept override {
    return new (std::nothrow) T(name, fc);
  }

  nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *data, size_t len) noexcept override {
    return new (std::nothrow) T(name, data, len);
  }

 protected:
  static nvinfer1::PluginFieldCollection field_collection_;
  static std::vector<nvinfer1::PluginField> fields_;
  std::string name_space_;
  std::string plugin_version_{"1"};
  std::string plugin_name_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_TENSORRT_PLUGIN_H_
