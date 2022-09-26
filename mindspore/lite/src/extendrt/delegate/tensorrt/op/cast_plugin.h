/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_CAST_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_CAST_PLUGIN_H_
#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
constexpr auto CAST_PLUGIN_NAME{"CastPluginCreater"};
class CastPlugin : public TensorRTPlugin {
 public:
  CastPlugin(const std::string name, nvinfer1::DataType origin_datatype, nvinfer1::DataType dest_datatype,
             uint32_t device_id = 0)
      : TensorRTPlugin(name, std::string(CAST_PLUGIN_NAME), device_id),
        origin_datatype_(origin_datatype),
        dest_datatype_(dest_datatype) {}

  CastPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(CAST_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    origin_datatype_ = static_cast<const nvinfer1::DataType *>(fields[0].data)[0];
    dest_datatype_ = static_cast<const nvinfer1::DataType *>(fields[1].data)[0];
  }

  CastPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(CAST_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &origin_datatype_, sizeof(nvinfer1::DataType));
    DeserializeValue(&serialData, &serialLength, &dest_datatype_, sizeof(nvinfer1::DataType));
  }

  CastPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

  nvinfer1::DataType getOutputDataType(int, const nvinfer1::DataType *, int) const noexcept override;

  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;

 private:
  nvinfer1::DataType origin_datatype_;
  nvinfer1::DataType dest_datatype_;
};
class CastPluginCreater : public TensorRTPluginCreater<CastPlugin> {
 public:
  CastPluginCreater() : TensorRTPluginCreater(std::string(CAST_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_CAST_PLUGIN_H_
