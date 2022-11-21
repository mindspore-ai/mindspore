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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_PRINT_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_PRINT_PLUGIN_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
constexpr auto *PRINT_PLUGIN_NAME{"PrintPlugin"};
class PrintPlugin : public TensorRTPlugin {
 public:
  PrintPlugin(const std::string name, uint32_t device_id)
      : TensorRTPlugin(name, std::string(PRINT_PLUGIN_NAME), device_id) {}

  PrintPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(PRINT_PLUGIN_NAME)) {}

  PrintPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(PRINT_PLUGIN_NAME)) {}

  PrintPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept {
    return tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }

 private:
  int RunPrint(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
               cudaStream_t stream);
};
class PrintPluginCreater : public TensorRTPluginCreater<PrintPlugin> {
 public:
  PrintPluginCreater() : TensorRTPluginCreater(std::string(PRINT_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_PRINT_PLUGIN_H_
