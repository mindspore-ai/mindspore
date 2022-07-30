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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_NORMALIZE_OPT_PLUGIN_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_NORMALIZE_OPT_PLUGIN_H_
#include <string>
#include <vector>
#include "src/litert/delegate/tensorrt/op/tensorrt_op.h"
#include "src/litert/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
constexpr char *NORMALIZE_OPT_PLUGIN_NAME{"NormalizeOptPlugin"};
class NormalizeOptPlugin : public TensorRTPlugin {
 public:
  NormalizeOptPlugin(const std::string name, size_t axis, float epsilion, uint32_t device_id)
      : TensorRTPlugin(name, std::string(NORMALIZE_OPT_PLUGIN_NAME), device_id), axis_(axis), epsilion_(epsilion) {}

  NormalizeOptPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(NORMALIZE_OPT_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    axis_ = static_cast<const size_t *>(fields[0].data)[0];
    epsilion_ = static_cast<const float *>(fields[1].data)[0];
  }

  NormalizeOptPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(NORMALIZE_OPT_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &axis_, sizeof(size_t));
    DeserializeValue(&serialData, &serialLength, &epsilion_, sizeof(float));
  }

  NormalizeOptPlugin() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;

 private:
  size_t axis_{0};
  float epsilion_{0.0f};
};
class NormalizeOptPluginCreater : public TensorRTPluginCreater<NormalizeOptPlugin> {
 public:
  NormalizeOptPluginCreater() : TensorRTPluginCreater(std::string(NORMALIZE_OPT_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_MATMUL_OPT_PLUGIN_H_
