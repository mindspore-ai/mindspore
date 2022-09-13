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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_FSE_PLUGIN_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_FSE_PLUGIN_H_
#include <string>
#include <vector>
#include <cstring>
#include "src/litert/delegate/tensorrt/op/tensorrt_plugin.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/quant_impl.cuh"

namespace mindspore::lite {
constexpr char *FSE_PLUGIN_NAME{"FSEPluginCreater"};
class FSEPlugin : public TensorRTPlugin {
 public:
  FSEPlugin(const std::string &name, const std::vector<int> &shape, const BitStreamState &bs,
            uint16_t *states_table_device, uint8_t *bit_count_table_device, uint16_t *symbol_table_device,
            size_t table_size, size_t table_log, void *centroids_device, size_t centroids_size, uint32_t device_id = 0)
      : TensorRTPlugin(name, std::string(FSE_PLUGIN_NAME), device_id),
        shape_(shape),
        bs_(bs),
        states_table_device_(states_table_device),
        bit_count_table_device_(bit_count_table_device),
        symbol_table_device_(symbol_table_device),
        table_size_(table_size),
        table_log_(table_log),
        centroids_device_(centroids_device),
        centroids_size_(centroids_size) {}

  FSEPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(FSE_PLUGIN_NAME)) {}

  FSEPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(FSE_PLUGIN_NAME)) {}

  FSEPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
    noexcept override {
    return nvinfer1::DataType::kFLOAT;
  }

  size_t getSerializationSize() const noexcept override { return 0; };
  void serialize(void *buffer) const noexcept override{};

  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;

 private:
  std::vector<int> shape_;
  BitStreamState bs_;
  BitStreamState *bs_device_ = nullptr;

  uint16_t *states_table_device_ = nullptr;
  uint8_t *bit_count_table_device_ = nullptr;
  uint16_t *symbol_table_device_ = nullptr;

  size_t table_size_ = 0;
  size_t table_log_ = 0;

  void *centroids_device_ = nullptr;
  size_t centroids_size_ = 0;
};
class FSEPluginCreater : public TensorRTPluginCreater<FSEPlugin> {
 public:
  FSEPluginCreater() : TensorRTPluginCreater(std::string(FSE_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_FSE_PLUGIN_H_
