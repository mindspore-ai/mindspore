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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_TILE_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_TILE_TENSORRT_H_
#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tile_impl.cuh"

namespace mindspore::lite {
constexpr auto TILE_PLUGIN_NAME{"TilePluginCreater"};
class TileTensorRT : public TensorRTOp {
 public:
  TileTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
               const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~TileTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  bool IsWeightInputHanledInner() const override { return true; }

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;

 private:
  int RunAsConcat(TensorRTContext *ctx, const ITensorHelper &tile_input);
  int RunAsPlugin(TensorRTContext *ctx, const ITensorHelper &tile_input);
  std::vector<float> repeats_;
};

class TilePlugin : public TensorRTPlugin {
 public:
  explicit TilePlugin(const std::string name, const std::vector<float> &repeats, uint32_t device_id)
      : TensorRTPlugin(name, std::string(TILE_PLUGIN_NAME), device_id), repeats_(repeats) {}

  TilePlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(TILE_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    size_t dims = static_cast<const size_t *>(fields[0].data)[0];
    for (size_t i = 0; i < dims; i++) {
      float one_repeat = static_cast<const float *>(fields[0].data)[i + 1];
      repeats_.push_back(one_repeat);
    }
  }

  TilePlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(TILE_PLUGIN_NAME)) {
    size_t dims;
    DeserializeValue(&serialData, &serialLength, &dims, sizeof(size_t));
    for (size_t i = 0; i < dims; i++) {
      float one_repeat;
      DeserializeValue(&serialData, &serialLength, &one_repeat, sizeof(float));
      repeats_.push_back(one_repeat);
    }
  }

  TilePlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  void terminate() noexcept override;

 private:
  std::vector<float> repeats_;
  size_t *device_input_shape_{nullptr};
  size_t *device_output_shape_{nullptr};
};
class TilePluginCreater : public TensorRTPluginCreater<TilePlugin> {
 public:
  TilePluginCreater() : TensorRTPluginCreater(std::string(TILE_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_TILE_TENSORRT_H_
