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
#ifndef MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_EQUAL_TENSORRT_H_
#define MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_EQUAL_TENSORRT_H_
#include <string>
#include <vector>
#include "src/delegate/tensorrt/op/tensorrt_op.h"
#include "src/delegate/tensorrt/op/tensorrt_plugin.h"
#include "src/delegate/tensorrt/cuda_impl/equal.cuh"

namespace mindspore::lite {
constexpr char *EQUAL_PLUGIN_NAME{"EqualPlugin"};
class EqualTensorRT : public TensorRTOp {
 public:
  EqualTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~EqualTensorRT() override = default;

  int AddInnerOp(nvinfer1::INetworkDefinition *network) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;
};

class EqualPlugin : public TensorRTPlugin {
 public:
  explicit EqualPlugin(const std::string name) : TensorRTPlugin(name, std::string(EQUAL_PLUGIN_NAME)) {}

  // It doesn't make sense to make GeluPluginDynamic without arguments, so we delete
  // default constructor.
  EqualPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
};

class EqualPluginCreater : public TensorRTPluginCreater {
 public:
  EqualPluginCreater() : TensorRTPluginCreater(std::string(EQUAL_PLUGIN_NAME)) {}

  nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept override;

  nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                                         size_t serialLength) noexcept override;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_EQUAL_TENSORRT_H_
