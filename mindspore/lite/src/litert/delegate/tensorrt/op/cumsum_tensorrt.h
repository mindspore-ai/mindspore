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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_CUMSUM_PLUGIN_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_CUMSUM_PLUGIN_H_

#include <string>
#include <vector>
#include "src/litert/delegate/tensorrt/op/tensorrt_op.h"
#include "src/litert/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
class CumsumTensorRT : public TensorRTOp {
 public:
  CumsumTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                 const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                 const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~CumsumTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;
};

constexpr char *CUMSUM_PLUGIN_NAME{"CumsumPlugin"};
class CumsumPlugin : public TensorRTPlugin {
 public:
  CumsumPlugin(const std::string name, int axis, bool exclusive, bool reverse, uint32_t device_id)
      : TensorRTPlugin(name, std::string(CUMSUM_PLUGIN_NAME), device_id),
        axis_(axis),
        exclusive_(exclusive),
        reverse_(reverse) {}

  CumsumPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(CUMSUM_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    axis_ = static_cast<const int *>(fields[0].data)[0];
    exclusive_ = static_cast<const bool *>(fields[1].data)[0];
    reverse_ = static_cast<const bool *>(fields[2].data)[0];
  }

  CumsumPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(CUMSUM_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &axis_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &exclusive_, sizeof(bool));
    DeserializeValue(&serialData, &serialLength, &reverse_, sizeof(bool));
  }

  CumsumPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;

 private:
  int RunCudaCumsum(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
                    cudaStream_t stream);
  bool exclusive_;
  bool reverse_;
  int axis_;
  const std::string layer_name_;
  std::string name_space_;
};
class CumsumPluginCreater : public TensorRTPluginCreater<CumsumPlugin> {
 public:
  CumsumPluginCreater() : TensorRTPluginCreater(std::string(CUMSUM_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_CUMSUM_PLUGIN_H_
