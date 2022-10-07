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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_WHERE_TENSORRT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_WHERE_TENSORRT_H_

#include <string>
#include <vector>
#include "src/litert/delegate/tensorrt/op/tensorrt_op.h"
#include "src/litert/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
class WhereTensorRT : public TensorRTOp {
 public:
  WhereTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~WhereTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;
};

constexpr char *WHERE_PLUGIN_NAME{"WherePlugin"};
class WherePlugin : public TensorRTPlugin {
 public:
  WherePlugin(const std::string name, schema::PrimitiveType primitive_type)
      : TensorRTPlugin(name, std::string(WHERE_PLUGIN_NAME)), primitive_type_(primitive_type) {}

  WherePlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(WHERE_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    primitive_type_ = static_cast<const schema::PrimitiveType *>(fields[0].data)[0];
  }

  WherePlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(WHERE_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &primitive_type_, sizeof(schema::PrimitiveType));
  }

  WherePlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
    noexcept override;

 private:
  int RunCudaWhere(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
                   cudaStream_t stream);
  schema::PrimitiveType primitive_type_;
};
class WherePluginCreater : public TensorRTPluginCreater<WherePlugin> {
 public:
  WherePluginCreater() : TensorRTPluginCreater(std::string(WHERE_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_WHERE_PLUGIN_H_
