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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_ONEHOT_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_ONEHOT_PLUGIN_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
class OnehotTensorRT : public TensorRTOp {
 public:
  OnehotTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                 const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                 const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~OnehotTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;
};

constexpr char *ONEHOT_PLUGIN_NAME{"OnehotPlugin"};
class OnehotPlugin : public TensorRTPlugin {
 public:
  OnehotPlugin(const std::string name, int depth, schema::PrimitiveType primitive_type)
      : TensorRTPlugin(name, std::string(ONEHOT_PLUGIN_NAME)), depth_(depth), primitive_type_(primitive_type) {}

  OnehotPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(ONEHOT_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    depth_ = static_cast<const int *>(fields[0].data)[0];
    primitive_type_ = static_cast<const schema::PrimitiveType *>(fields[1].data)[0];
  }

  OnehotPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(ONEHOT_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &batch_dims_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &feature_dims_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &depth_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &primitive_type_, sizeof(schema::PrimitiveType));
  }

  OnehotPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;

 private:
  int RunCudaOneHot(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
                    cudaStream_t stream);

  int batch_dims_{1};
  int feature_dims_{1};
  int depth_{1};
  schema::PrimitiveType primitive_type_;
};
class OnehotPluginCreater : public TensorRTPluginCreater<OnehotPlugin> {
 public:
  OnehotPluginCreater() : TensorRTPluginCreater(std::string(ONEHOT_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_ONEHOT_PLUGIN_H_
