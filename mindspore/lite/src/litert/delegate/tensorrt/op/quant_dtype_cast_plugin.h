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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_QUANT_DTYPE_CAST_PLUGIN_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_QUANT_DTYPE_CAST_PLUGIN_H_
#include <string>
#include <vector>
#include <cstring>
#include "src/litert/delegate/tensorrt/op/tensorrt_plugin.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
constexpr char *QUANT_DTYPE_CAST_PLUGIN_NAME{"QuantDTypeCastPluginCreater"};
class QuantDTypeCastPlugin : public TensorRTPlugin {
 public:
  QuantDTypeCastPlugin(const std::string &name, size_t preferred_dim, float *scale, int *zp, size_t param_num,
                       const std::vector<int> &shape, uint32_t device_id = 0)
      : TensorRTPlugin(name, std::string(QUANT_DTYPE_CAST_PLUGIN_NAME), device_id),
        preferred_dim_(preferred_dim),
        scale_(scale),
        zp_(zp),
        param_num_(param_num),
        shape_(shape) {}

  QuantDTypeCastPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(QUANT_DTYPE_CAST_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    memcpy(&preferred_dim_, fields[FIRST_INPUT].data, fields[FIRST_INPUT].length);
    memcpy(scale_, fields[SECOND_INPUT].data, fields[SECOND_INPUT].length);
    memcpy(zp_, fields[THIRD_INPUT].data, fields[THIRD_INPUT].length);
    memcpy(shape_.data(), fields[FIFTH_INPUT].data, fields[FIFTH_INPUT].length);
  }

  QuantDTypeCastPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(QUANT_DTYPE_CAST_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &preferred_dim_, sizeof(size_t));
    DeserializeValue(&serialData, &serialLength, scale_, param_num_ * sizeof(float));
    DeserializeValue(&serialData, &serialLength, zp_, param_num_ * sizeof(int));
    DeserializeValue(&serialData, &serialLength, shape_.data(), shape_.size() * sizeof(int64_t));
  }

  QuantDTypeCastPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const noexcept override {
    return nvinfer1::DataType::kFLOAT;
  }

  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;

  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;

 private:
  size_t preferred_dim_;
  float *scale_ = nullptr;
  int *zp_ = nullptr;
  size_t param_num_;
  std::vector<int> shape_;
};
class QuantDTypeCastPluginCreater : public TensorRTPluginCreater<QuantDTypeCastPlugin> {
 public:
  QuantDTypeCastPluginCreater() : TensorRTPluginCreater(std::string(QUANT_DTYPE_CAST_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_QUANT_DTYPE_CAST_PLUGIN_H_
