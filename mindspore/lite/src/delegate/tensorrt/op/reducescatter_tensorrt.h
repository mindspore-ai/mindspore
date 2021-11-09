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
#ifndef MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_REDUCESCATTER_TENSORRT_H_
#define MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_REDUCESCATTER_TENSORRT_H_
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "src/delegate/tensorrt/op/tensorrt_op.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"
#include "src/delegate/tensorrt/distribution/distribution_collective.h"

namespace mindspore::lite {
class ReduceScatterTensorRT : public TensorRTOp {
 public:
  ReduceScatterTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                        const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name)
      : TensorRTOp(primitive, in_tensors, out_tensors, name) {}

  ~ReduceScatterTensorRT() override = default;

  int AddInnerOp(nvinfer1::INetworkDefinition *network) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

 private:
  // ReduceScatterTensorRT
};
class ReduceScatterPluginCreater : public nvinfer1::IPluginCreator {
 public:
  ReduceScatterPluginCreater();

  const char *getPluginName() const noexcept override;

  const char *getPluginVersion() const noexcept override;

  const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override;

  nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept override;

  nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                                         size_t serialLength) noexcept override;

  void setPluginNamespace(const char *pluginNamespace) noexcept override;

  const char *getPluginNamespace() const noexcept override;

 private:
  static nvinfer1::PluginFieldCollection field_collection_;
  static std::vector<nvinfer1::PluginField> fields_;
  std::string name_space_;
};

class ReduceScatterPlugin : public nvinfer1::IPluginV2DynamicExt {
 public:
  ReduceScatterPlugin(const std::string name, schema::ReduceMode red_mode, int rank)
      : layer_name_(name), red_mode_(red_mode), rank_(rank) {}

  // It doesn't make sense to make GeluPluginDynamic without arguments, so we delete
  // default constructor.
  ReduceScatterPlugin() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
    noexcept override;

  // IPluginV2 Methods
  const char *getPluginType() const noexcept override;
  const char *getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char *pluginNamespace) noexcept override;
  const char *getPluginNamespace() const noexcept override;

 private:
  const std::string layer_name_;
  std::string name_space_;
  int rank_;
  schema::ReduceMode red_mode_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_REDUCESCATTER_TENSORRT_H_
