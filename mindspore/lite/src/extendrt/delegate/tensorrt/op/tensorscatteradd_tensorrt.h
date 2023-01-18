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
#ifndef MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_TENSOR_SCATTER_ADD_TENSORRT_H_
#define MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_TENSOR_SCATTER_ADD_TENSORRT_H_
#include <string>
#include <vector>
#include <algorithm>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
class TensorScatterAddTensorRT : public TensorRTOp {
 public:
  TensorScatterAddTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                           const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~TensorScatterAddTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};

constexpr auto TENSORSCATTERADD_PLUGIN_NAME{"TensorScatterAddPlugin"};
class TensorScatterAddPlugin : public TensorRTPlugin {
 public:
  TensorScatterAddPlugin(const std::string &name, int device_id)
      : TensorRTPlugin(name, std::string(TENSORSCATTERADD_PLUGIN_NAME), device_id) {}

  TensorScatterAddPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(TENSORSCATTERADD_PLUGIN_NAME)) {}

  TensorScatterAddPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(TENSORSCATTERADD_PLUGIN_NAME)) {}

  TensorScatterAddPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
    noexcept override {
    return inputTypes[0];
  }

 private:
  int RunCudaTensorScatterAdd(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                              void *const *outputs, cudaStream_t stream);
};
class TensorScatterAddPluginCreater : public TensorRTPluginCreater<TensorScatterAddPlugin> {
 public:
  TensorScatterAddPluginCreater() : TensorRTPluginCreater(std::string(TENSORSCATTERADD_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_TENSOR_SCATTER_ADD_TENSORRT_H_
