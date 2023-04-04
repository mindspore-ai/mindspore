//  * Copyright 2023 Huawei Technologies Co., Ltd
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_USE_PAST_EMBEDDING_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_USE_PAST_EMBEDDING_TENSORRT_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"
#include "src/extendrt/delegate/tensorrt/cuda_impl/cudnn_utils.h"
namespace mindspore::lite {
constexpr auto USE_PAST_EMBEDDING_PLUGIN_NAME{"UsePastEmbeddingPlugin"};
class UsePastEmbeddingTensorRT : public TensorRTOp {
 public:
  UsePastEmbeddingTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                           const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~UsePastEmbeddingTensorRT() override = default;
  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};
class UsePastEmbeddingPlugin : public TensorRTPlugin {
 public:
  UsePastEmbeddingPlugin(const char *name, uint32_t device_id)
      : TensorRTPlugin(name, std::string(USE_PAST_EMBEDDING_PLUGIN_NAME), device_id) {}
  UsePastEmbeddingPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(USE_PAST_EMBEDDING_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    size_ = static_cast<const int *>(fields[0].data)[0];
  }

  UsePastEmbeddingPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(USE_PAST_EMBEDDING_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &size_, sizeof(int));
  }
  UsePastEmbeddingPlugin() = delete;

  ~UsePastEmbeddingPlugin() override {}

  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;

 private:
  int size_;
};
class UsePastEmbeddingPluginCreater : public TensorRTPluginCreater<UsePastEmbeddingPlugin> {
 public:
  UsePastEmbeddingPluginCreater() : TensorRTPluginCreater(std::string(USE_PAST_EMBEDDING_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_USE_PAST_EMBEDDING_TENSORRT_H_
