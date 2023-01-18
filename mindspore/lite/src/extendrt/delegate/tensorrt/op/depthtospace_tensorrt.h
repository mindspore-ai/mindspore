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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_DEPTHTOSPACETENSORRT_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_DEPTHTOSPACETENSORRT_PLUGIN_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
class DepthToSpaceTensorRT : public TensorRTOp {
 public:
  DepthToSpaceTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                       const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~DepthToSpaceTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};

constexpr auto DEPTHTOSPACETENSORRT_PLUGIN_NAME{"DepthToSpacePlugin"};
class DepthToSpacePlugin : public TensorRTPlugin {
 public:
  DepthToSpacePlugin(const std::string name, int block_size, uint32_t device_id)
      : TensorRTPlugin(name, std::string(DEPTHTOSPACETENSORRT_PLUGIN_NAME), device_id), block_size_(block_size) {}

  DepthToSpacePlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(DEPTHTOSPACETENSORRT_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    block_size_ = static_cast<const int *>(fields[0].data)[0];
  }

  DepthToSpacePlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(DEPTHTOSPACETENSORRT_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &block_size_, sizeof(int));
  }

  DepthToSpacePlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;

 private:
  int RunCudaDepthToSpace(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
                          cudaStream_t stream);
  int block_size_;
  const std::string layer_name_;
  std::string name_space_;
};
class DepthToSpacePluginCreater : public TensorRTPluginCreater<DepthToSpacePlugin> {
 public:
  DepthToSpacePluginCreater() : TensorRTPluginCreater(std::string(DEPTHTOSPACETENSORRT_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_DEPTHTOSPACETENSORRT_PLUGIN_H_
