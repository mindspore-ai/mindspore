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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_OP_SELECT_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_OP_SELECT_PLUGIN_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
constexpr auto SELECT_PLUGIN_NAME{"SelectPlugin"};
class SelectTensorRT : public TensorRTOp {
 public:
  SelectTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                 const std::vector<TensorInfo> &out_tensors, const std::string &name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~SelectTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};
class SelectPlugin : public TensorRTPlugin {
 public:
  explicit SelectPlugin(const std::string name) : TensorRTPlugin(name, std::string(SELECT_PLUGIN_NAME)) {}

  SelectPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(SELECT_PLUGIN_NAME)) {}

  SelectPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(SELECT_PLUGIN_NAME)) {}

  SelectPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

 private:
  int RunCudaSelect(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
                    cudaStream_t stream);
  const std::string layer_name_;
  std::string name_space_;
};
class SelectPluginCreater : public TensorRTPluginCreater<SelectPlugin> {
 public:
  SelectPluginCreater() : TensorRTPluginCreater(std::string(SELECT_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_OP_SELECT_PLUGIN_H_
