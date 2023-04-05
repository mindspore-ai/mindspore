/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_ALLREDUCE_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_ALLREDUCE_TENSORRT_H_
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "src/extendrt/delegate/tensorrt/distribution/distribution_collective.h"

namespace mindspore::lite {
constexpr auto ALLREDUCE_PLUGIN_NAME{"AllReducePlugin"};
class AllReduceTensorRT : public TensorRTOp {
 public:
  AllReduceTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                    const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~AllReduceTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};

class AllReducePlugin : public TensorRTPlugin {
 public:
  AllReducePlugin(const std::string name, ReduceMode red_mode, int rank, uint32_t device_id)
      : TensorRTPlugin(name, std::string(ALLREDUCE_PLUGIN_NAME), device_id), red_mode_(red_mode), rank_(rank) {}

  AllReducePlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(ALLREDUCE_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    red_mode_ = static_cast<const ReduceMode *>(fields[0].data)[0];
    rank_ = static_cast<const int *>(fields[1].data)[0];
  }

  AllReducePlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(ALLREDUCE_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &red_mode_, sizeof(ReduceMode));
    DeserializeValue(&serialData, &serialLength, &rank_, sizeof(int));
  }

  AllReducePlugin() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;

 private:
  ReduceMode red_mode_;
  int rank_{0};
};
class AllReducePluginCreater : public TensorRTPluginCreater<AllReducePlugin> {
 public:
  AllReducePluginCreater() : TensorRTPluginCreater(std::string(ALLREDUCE_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_ALLREDUCE_TENSORRT_H_
