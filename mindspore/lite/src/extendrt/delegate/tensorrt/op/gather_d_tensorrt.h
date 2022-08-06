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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_GATHER_D_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_GATHER_D_TENSORRT_H_
#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gather.cuh"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
constexpr auto GATHER_D_PLUGIN_NAME{"GatherDPluginCreater"};
class GatherDTensorRT : public TensorRTOp {
 public:
  GatherDTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                  const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~GatherDTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};

class GatherDPlugin : public TensorRTPlugin {
 public:
  GatherDPlugin(const std::string name, size_t dim, uint32_t device_id)
      : TensorRTPlugin(name, std::string(GATHER_D_PLUGIN_NAME), device_id), axis_(dim) {}

  GatherDPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(GATHER_D_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    axis_ = static_cast<const int *>(fields[0].data)[0];
  }

  GatherDPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(GATHER_D_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &axis_, sizeof(int));
  }

  GatherDPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

 private:
  int axis_;
  size_t dim_before_axis_;
  size_t dim_at_axis_input_;
  size_t dim_at_axis_output_;
  size_t dim_after_axis_;
  void Reshape(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc);
};

class GatherDPluginCreater : public TensorRTPluginCreater<GatherDPlugin> {
 public:
  GatherDPluginCreater() : TensorRTPluginCreater(std::string(GATHER_D_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_GATHER_D_TENSORRT_H_
