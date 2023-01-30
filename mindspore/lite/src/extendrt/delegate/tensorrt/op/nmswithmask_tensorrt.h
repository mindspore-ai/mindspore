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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_OP_NMSWITHMASK_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_OP_NMSWITHMASK_PLUGIN_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
constexpr auto NMSWITHMASK_PLUGIN_NAME{"NMSwithmaskPlugin"};
class NMSwithmaskTensorRT : public TensorRTOp {
 public:
  NMSwithmaskTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                      const std::vector<TensorInfo> &out_tensors, const std::string &name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~NMSwithmaskTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};
class NMSwithmaskPlugin : public TensorRTPlugin {
 public:
  explicit NMSwithmaskPlugin(const std::string name, int num_input, float iou_value)
      : TensorRTPlugin(name, std::string(NMSWITHMASK_PLUGIN_NAME)), num_input_(num_input), iou_value_(iou_value) {}

  NMSwithmaskPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(NMSWITHMASK_PLUGIN_NAME)) {}

  NMSwithmaskPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(NMSWITHMASK_PLUGIN_NAME)) {}

  NMSwithmaskPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
    noexcept override;
  int getNbOutputs() const noexcept override { return 3; }
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept override {
    return tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }

 private:
  int RunCudaNMSwithmask(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
                         cudaStream_t stream);
  const std::string layer_name_;
  std::string name_space_;
  int num_input_{0};
  float iou_value_{0.5};
};
class NMSwithmaskPluginCreater : public TensorRTPluginCreater<NMSwithmaskPlugin> {
 public:
  NMSwithmaskPluginCreater() : TensorRTPluginCreater(std::string(NMSWITHMASK_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_OP_NMSWITHMASK_PLUGIN_H_
