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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_MAXPOOL_WITH_ARGMAX_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_MAXPOOL_WITH_ARGMAX_PLUGIN_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
class MaxPoolWithArgMaxTensorRT : public TensorRTOp {
 public:
  MaxPoolWithArgMaxTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                            const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~MaxPoolWithArgMaxTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};

constexpr auto MAXPOOL_WITH_ARGMAX_PLUGIN_NAME{"MaxPoolWithArgMaxPlugin"};
class MaxPoolWithArgMaxPlugin : public TensorRTPlugin {
 public:
  MaxPoolWithArgMaxPlugin(const std::string name, const std::vector<int64_t> &kernel_size,
                          const std::vector<int64_t> &strides, const PadMode &pad_mode)
      : TensorRTPlugin(name, std::string(MAXPOOL_WITH_ARGMAX_PLUGIN_NAME)),
        kernel_size_(kernel_size),
        strides_(strides),
        pad_mode_(pad_mode) {}

  MaxPoolWithArgMaxPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(MAXPOOL_WITH_ARGMAX_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    kernel_size_.resize(INPUT_SIZE4);
    kernel_size_[0] = static_cast<const float *>(fields[0].data)[0];
    kernel_size_[1] = static_cast<const float *>(fields[1].data)[0];
    kernel_size_[INPUT_SIZE2] = static_cast<const float *>(fields[INPUT_SIZE2].data)[0];
    kernel_size_[INPUT_SIZE3] = static_cast<const float *>(fields[INPUT_SIZE3].data)[0];
    strides_.resize(INPUT_SIZE4);
    strides_[0] = static_cast<const float *>(fields[INPUT_SIZE4].data)[0];
    strides_[1] = static_cast<const float *>(fields[INPUT_SIZE5].data)[0];
    strides_[INPUT_SIZE2] = static_cast<const float *>(fields[INPUT_SIZE6].data)[0];
    strides_[INPUT_SIZE3] = static_cast<const float *>(fields[INPUT_SIZE7].data)[0];
    pad_mode_ = static_cast<const PadMode *>(fields[INPUT_SIZE8].data)[0];
  }

  MaxPoolWithArgMaxPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(MAXPOOL_WITH_ARGMAX_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &kernel_size_[0], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &kernel_size_[1], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &kernel_size_[INPUT_SIZE2], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &kernel_size_[INPUT_SIZE3], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &strides_[0], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &strides_[1], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &strides_[INPUT_SIZE2], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &strides_[INPUT_SIZE3], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &pad_mode_, sizeof(PadMode));
  }

  MaxPoolWithArgMaxPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  int getNbOutputs() const noexcept override { return INPUT_SIZE2; }

 private:
  int RunCudaMaxPoolWithArgmax(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                               void *const *outputs, cudaStream_t stream);
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> strides_;
  PadMode pad_mode_;
};
class MaxPoolWithArgMaxPluginCreater : public TensorRTPluginCreater<MaxPoolWithArgMaxPlugin> {
 public:
  MaxPoolWithArgMaxPluginCreater() : TensorRTPluginCreater(std::string(MAXPOOL_WITH_ARGMAX_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_MAXPOOL_WITH_ARGMAX_PLUGIN_H_
