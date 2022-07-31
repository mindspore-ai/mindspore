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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_ACTIVATION_OPT_PLUGIN_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_ACTIVATION_OPT_PLUGIN_H_

#include <string>
#include "src/litert/delegate/tensorrt/op/tensorrt_op.h"
#include "src/litert/delegate/tensorrt/op/tensorrt_plugin.h"
#include "src/litert/delegate/tensorrt/cuda_impl/cudnn_utils.h"

namespace mindspore::lite {
constexpr char *ACTIVATION_OPT_PLUGIN_NAME{"ActivationOptPlugin"};
class ActivationOptPlugin : public TensorRTPlugin {
 public:
  ActivationOptPlugin(const std::string name, schema::ActivationType activation_type, uint32_t device_id)
      : TensorRTPlugin(name, std::string(ACTIVATION_OPT_PLUGIN_NAME), device_id), activation_type_(activation_type) {}

  ActivationOptPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(ACTIVATION_OPT_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    activation_type_ = static_cast<const schema::ActivationType *>(fields[0].data)[0];
  }

  ActivationOptPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(ACTIVATION_OPT_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &activation_type_, sizeof(schema::ActivationType));
  }

  ActivationOptPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;

 private:
  bool needResize(const int *current_dims, const int *last_dims);
  int RunCudaActivation(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
                        cudaStream_t stream);
  int RunCuDNNActivation(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
                         cudaStream_t stream);
  const std::string layer_name_;
  std::string name_space_;
  schema::ActivationType activation_type_;
  cudnnHandle_t cudnn_handle_{nullptr};
  cudnnActivationDescriptor_t activation_desc_{nullptr};
  cudnnTensorDescriptor_t input_desc_{nullptr};
  int infer_dims_[5]{1, 1, 1, 1, 1};
  int infer_stride_[5]{1, 1, 1, 1, 1};
  int infer_dims_cnt_{0};
};
class ActivationOptPluginCreater : public TensorRTPluginCreater<ActivationOptPlugin> {
 public:
  ActivationOptPluginCreater() : TensorRTPluginCreater(std::string(ACTIVATION_OPT_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_ACTIVATION_OPT_PLUGIN_H_
