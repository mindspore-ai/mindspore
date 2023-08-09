//  * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_VSL_COMPRESS_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_VSL_COMPRESS_TENSORRT_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"
#include "src/extendrt/delegate/tensorrt/cuda_impl/cudnn_utils.h"
#include "src/fastertransformer/layers/ms_layers/debug_utils.h"
namespace mindspore::lite {
constexpr auto VSL_COMPRESS_PLUGIN_NAME{"VslCompressPlugin"};
class VslCompressPlugin : public TensorRTPlugin {
 public:
  VslCompressPlugin(const char *name, uint32_t device_id, bool is_optimize_pangu_sigma = false)
      : TensorRTPlugin(name, std::string(VSL_COMPRESS_PLUGIN_NAME), device_id),
        is_optimize_pangu_sigma_(is_optimize_pangu_sigma) {}
  VslCompressPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc, bool is_optimize_pangu_sigma = false)
      : TensorRTPlugin(std::string(name), std::string(VSL_COMPRESS_PLUGIN_NAME)),
        is_optimize_pangu_sigma_(is_optimize_pangu_sigma) {}

  VslCompressPlugin(const char *name, const void *serialData, size_t serialLength, bool is_optimize_pangu_sigma = false)
      : TensorRTPlugin(std::string(name), std::string(VSL_COMPRESS_PLUGIN_NAME)),
        is_optimize_pangu_sigma_(is_optimize_pangu_sigma) {
    DeserializeValue(&serialData, &serialLength, &seq_len_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &batch_size_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &is_optimize_pangu_sigma_, sizeof(bool));
  }
  VslCompressPlugin() = delete;

  ~VslCompressPlugin() override {}

  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept override;
  int getNbOutputs() const noexcept override { return C4NUM; }
  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;

 private:
  int batch_size_;
  int seq_len_;
  bool is_optimize_pangu_sigma_;
};
class VslCompressPluginCreater : public TensorRTPluginCreater<VslCompressPlugin> {
 public:
  VslCompressPluginCreater() : TensorRTPluginCreater(std::string(VSL_COMPRESS_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_VSL_COMPRESS_TENSORRT_H_
