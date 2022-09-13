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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_MHA_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_MHA_TENSORRT_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"
#include "src/extendrt/delegate/tensorrt/cuda_impl/cudnn_utils.h"

namespace mindspore::lite {
class MhaTensorRT : public TensorRTOp {
 public:
  MhaTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
              const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~MhaTensorRT() override = default;
  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};

constexpr auto MHA_PLUGIN_NAME{"AttentionPlugin"};
class MhaPlugin : public TensorRTPlugin {
 public:
  MhaPlugin(const std::string name, int compute_type, int head_number, int head_size, int is_cross,
            cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, uint32_t device_id)
      : TensorRTPlugin(name, std::string(MHA_PLUGIN_NAME), device_id),
        compute_type_(compute_type),
        head_number_(head_number),
        head_size_(head_size),
        is_cross_(is_cross),
        cublas_handle_(cublas_handle),
        cublaslt_handle_(cublaslt_handle) {}

  MhaPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(MHA_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    compute_type_ = static_cast<const int *>(fields[0].data)[0];
    head_number_ = static_cast<const int *>(fields[1].data)[0];
    head_size_ = static_cast<const int *>(fields[2].data)[0];
    is_cross_ = static_cast<const int *>(fields[3].data)[0];
  }

  MhaPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(MHA_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &compute_type_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &head_number_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &head_size_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &is_cross_, sizeof(int));
  }

  MhaPlugin() = delete;

  ~MhaPlugin() override {
    // std::cout << "~MhaPlugin" << std::endl;
  }

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  void terminate() noexcept override;

 private:
  bool needResize(const int *current_dims, const int *last_dims);
  int RunCudaMha(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                 const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream);
  int RunCudaCrossMha(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                      const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream);

  const std::string layer_name_;
  std::string name_space_;
  int compute_type_;
  int head_number_;
  int head_size_;
  int is_cross_;
  cublasHandle_t cublas_handle_;
  cublasLtHandle_t cublaslt_handle_;
  void *qkv_buf_{nullptr};
  void *q_buf_2_{nullptr};
  void *qk_buf_{nullptr};
  void *qkv_buf_2_{nullptr};
  void *qkv_buf_3_{nullptr};
  void *output1_{nullptr};
  void *output2_{nullptr};
};
class MhaPluginCreater : public TensorRTPluginCreater<MhaPlugin> {
 public:
  MhaPluginCreater() : TensorRTPluginCreater(std::string(MHA_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_MHA_TENSORRT_H_
