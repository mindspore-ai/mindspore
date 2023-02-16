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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_ACTIVATION_OPT_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_ACTIVATION_OPT_PLUGIN_H_

#include <string>
#include <map>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"
#include "src/extendrt/delegate/tensorrt/cuda_impl/cudnn_utils.h"

namespace mindspore::lite {
constexpr auto LSTM_PLUGIN_NAME{"LSTMPlugin"};
constexpr size_t kInputShapeSize = 3;
class LSTMPlugin : public TensorRTPlugin {
 public:
  LSTMPlugin(const std::string &name, int num_layers, int batch_size, int seq_len, int input_size, int hidden_size,
             float dropout, bool has_bias, bool bidirectional, uint32_t device_id)
      : TensorRTPlugin(name, std::string(LSTM_PLUGIN_NAME), device_id),
        num_layers_(num_layers),
        batch_size_(batch_size),
        seq_len_(seq_len),
        input_size_(input_size),
        hidden_size_(hidden_size),
        dropout_(dropout),
        has_bias_(has_bias),
        bidirectional_(bidirectional) {}

  LSTMPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(LSTM_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    num_layers_ = static_cast<const int *>(fields[0].data)[0];
    batch_size_ = static_cast<const int *>(fields[1].data)[0];
    seq_len_ = static_cast<const int *>(fields[INPUT_SIZE2].data)[0];
    input_size_ = static_cast<const int *>(fields[INPUT_SIZE3].data)[0];
    hidden_size_ = static_cast<const int *>(fields[INPUT_SIZE4].data)[0];
    dropout_ = static_cast<const float *>(fields[INPUT_SIZE5].data)[0];
    has_bias_ = static_cast<const bool *>(fields[INPUT_SIZE6].data)[0];
    bidirectional_ = static_cast<const bool *>(fields[INPUT_SIZE7].data)[0];
  }

  LSTMPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(LSTM_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &num_layers_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &batch_size_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &seq_len_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &input_size_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &hidden_size_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &dropout_, sizeof(float));
    DeserializeValue(&serialData, &serialLength, &has_bias_, sizeof(bool));
    DeserializeValue(&serialData, &serialLength, &bidirectional_, sizeof(bool));
  }

  LSTMPlugin() = delete;

  ~LSTMPlugin() {}

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
    noexcept override;
  int getNbOutputs() const noexcept override { return 1; }
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept override {
    return tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           tensorsDesc[pos].type == nvinfer1::DataType::kFLOAT;
  }

 private:
  int num_layers_;
  int batch_size_;
  int seq_len_;
  int input_size_;
  int hidden_size_;

  float dropout_;
  bool has_bias_;
  bool bidirectional_;
};
class LSTMPluginCreater : public TensorRTPluginCreater<LSTMPlugin> {
 public:
  LSTMPluginCreater() : TensorRTPluginCreater(std::string(LSTM_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_ACTIVATION_OPT_PLUGIN_H_
