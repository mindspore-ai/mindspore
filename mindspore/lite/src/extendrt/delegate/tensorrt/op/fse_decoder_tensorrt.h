/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_FSE_DECODER_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_FSE_DECODER_TENSORRT_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
class FseDecoderTensorRT : public TensorRTOp {
 public:
  FseDecoderTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                     const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~FseDecoderTensorRT() override = default;

  bool IsWeightInputHanledInner() const override { return true; }
  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;

 private:
  bool IsChunkEndDataValid();
};

constexpr auto FSE_DECODER_PLUGIN_NAME{"FseDecoderPlugin"};
class FseDecoderPlugin : public TensorRTPlugin {
 public:
  FseDecoderPlugin(const std::string name, int64_t curr_chunk_idx, int64_t dst_type, int64_t curr_bit_count,
                   int64_t table_log, uint64_t curr_chunk, const ShapeVector &out_shape, uint32_t device_id)
      : TensorRTPlugin(name, std::string(FSE_DECODER_PLUGIN_NAME), device_id),
        curr_chunk_idx_(curr_chunk_idx),
        dst_type_(dst_type),
        curr_bit_count_(curr_bit_count),
        table_log_(table_log),
        curr_chunk_(curr_chunk),
        output_shape_(out_shape) {}

  FseDecoderPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(FSE_DECODER_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    curr_chunk_idx_ = static_cast<const int *>(fields[0].data)[0];
    dst_type_ = static_cast<const int *>(fields[1].data)[0];
    curr_bit_count_ = static_cast<const int *>(fields[INPUT_SIZE2].data)[0];
    table_log_ = static_cast<const int *>(fields[INPUT_SIZE3].data)[0];
    curr_chunk_ = static_cast<const int *>(fields[INPUT_SIZE4].data)[0];
  }

  FseDecoderPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(FSE_DECODER_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &curr_chunk_idx_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &dst_type_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &curr_bit_count_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &table_log_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &curr_chunk_, sizeof(int));
  }

  FseDecoderPlugin() = delete;

  ~FseDecoderPlugin() override {}

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
    noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept override;
  void terminate() noexcept override;
  int initialize() noexcept override;

 private:
  template <typename T>
  int RunFseDecoder(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream);

  const std::string layer_name_;
  std::string name_space_;
  uint64_t curr_chunk_idx_;
  int64_t dst_type_;
  uint64_t curr_bit_count_;
  uint64_t table_log_;
  uint64_t curr_chunk_;
  ShapeVector output_shape_;
};
class FseDecoderPluginCreater : public TensorRTPluginCreater<FseDecoderPlugin> {
 public:
  FseDecoderPluginCreater() : TensorRTPluginCreater(std::string(FSE_DECODER_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_FSE_DECODER_TENSORRT_H_
