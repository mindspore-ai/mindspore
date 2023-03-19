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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_DECODER_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_DECODER_TENSORRT_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"
#include "src/extendrt/delegate/tensorrt/cuda_impl/cudnn_utils.h"
#include "src/fastertransformer/layers/ms_layers/decoder.h"
#include "src/extendrt/delegate/tensorrt/op/vsl_compress_tensorrt.h"

namespace mindspore::lite {
class DecoderTensorRT : public TensorRTOp {
 public:
  DecoderTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                  const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~DecoderTensorRT() override = default;
  bool IsWeightInputHanledInner() const override {
    return (runtime_->GetTransformerFfnFp16() && runtime_->GetRuntimePrecisionMode() == RuntimePrecisionMode_FP32);
  }
  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;

 private:
  nvinfer1::ITensor *CastTensor(TensorRTContext *ctx, const TensorInfo &ms_tensor, const std::string &op_name);
  int AddVsl(int encoder_input_idx, int decoder_input_idx, int input_number, TensorRTContext *ctx,
             nvinfer1::ITensor **inputTensors, const char *name);
  void CastFfnTensors(fastertransformer::decoderParamRun *params, TensorRTContext *ctx);
  int InitParam(fastertransformer::decoderParamRun *params);
};

constexpr auto DECODER_PLUGIN_NAME{"DecoderPlugin"};
class DecoderPlugin : public TensorRTPlugin {
 public:
  DecoderPlugin(const std::string name, int compute_type, fastertransformer::decoderParamRun params, uint32_t device_id)
      : TensorRTPlugin(name, std::string(DECODER_PLUGIN_NAME), device_id), compute_type_(compute_type) {
    params_ = params;
    params_.attn1.common_param = &params_.common_param;
    params_.attn2.common_param = &params_.common_param;
    params_.ffn_param.common_param = &params_.common_param;
  }

  DecoderPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(DECODER_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    compute_type_ = static_cast<const int *>(fields[0].data)[0];
    params_ = static_cast<const fastertransformer::decoderParamRun *>(fields[1].data)[0];
    params_.attn1.common_param = &params_.common_param;
    params_.attn2.common_param = &params_.common_param;
    params_.ffn_param.common_param = &params_.common_param;
  }

  DecoderPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(DECODER_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &compute_type_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &params_, sizeof(fastertransformer::decoderParamRun));
    params_.attn1.common_param = &params_.common_param;
    params_.attn2.common_param = &params_.common_param;
    params_.ffn_param.common_param = &params_.common_param;
  }

  DecoderPlugin() = delete;

  ~DecoderPlugin() override {}

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept override;

 private:
  std::string name_space_;
  int compute_type_;
  mutable fastertransformer::decoderParamRun params_;
  int num_of_inputs_;
  int num_of_outputs_;
  template <typename T>
  int RunCudaDecoder(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                     const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream,
                     cublasGemmAlgo_t algoId);
};
class DecoderPluginCreater : public TensorRTPluginCreater<DecoderPlugin> {
 public:
  DecoderPluginCreater() : TensorRTPluginCreater(std::string(DECODER_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_DECODER_TENSORRT_H_
