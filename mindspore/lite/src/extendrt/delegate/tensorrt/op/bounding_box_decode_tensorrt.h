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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_BOUNDING_BOX_DECODE_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_BOUNDING_BOX_DECODE_TENSORRT_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
class BoundingBoxDecodeTensorRT : public TensorRTOp {
 public:
  BoundingBoxDecodeTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                            const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~BoundingBoxDecodeTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};

constexpr auto BOUNDING_BOX_DECODE_PLUGIN_NAME{"BoundingBoxDecodePlugin"};
class BoundingBoxDecodePlugin : public TensorRTPlugin {
 public:
  BoundingBoxDecodePlugin(const std::string name, const std::vector<float> &means, const std::vector<float> &stds,
                          const std::vector<int> &max_shape, float wh_ratio_clip)
      : TensorRTPlugin(name, std::string(BOUNDING_BOX_DECODE_PLUGIN_NAME)),
        means_(means),
        stds_(stds),
        max_shape_(max_shape),
        wh_ratio_clip_(wh_ratio_clip) {}

  BoundingBoxDecodePlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(BOUNDING_BOX_DECODE_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    means_.resize(INPUT_SIZE4);
    means_[0] = static_cast<const float *>(fields[0].data)[0];
    means_[1] = static_cast<const float *>(fields[1].data)[0];
    means_[INPUT_SIZE2] = static_cast<const float *>(fields[INPUT_SIZE2].data)[0];
    means_[INPUT_SIZE3] = static_cast<const float *>(fields[INPUT_SIZE3].data)[0];
    stds_.resize(INPUT_SIZE4);
    stds_[0] = static_cast<const float *>(fields[INPUT_SIZE4].data)[0];
    stds_[1] = static_cast<const float *>(fields[INPUT_SIZE5].data)[0];
    stds_[INPUT_SIZE2] = static_cast<const float *>(fields[INPUT_SIZE6].data)[0];
    stds_[INPUT_SIZE3] = static_cast<const float *>(fields[INPUT_SIZE7].data)[0];
    max_shape_.resize(INPUT_SIZE2);
    max_shape_[0] = static_cast<const int *>(fields[INPUT_SIZE8].data)[0];
    max_shape_[1] = static_cast<const int *>(fields[INPUT_SIZE9].data)[0];
    wh_ratio_clip_ = static_cast<const float *>(fields[INPUT_SIZE10].data)[0];
  }

  BoundingBoxDecodePlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(BOUNDING_BOX_DECODE_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &means_[0], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &means_[1], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &means_[INPUT_SIZE2], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &means_[INPUT_SIZE3], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &stds_[0], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &stds_[1], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &stds_[INPUT_SIZE2], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &stds_[INPUT_SIZE3], sizeof(float));
    DeserializeValue(&serialData, &serialLength, &max_shape_[0], sizeof(int));
    DeserializeValue(&serialData, &serialLength, &max_shape_[1], sizeof(int));
    DeserializeValue(&serialData, &serialLength, &wh_ratio_clip_, sizeof(float));
  }

  BoundingBoxDecodePlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  int getNbOutputs() const noexcept override { return INPUT_SIZE3; }
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
    noexcept override {
    return nvinfer1::DataType::kFLOAT;
  }

 private:
  int RunCudaLogical(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
                     cudaStream_t stream);
  const std::string layer_name_;
  std::string name_space_;
  std::vector<float> means_;
  std::vector<float> stds_;
  std::vector<int> max_shape_;
  float wh_ratio_clip_;
};
class BoundingBoxDecodePluginCreater : public TensorRTPluginCreater<BoundingBoxDecodePlugin> {
 public:
  BoundingBoxDecodePluginCreater() : TensorRTPluginCreater(std::string(BOUNDING_BOX_DECODE_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_BOUNDING_BOX_DECODE_TENSORRT_H_
