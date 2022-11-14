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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_OP_ROIALIGN_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_OP_ROIALIGN_PLUGIN_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
constexpr auto ROIALIGN_PLUGIN_NAME{"ROIAlignPlugin"};
class ROIAlignTensorRT : public TensorRTOp {
 public:
  ROIAlignTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                   const std::vector<TensorInfo> &out_tensors, const std::string &name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~ROIAlignTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};
class ROIAlignPlugin : public TensorRTPlugin {
 public:
  explicit ROIAlignPlugin(const std::string name, int pooled_height, int pooled_width, float spatial_scale,
                          int sample_num, int roi_end_mode, int channel, int height, int width, int roi_rows,
                          int roi_cols)
      : TensorRTPlugin(name, std::string(ROIALIGN_PLUGIN_NAME)),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        spatial_scale_(spatial_scale),
        sample_num_(sample_num),
        roi_end_mode_(roi_end_mode),
        channel_(channel),
        height_(height),
        width_(width),
        roi_rows_(roi_rows),
        roi_cols_(roi_cols) {}

  ROIAlignPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(ROIALIGN_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    pooled_height_ = static_cast<const int *>(fields[0].data)[0];
    pooled_width_ = static_cast<const int *>(fields[1].data)[0];
    spatial_scale_ = static_cast<const float *>(fields[INPUT_SIZE2].data)[0];
    sample_num_ = static_cast<const int *>(fields[INPUT_SIZE3].data)[0];
    roi_end_mode_ = static_cast<const int *>(fields[INPUT_SIZE4].data)[0];
    channel_ = static_cast<const int *>(fields[INPUT_SIZE5].data)[0];
    height_ = static_cast<const int *>(fields[INPUT_SIZE6].data)[0];
    width_ = static_cast<const int *>(fields[INPUT_SIZE7].data)[0];
    roi_rows_ = static_cast<const int *>(fields[INPUT_SIZE8].data)[0];
    roi_cols_ = static_cast<const int *>(fields[INPUT_SIZE9].data)[0];
  }

  ROIAlignPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(ROIALIGN_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &pooled_height_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &pooled_width_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &spatial_scale_, sizeof(float));
    DeserializeValue(&serialData, &serialLength, &sample_num_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &roi_end_mode_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &channel_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &height_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &width_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &roi_rows_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &roi_cols_, sizeof(int));
  }

  ROIAlignPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept override;

 private:
  int RunCudaROIAlign(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
                      cudaStream_t stream);
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
  int sample_num_{INPUT_SIZE2};
  int roi_end_mode_{1};
  int channel_;
  int height_;
  int width_;
  int roi_rows_;
  int roi_cols_;
};
class ROIAlignPluginCreater : public TensorRTPluginCreater<ROIAlignPlugin> {
 public:
  ROIAlignPluginCreater() : TensorRTPluginCreater(std::string(ROIALIGN_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_OP_ROIALIGN_PLUGIN_H_
