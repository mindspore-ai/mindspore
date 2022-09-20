/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_RESIZE_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_RESIZE_TENSORRT_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"
#include "ops/resize.h"

namespace mindspore::lite {
class ResizeTensorRT : public TensorRTOp {
 public:
  ResizeTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                 const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~ResizeTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  bool IsWeightInputHanledInner() const override { return true; }

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;

 private:
  nvinfer1::ITensor *RunPlugin(TensorRTContext *ctx, nvinfer1::ITensor *resize_in_tensor);
  nvinfer1::ITensor *RunTensorRT(TensorRTContext *ctx, nvinfer1::ITensor *resize_in_tensor);

  int SetOutputDims(TensorRTContext *ctx, nvinfer1::ITensor *resize_in_tensor, nvinfer1::IResizeLayer *resize_layer);

  void ParseValueFromShapeTensor(TensorRTContext *ctx, const TensorInfo &shape_value_tensor,
                                 std::vector<float> *out_shape);

  bool IsScaleOutputDim(const std::vector<int64_t> &in_shape, const std::vector<int64_t> &out_shape,
                        const std::vector<float> &shape_tensor_val);

  int SetParams(nvinfer1::IResizeLayer *resize_layer);

  std::shared_ptr<ops::Resize> resize_op_{nullptr};
  int mask1_[4]{1, 1, 0, 0};
  int mask2_[4]{0, 0, 0, 0};
};

constexpr auto RESIZELINEAR2D_PLUGIN_NAME{"ResizeLinear2DPlugin"};
class ResizeLinear2DPlugin : public TensorRTPlugin {
 public:
  ResizeLinear2DPlugin(const std::string name, int resize_h, int resize_w, bool using_half_pixel, uint32_t device_id)
      : TensorRTPlugin(name, std::string(RESIZELINEAR2D_PLUGIN_NAME), device_id),
        resize_h_(resize_h),
        resize_w_(resize_w),
        using_half_pixel_(using_half_pixel) {}

  ResizeLinear2DPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(RESIZELINEAR2D_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    resize_h_ = static_cast<const int *>(fields[0].data)[0];
    resize_w_ = static_cast<const int *>(fields[1].data)[0];
    using_half_pixel_ = static_cast<const bool *>(fields[2].data)[0];
  }

  ResizeLinear2DPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(RESIZELINEAR2D_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &resize_h_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &resize_w_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &using_half_pixel_, sizeof(bool));
  }

  ResizeLinear2DPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;

 private:
  int RunCudaResizeLinear2D(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                            void *const *outputs, cudaStream_t stream);
  int resize_h_;
  int resize_w_;
  bool using_half_pixel_;
  const std::string layer_name_;
  std::string name_space_;
};
class ResizeLinear2DPluginCreater : public TensorRTPluginCreater<ResizeLinear2DPlugin> {
 public:
  ResizeLinear2DPluginCreater() : TensorRTPluginCreater(std::string(RESIZELINEAR2D_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_RESIZE_TENSORRT_H_
