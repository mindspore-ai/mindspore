/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "ops/resize_bilinear_v2.h"

namespace mindspore::lite {
class ResizeBilinearV2TensorRT : public TensorRTOp {
 public:
  ResizeBilinearV2TensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                           const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~ResizeBilinearV2TensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  // bool IsWeightInputHanledInner() const override { return true; }

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;

 private:
  nvinfer1::ITensor *RunTensorRT(TensorRTContext *ctx, nvinfer1::ITensor *resize_in_tensor);

  int SetOutputDims(TensorRTContext *ctx, nvinfer1::ITensor *resize_in_tensor, nvinfer1::IResizeLayer *resize_layer);

  int SetParams(nvinfer1::IResizeLayer *resize_layer);

  std::shared_ptr<ops::ResizeBilinearV2> resize_op_{nullptr};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_RESIZE_TENSORRT_H_
