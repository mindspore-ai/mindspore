/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_ELEMENTWISE_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_ELEMENTWISE_TENSORRT_H_
#include <string>
#include <vector>
#include <map>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
class ElementWiseTensorRT : public TensorRTOp {
 public:
  ElementWiseTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                      const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~ElementWiseTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;

 private:
  nvinfer1::ITensor *AddActivation(TensorRTContext *ctx, nvinfer1::ITensor *in_tensor);

  nvinfer1::IElementWiseLayer *AddFoorMod(TensorRTContext *ctx, nvinfer1::ITensor *x0_trt, nvinfer1::ITensor *x1_trt);
  void LogicalOpChangeInputType(TensorRTContext *ctx, ITensorHelper *x_input, ITensorHelper *y_input);

  int AddConstTensor(TensorRTContext *ctx);

  int BroadcastInputTensors(TensorRTContext *ctx, ITensorHelper *x_input, ITensorHelper *y_input);

  bool SameTensor(nvinfer1::ITensor *trt_tensor, TensorInfo *ms_tensor);

  int PreprocessInputTensors(TensorRTContext *ctx, ITensorHelper *x_input, ITensorHelper *y_input);

  nvinfer1::ElementWiseOperation element_wise_op_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_ELEMENTWISE_TENSORRT_H_
