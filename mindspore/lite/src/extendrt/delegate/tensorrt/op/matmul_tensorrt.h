/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_MATMUL_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_MATMUL_TENSORRT_H_
#include <utility>
#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
class MatMulTensorRT : public TensorRTOp {
 public:
  MatMulTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                 const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~MatMulTensorRT() override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;

  int AddInnerOp(TensorRTContext *ctx) override;

  bool IsWeightInputHanledInner() const override { return true; }

  bool HasConst() const override;

 private:
  int PreprocessMatMulInputs(TensorRTContext *ctx, ITensorHelper *matmul_a, ITensorHelper *matmul_b);

  nvinfer1::ITensor *ProcessWeightTensor(TensorRTContext *ctx);

  nvinfer1::ITensor *AddAsMatmul(TensorRTContext *ctx);

  nvinfer1::ITensor *AddAsFullConnect(TensorRTContext *ctx);

  nvinfer1::ITensor *AddBias(TensorRTContext *ctx, nvinfer1::ITensor *input_tensor);

  bool RunFullConnect(TensorRTContext *ctx);

  bool transpose_a_{false};
  bool transpose_b_{false};
  Format out_format_{Format::NCHW};
  ActivationType activation_{ActivationType::NO_ACTIVATION};
  void *weight_ptr_{nullptr};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_MATMUL_TENSORRT_H_
