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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_TOPK_TENSORRT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_TOPK_TENSORRT_H_
#include <string>
#include <vector>
#include <map>
#include "src/litert/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
class TopKTensorRT : public TensorRTOp {
 public:
  TopKTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
               const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
               const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~TopKTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

 private:
  int ParseParams(TensorRTContext *ctx);

  int PreprocessInputs(TensorRTContext *ctx, ITensorHelper *topk_input);

  nvinfer1::TopKOperation topk_op_{nvinfer1::TopKOperation::kMAX};
  uint32_t axis_{0};
  int axis_value_{0};
  int32_t top_k_{0};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_TOPK_TENSORRT_H_
