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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_BATCH_NORM_TENSORRT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_BATCH_NORM_TENSORRT_H_
#include <string>
#include <vector>
#include "src/litert/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
constexpr int BETA_INDEX = 2;
constexpr int MEAN_INDEX = 3;
constexpr int VAR_INDEX = 4;

class BatchNormTensorRT : public TensorRTOp {
 public:
  BatchNormTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                    const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                    const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~BatchNormTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

 private:
  int RunAsTrtOps(TensorRTContext *ctx, ITensorHelper helper);

  float epsilon_{0.0f};
  nvinfer1::ITensor *gamma_{nullptr};
  nvinfer1::ITensor *beta_{nullptr};
  nvinfer1::ITensor *mean_{nullptr};
  nvinfer1::ITensor *var_{nullptr};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_BATCH_NORM_TENSORRT_H_
