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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_CONCATE_TENSORRT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_CONCATE_TENSORRT_H_
#include <string>
#include <vector>
#include "src/litert/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
class ConcateTensorRT : public TensorRTOp {
 public:
  ConcateTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                  const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                  const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {
    type_ = primitive->value_type();
    axis_ = (type_ == schema::PrimitiveType_Concat ? primitive->value_as_Concat()->axis()
                                                   : primitive->value_as_Stack()->axis());
  }

  ~ConcateTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

 private:
  int PreProcessInputs(TensorRTContext *ctx, nvinfer1::ITensor *trt_input_tensors[]);
  int CheckParams(TensorRTContext *ctx);

  Format out_format_{Format::NHWC};
  bool same_format_{true};
  schema::PrimitiveType type_;
  int axis_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_CONCATE_TENSORRT_H_
