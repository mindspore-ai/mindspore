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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_MATMUL_TENSORRT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_MATMUL_TENSORRT_H_
#include <utility>
#include <string>
#include <vector>
#include "src/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
class MatMulTensorRT : public TensorRTOp {
 public:
  MatMulTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                 const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name)
      : TensorRTOp(primitive, in_tensors, out_tensors, name) {}

  ~MatMulTensorRT() override = default;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

  int AddInnerOp(nvinfer1::INetworkDefinition *network) override;

 private:
  nvinfer1::MatrixOperation transpose_a_ = nvinfer1::MatrixOperation::kNONE;
  nvinfer1::MatrixOperation transpose_b_ = nvinfer1::MatrixOperation::kNONE;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_MATMUL_TENSORRT_H_
