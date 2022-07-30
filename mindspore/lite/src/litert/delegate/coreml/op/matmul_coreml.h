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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_MATMUL_COREML_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_MATMUL_COREML_H_

#include <vector>
#include <string>
#include <memory>
#include "src/litert/delegate/coreml/op/coreml_op.h"
namespace mindspore::lite {
class MatMulCoreMLOp : public CoreMLOp {
 public:
  MatMulCoreMLOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                 const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : CoreMLOp(primitive, in_tensors, out_tensors, name) {}

  int IsSupport() override;

  int InitParams() override;

  int BuildLayer() override;

  std::vector<CoreML::Specification::NeuralNetworkLayer *> GetLayers() override;

  void SetMLOpInOut() override;

  int ConstMatMulWithTransB();

  int ConstMatMulWithoutTransB();

 protected:
  const schema::MatMulFusion *matmul_prim_ = nullptr;
  CoreML::Specification::BatchedMatMulLayerParams *matmul_param_ = nullptr;
  std::unique_ptr<CoreML::Specification::NeuralNetworkLayer> bias_op_ = nullptr;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_MATMUL_COREML_H_
