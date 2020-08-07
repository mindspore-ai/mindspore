/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PREDICT_CONST_FOLD_PASS_H
#define MINDSPORE_PREDICT_CONST_FOLD_PASS_H

#include <vector>
#include "mindspore/lite/tools/converter/optimizer.h"
#include "include/tensor.h"
#include "utils/log_adapter.h"
#include "converter/common/converter_op_utils.h"
#include "securec/include/securec.h"
#include "src/op.h"

namespace mindspore {
namespace lite {
class ConstFoldPass : public NodePass {
 public:
  explicit ConstFoldPass(schema::PrimitiveType opType) : opType(opType) {}

  ~ConstFoldPass() override = default;

  STATUS Run(GraphNode *graphNode) override;

 protected:
  bool IsFoldable(SubGraphDefT *subGraph, OpDefT *node);

  virtual STATUS CreateOp(SubGraphDefT *subGraph, OpDefT *node) = 0;

  virtual STATUS DoFold(SubGraphDefT *subGraph, OpDefT *node) = 0;

 protected:
  OpDef *PackOpDefT(const OpDefT *opDefT);

  Tensor *CopyTensorDefT2Tensor(const TensorDefT *tensorDefT, bool needCopyData = true);

  STATUS CopyTensor2TensorDefT(const Tensor *tensor, TensorDefT *tensorDefT);

  void FreeTensors();

 protected:
  schema::PrimitiveType opType;
  TensorDefT *outputTensor = nullptr;
  std::vector<Tensor *> inputs;
  std::vector<Tensor *> outputs;
  OpBase *op = nullptr;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_CONST_FOLD_PASS_H
