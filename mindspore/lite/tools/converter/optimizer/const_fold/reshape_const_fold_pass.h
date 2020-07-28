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

#ifndef MINDSPORE_PREDICT_RESHAPE_CONST_FOLD_PASS_H
#define MINDSPORE_PREDICT_RESHAPE_CONST_FOLD_PASS_H

#include <vector>
#include "converter/optimizer/const_fold/const_fold_pass.h"

namespace mindspore {
namespace lite {
class ReshapeConstFoldPass : public ConstFoldPass {
 public:
  ReshapeConstFoldPass() : ConstFoldPass(OpT_Reshape) {}

  ~ReshapeConstFoldPass() override = default;

  STATUS Run(GraphNode *graphNode) override;

  STATUS CreateOp(SubGraphDefT *subGraph, OpDefT *node) override;

  STATUS DoFold(SubGraphDefT *subGraph, OpDefT *node) override;

 private:
  STATUS CalNewShape(const TensorDefT &inTensor, std::vector<int64_t> &outShape);
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_RESHAPE_CONST_FOLD_PASS_H
