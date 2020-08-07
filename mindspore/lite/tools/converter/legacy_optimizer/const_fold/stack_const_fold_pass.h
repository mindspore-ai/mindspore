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

#ifndef MINDSPORE_PREDICT_STACK_CONST_FOLD_PASS_H
#define MINDSPORE_PREDICT_STACK_CONST_FOLD_PASS_H

#include "converter/optimizer/const_fold/const_fold_pass.h"
#include "securec/include/securec.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace lite {
class StackConstFoldPass : public ConstFoldPass {
 public:
  StackConstFoldPass() : ConstFoldPass(OpT_Stack) {}

  ~StackConstFoldPass() override = default;

  STATUS Run(GraphNode *graphNode) override;

  STATUS CreateOp(SubGraphDefT *subGraph, OpDefT *node) override;

  STATUS DoFold(SubGraphDefT *subGraph, OpDefT *node) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_STACK_CONST_FOLD_PASS_H

