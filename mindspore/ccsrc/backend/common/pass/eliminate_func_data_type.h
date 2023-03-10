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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_ELIMINATE_FUNC_TYPE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_ELIMINATE_FUNC_TYPE_H_

#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pattern_engine.h"

// In control flow case, the function data type wil existed in graph to avoid expanding closures.
// The function data type will be processed in mindRT, but it is not supported in kernel graph.
// This pass is used to eliminate function data type for kernel graph.
namespace mindspore::opt {
class EliminateFuncDataType : public PatternProcessPass {
 public:
  explicit EliminateFuncDataType(bool multigraph = true) : PatternProcessPass("eliminate_func_type", multigraph) {
    Init();
  }
  ~EliminateFuncDataType() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  void Init();
  ValueNodePtr constant_;
  AbstractBasePtr constant_abs_;
};
}  // namespace mindspore::opt

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_ELIMINATE_FUNC_TYPE_H_
