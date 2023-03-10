/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_REDUCE_SAME_OP_IN_HORIZON_H
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_REDUCE_SAME_OP_IN_HORIZON_H

#include <memory>
#include "ir/anf.h"
#include "include/backend/optimizer/pass.h"
#include "tools/converter/cxx_api/converter_para.h"

namespace mindspore {
namespace opt {
using ConverterParaPtr = std::shared_ptr<ConverterPara>;
class ReduceSameOpInHorizon : public Pass {
 public:
  explicit ReduceSameOpInHorizon(const ConverterParaPtr &param) : Pass("HorizontalFusion"), param_(param) {}
  ~ReduceSameOpInHorizon() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  int Process(const FuncGraphPtr &func_graph);
  bool CheckCNodeIsEqual(const CNodePtr &left, const CNodePtr &right);
  ConverterParaPtr param_{nullptr};
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_REDUCE_SAME_OP_IN_HORIZON_H
