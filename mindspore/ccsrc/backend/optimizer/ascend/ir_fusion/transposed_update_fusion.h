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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_TRANSPOSED_UPDATE_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_TRANSPOSED_UPDATE_FUSION_H_
#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "backend/optimizer/common/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "backend/optimizer/common/helper.h"
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/ascend/ascend_helper.h"

namespace mindspore {
namespace opt {
class TransposedUpdateFusion : public PatternProcessPass {
 public:
  explicit TransposedUpdateFusion(bool multigraph = true, const string &name = "transposed_update_fusion")
      : PatternProcessPass(name, multigraph),
        kernel_select_(std::make_shared<KernelSelect>()),
        tbe_kernel_query_(std::make_shared<TbeKernelQuery>()) {}
  ~TransposedUpdateFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 protected:
  CNodePtr DoSplit(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
  bool IsFormatInvaild(const AnfNodePtr &node) const;
  KernelSelectPtr kernel_select_;
  TbeKernelQueryPtr tbe_kernel_query_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_TRANSPOSED_UPDATE_FUSION_H_
