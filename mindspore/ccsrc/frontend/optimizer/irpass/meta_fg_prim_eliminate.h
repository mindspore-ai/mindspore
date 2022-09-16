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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_META_FG_PRIM_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_META_FG_PRIM_ELIMINATE_H_

#include <vector>
#include <memory>

#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"
#include "utils/ms_utils.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/ad/grad.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimJ, C}
class ExpandMetaFgPrim {
 public:
  ExpandMetaFgPrim() = default;
  virtual ~ExpandMetaFgPrim() = default;
  virtual bool CheckIfEmbedMetaFgPrim(const CNodePtr &node) const;
  const std::vector<CNodePtr> &prim_nodes() const { return prim_nodes_; }
  virtual bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) = 0;
  void GetMetaFgPrim(const std::vector<AnfNodePtr> &all_nodes);

 protected:
  std::vector<CNodePtr> prim_nodes_;
  PrimitivePtr prim_{nullptr};
};
using ExpandMetaFGPrimPtr = std::shared_ptr<ExpandMetaFgPrim>;
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_META_FG_PRIM_ELIMINATE_H_
