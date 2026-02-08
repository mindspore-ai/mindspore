/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MERGE_ADDN_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MERGE_ADDN_H_

#include <vector>
#include <algorithm>
#include <memory>

#include "frontend/optimizer/irpass.h"
#include "primitive/sequence_ops.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {PrimAddN, {prim::kPrimMakeTuple, {PrimAddN, {prim::kPrimMakeTuple, Xs}}, Ys}} ->
// {{PrimAddNClass}, {prim::kPrimMakeTuple, Xs, Ys}}
// {PrimAddN, {prim::kPrimMakeTuple, Ys, {PrimAddN, {prim::kPrimMakeTuple, Xs}}}} ->
// {{PrimAddNClass}, {prim::kPrimMakeTuple, Ys, Xs}}
class MergeAddN : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override;
  void Visit(const CNodePtr &cnode) override;

  bool is_unique(const AnfNodePtr &node);

  void Reset();

  void UpdateDumpFlag(const AnfNodePtr &node);

 private:
  FuncGraphManagerPtr mng_{nullptr};
  std::vector<AnfNodePtr> Xs_{}, Ys_{}, args_{}, addn_nodes_{};
  bool is_inner_{false}, is_outer_{false}, is_match_{false};
};

// {PrimAddN, {kPrimMakeTuple, Xs}}
class AddNZeroFilter : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
  bool IsReshapeZeros(const AnfNodePtr &node);

  void Visit(const CNodePtr &cnode) override;

  void Reset();

 private:
  std::vector<AnfNodePtr> filtered_Xs_{}, Xs_{};
  bool has_zero_like_{false};
};

// {PrimAddN, {kPrimMakeTuple, Xs}}
class AddNCheckDump : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  void Visit(const CNodePtr &cnode) override;

  void Reset();

 private:
  bool set_dump_{false};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_MERGE_ADDN_H_
