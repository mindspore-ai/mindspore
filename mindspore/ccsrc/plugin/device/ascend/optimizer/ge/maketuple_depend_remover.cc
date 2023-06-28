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
#include "plugin/device/ascend/optimizer/ge/maketuple_depend_remover.h"
#include <memory>
#include <vector>
#include <string>
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kDependInputNum = 2;
constexpr size_t kMaketupleInputNum = 0;
// assuming 1 is the index for required input, 2 is the place-holder maketuple
constexpr size_t selected_depend_input = 1;
constexpr auto var = "var";
constexpr auto maketuple = "maketuple";
constexpr auto depend = "depend";
constexpr auto dep_input = "depend_first_input";
constexpr auto kXs = "Xs";
constexpr auto kV = "V";
}  // namespace
bool MakeTupleDependRemover::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &graph,
                                             const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  return true;
}

AnfNodePtr SelectInput(const PatternMap &m, const AnfNodePtr &default_node) {
  MS_EXCEPTION_IF_NULL(default_node);
  const auto &depend_node = m.Get(depend);
  MS_EXCEPTION_IF_NULL(depend_node);
  auto depend_cnode = depend_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(depend_cnode);
  const auto selected_input = depend_cnode->input(selected_depend_input);
  return selected_input;
}

void MakeTupleDependRemover::DefineSrcPattern(SrcPattern *src_pattern) {
  (void)(*src_pattern)
    .AddVar(kV)
    .AddSeqVar(kXs)
    .AddCNode(var, {kV, kXs})
    .AddCNode(maketuple, {prim::kPrimMakeTuple})
    .AddCNode(depend, {prim::kPrimDepend, var, maketuple});
}

void MakeTupleDependRemover::DefineDstPattern(DstPattern *dst_pattern) {
  (void)(*dst_pattern).AddCNode(dep_input, {kV, kXs}, SelectInput);
}
}  // namespace opt
}  // namespace mindspore
