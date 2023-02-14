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

#include <regex>
#include "frontend/parallel/graph_util/graph_info.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/debug/anf_ir_utils.h"
#include "include/common/debug/draw.h"
#include "utils/ms_context.h"
#include "ir/graph_utils.h"
#include "pipeline/jit/pipeline.h"
#include "frontend/parallel/ops_info/ops_utils.h"

namespace mindspore {
namespace parallel {
std::vector<PrimitivePtr> FindPrimtive(const FuncGraphPtr &graph, const std::string &name) {
  AnfNodePtr ret = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  std::vector<PrimitivePtr> prim_list;
  for (auto &node : all_nodes) {
    if (!IsValueNode<Primitive>(node)) {
      continue;
    }
    ValueNodePtr prim_node_anf = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_node_anf);
    PrimitivePtr node_prim = prim_node_anf->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    if (node_prim->name() == name) {
      (void)prim_list.emplace_back(node_prim);
    }
  }
  return prim_list;
}

void DumpGraph(const FuncGraphPtr &root, const std::string &name) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    static const auto switch_order = (common::GetEnv("MS_DEV_SAVE_GRAPHS_SORT_MODE") == "1");
    if (switch_order) {
      ExportIR(name + ".ir", root);
    } else {
      DumpIR(name + ".ir", root);
    }
    if (context->CanDump(kFully)) {
      draw::Draw(name + ".dot", root);
    }
  }
#endif
}

// Return true if the cnode is in a for-loop and loop_index indicates the i-th loop;
// otherwise return false
bool GetLoopIndexFromCNode(const CNodePtr &cnode, size_t *loop_index) {
  std::regex pattern(CELLLIST_KEYWORD_PATTERN);
  std::smatch result;
  const auto &cnode_fullname = cnode->fullname_with_scope();
  if (std::regex_search(cnode_fullname, result, pattern)) {
    if (result.length() < 2) {
      MS_LOG(EXCEPTION) << "Wrong format of fullname_with_scope: " << cnode_fullname;
    }
    *loop_index = IntToSize(std::stoi(result[1]));
    return true;
  }
  return false;
}

void SetOpsNumToExecutor(size_t num_ops) {
  auto executor = pipeline::GraphExecutorPy::GetInstance();
  executor->SetNumOpsInfo(num_ops);
}
}  // namespace parallel
}  // namespace mindspore
