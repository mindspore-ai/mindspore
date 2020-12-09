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
#include "backend/optimizer/graph_kernel/clean_all_in_once.h"
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"

namespace mindspore {
namespace opt {
namespace {
ShapeVector GetValidShape(const AnfNodePtr &node) {
  // Shape will not contain 1 in head.
  auto shape = GetShape(node);
  ShapeVector valid_shape;
  bool valid = false;
  for (auto s : shape) {
    if (!valid && s == 1) {
      continue;
    }
    valid = true;
    valid_shape.push_back(s);
  }
  return valid_shape;
}

bool IsAtomicCleanNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kAnfPrimitiveIndex));
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!func_graph->has_attr("composite_type")) {
    return false;
  }

  auto ctype_value = func_graph->get_attr("composite_type");
  if (!ctype_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Attribute composite_type should be a string!";
  }
  auto ctype = GetValue<std::string>(ctype_value);
  return ctype == "atomic_clean";
}

std::vector<AnfNodePtrList> SplitVectorByWidth(const AnfNodePtrList &nodes, int width) {
  std::vector<AnfNodePtrList> splitted_nodes;
  if (nodes.empty()) {
    return splitted_nodes;
  }

  int num = (nodes.size() - 1) / width + 1;
  splitted_nodes.resize(num);
  for (size_t i = 0; i < nodes.size(); ++i) {
    splitted_nodes[i / width].push_back(nodes[i]);
  }
  return splitted_nodes;
}
}  // namespace

bool CleanAllInOnce::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto todos = TopoSort(func_graph->get_return());
  std::map<ShapeVector, AnfNodePtrList> clean_map;
  std::for_each(todos.cbegin(), todos.cend(), [&clean_map](const AnfNodePtr &node) {
    if (AnfAlgo::IsGraphKernel(node) && IsAtomicCleanNode(node)) {
      auto valid_shape = GetValidShape(node);
      auto iter = clean_map.find(valid_shape);
      if (iter != clean_map.end()) {
        iter->second.push_back(node);
      } else {
        clean_map.insert({valid_shape, {node}});
      }
    }
  });

  bool changed = false;
  if (!clean_map.empty()) {
    for (auto iter : clean_map) {
      // Do all in once is not good, so do ten in once.
      auto splitted_nodes = SplitVectorByWidth(iter.second, 10);
      for (auto &snodes : splitted_nodes) {
        if (snodes.size() < 2) {
          continue;
        }
        AnfNodePtr clean_all_node;
        std::tie(clean_all_node, std::ignore) = FuseNodesToSubGraph(snodes, func_graph, "clean_all");
        MS_LOG(INFO) << "Add node to clean batch buffers in once(" << clean_all_node->fullname_with_scope()
                     << ") for atomic add!";
        changed = true;
      }
    }
  }

  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
