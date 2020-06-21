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

#include "optimizer/graph_kernel_reuse.h"
#include <vector>
#include <algorithm>
#include <string>
#include "./common.h"
#include "utils/graph_utils.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {

bool GraphKernelReuse::CompareNode(const AnfNodePtr a, const AnfNodePtr b) {
  if (a->abstract() && b->abstract()) {
    auto a_type = a->abstract()->GetTypeTrack();
    auto b_type = b->abstract()->GetTypeTrack();

    if (a_type != b_type) {
      return false;
    }

    auto a_shape = a->abstract()->GetShapeTrack();
    auto b_shape = b->abstract()->GetShapeTrack();
    if (a_shape != nullptr && a_shape == b_shape) {
      return true;
    }

    if (a_shape != nullptr && b_shape != nullptr && a_shape->isa<abstract::Shape>() &&
        b_shape->isa<abstract::Shape>()) {
      return a_shape->cast<abstract::ShapePtr>()->shape() == b_shape->cast<abstract::ShapePtr>()->shape();
    }
  }
  return false;
}

bool GraphKernelReuse::DoReplace(const FuncGraphManagerPtr manager) {
  bool changed = false;
  auto fgs = manager->func_graphs();
  for (FuncGraphPtr &fg : fgs) {
    if (!fg->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
      continue;
    }
    std::string key = GetValue<std::string>(fg->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
    if (graph_kernel_ops.find(key) != graph_kernel_ops.end()) {
      if (find(graph_kernel_ops[key].begin(), graph_kernel_ops[key].end(), fg) == graph_kernel_ops[key].end()) {
        FuncGraphPtr new_fg = nullptr;
        for (auto &cfg : graph_kernel_ops[key]) {
          // If two graphs have different size then continue
          auto fg_topos = TopoSort(fg->get_return());
          auto cfg_topos = TopoSort(cfg->get_return());
          if (fg_topos.size() != cfg_topos.size()) {
            continue;
          }

          // Compare const tensor
          bool has_same = true;
          for (size_t i = 0; i < fg_topos.size(); ++i) {
            if (IsValueNode<tensor::Tensor>(fg_topos[i])) {
              if (!IsValueNode<tensor::Tensor>(cfg_topos[i])) {
                has_same = false;
                break;
              }

              auto tensor1 = GetValueNode<tensor::TensorPtr>(fg_topos[i]);
              auto tensor2 = GetValueNode<tensor::TensorPtr>(cfg_topos[i]);
              if (!tensor1->ValueEqual(*tensor2)) {
                has_same = false;
                break;
              }
            }
          }

          if (!has_same) {
            continue;
          }

          auto fg_input = fg->parameters();
          auto cfg_input = cfg->parameters();
          if (fg_input.size() != cfg_input.size()) {
            continue;
          }
          // Compare input
          for (size_t i = 0; i < fg_input.size(); ++i) {
            if (!CompareNode(fg_input[i], cfg_input[i])) {
              has_same = false;
              break;
            }
          }
          if (!has_same) {
            continue;
          }

          // Compare output
          if (!CompareNode(fg->output(), cfg->output())) {
            continue;
          }

          // Find reusable fg
          new_fg = cfg;
          break;
        }

        if (new_fg != nullptr) {
          // Replace current fg with existing fg
          auto users = fg->func_graph_cnodes_index();
          for (auto &iter : users) {
            auto cnode = iter.first->first->cast<CNodePtr>();
            auto new_input = cnode->inputs();
            auto main_graph = cnode->func_graph();
            MS_EXCEPTION_IF_NULL(main_graph);
            if (IsPrimitiveCNode(cnode, prim::kPrimPartial)) {
              new_input[1] = NewValueNode(new_fg);
            } else {
              new_input[0] = NewValueNode(new_fg);
            }
            auto new_cnode = main_graph->NewCNode(new_input);
            manager->Replace(iter.first->first, new_cnode);
            changed = true;
          }

        } else {
          // Add current fg to map
          graph_kernel_ops[key].push_back(fg);
        }
      }
    } else {
      graph_kernel_ops[key] = {fg};
    }
  }

  return changed;
}

bool GraphKernelReuse::ReuseGraphKernel(const FuncGraphPtr root, const FuncGraphManagerPtr manager) {
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(root);

  return DoReplace(manager);
}

}  // namespace opt
}  // namespace mindspore
