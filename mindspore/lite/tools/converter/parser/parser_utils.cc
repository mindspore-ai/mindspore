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
#include "tools/converter/parser/parser_utils.h"
#include <memory>
#include "tools/converter/parser/tf_bidirection_gru_cf_fusion.h"
#include "tools/converter/parser/unused_node_remove_pass.h"
#include "tools/converter/parser/conv1d_inout_adjust.h"
#include "tools/converter/parser/inputs_adjust.h"

namespace mindspore::lite {
void GetAllFuncGraph(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *all_func_graphs) {
  if (all_func_graphs->find(func_graph) == all_func_graphs->end()) {
    all_func_graphs->insert(func_graph);
  } else {
    return;
  }

  auto nodes = func_graph->nodes();
  for (auto &node : nodes) {
    if (IsValueNode<FuncGraph>(node)) {
      auto new_fg = (node->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>();
      GetAllFuncGraph(new_fg, all_func_graphs);
    }
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = node->cast<CNodePtr>();
      for (auto &input : cnode->inputs()) {
        if (input->isa<ValueNode>()) {
          if (IsValueNode<FuncGraph>(input)) {
            auto new_fg = (input->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>();
            GetAllFuncGraph(new_fg, all_func_graphs);
          }
        }
      }
    }
  }
}

int PostAdjust(const std::set<FuncGraphPtr> &all_func_graphs) {
  for (auto func_graph : all_func_graphs) {
    {
      auto asylic_optimizer = std::make_shared<opt::GraphOptimizer>();
      auto asylic_pm = std::make_shared<opt::PassManager>("asylic pass manager", false);
      // fuse tf1.x bidirection_gru into GRU, must be placed here because graph is cyclic
      asylic_pm->AddPass(std::make_shared<opt::TfBidirectionGruCfFusion>());
      // remove remaining cyclic nodes
      asylic_pm->AddPass(std::make_shared<opt::UnusedNodeRemovePass>());
      asylic_optimizer->AddPassManager(asylic_pm);
      if (!asylic_optimizer->Optimize(func_graph)) {
        MS_LOG(ERROR) << "gru cf fusion pass failed.";
        ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
        return RET_ERROR;
      }
    }
    auto adjust_input = std::make_shared<InputAdjust>();
    if (!adjust_input->Run(func_graph)) {
      MS_LOG(ERROR) << "adjust input failed.";
      return RET_ERROR;
    }
    // adjust for conv1d
    auto conv1d_adjust = std::make_shared<Conv1DInOutAdjust>();
    if (!conv1d_adjust->Run(func_graph)) {
      MS_LOG(ERROR) << "adjust conv1d failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
