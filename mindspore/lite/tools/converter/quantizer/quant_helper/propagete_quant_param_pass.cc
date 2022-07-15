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
#include "tools/converter/quantizer/quant_helper/propagete_quant_param_pass.h"
#include <unordered_map>
#include <memory>
#include <set>
#include <string>
#include <list>
#include <map>
#include "src/common/log_adapter.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/node_util.h"
#include "tools/lite_exporter/fetch_content.h"
#include "tools/optimizer/common/format_utils.h"

namespace mindspore::lite::quant {
namespace {
const std::set<PrimitivePtr> kSelfInferOperator = {prim::kPrimReshape, prim::kPrimTranspose};
}  // namespace
int PropagateQuantParamPass::PropagateSelf(const CNodePtr &cnode, bool forward) {
  if (CheckNodeInSet(cnode, kSelfInferOperator)) {
    auto curr_quant_holder = GetCNodeQuantHolder(cnode);
    CHECK_NULL_RETURN(curr_quant_holder);
    auto curr_input_quant_params = curr_quant_holder->get_input_quant_params();
    auto curr_output_quant_params = curr_quant_holder->get_output_quant_params();
    if (curr_input_quant_params.empty() || curr_output_quant_params.empty()) {
      return RET_OK;
    }
    std::string primitive_name;
    auto ret = opt::GetPrimitiveType(cnode, &primitive_name);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Get primitive type failed.";
      return ret;
    }
    MS_LOG(INFO) << primitive_name << ":" << cnode->fullname_with_scope() << " propagate self";
    if (forward) {
      // output->input
      auto curr_input_quant_param = curr_input_quant_params.at(0);
      auto curr_output_quant_param = curr_output_quant_params.at(0);
      bool input_valid = curr_input_quant_param.empty() || !curr_input_quant_param.at(0).inited;
      bool output_valid = !curr_output_quant_param.empty() && curr_output_quant_param.at(0).inited;
      if (!input_valid && output_valid) {
        curr_quant_holder->set_input_quant_param(0, curr_output_quant_param);
      }
    } else {
      // input->output
      auto curr_input_quant_param = curr_input_quant_params.at(0);
      auto curr_output_quant_param = curr_output_quant_params.at(0);
      bool input_valid = !curr_input_quant_param.empty() && curr_input_quant_param.at(0).inited;
      bool output_valid = curr_output_quant_param.empty() || curr_output_quant_param.at(0).inited;
      if (input_valid && !output_valid) {
        curr_quant_holder->set_output_quant_param(0, curr_input_quant_param);
      }
    }
  }
  return RET_OK;
}

int PropagateQuantParamPass::ForwardPropagate(const std::list<CNodePtr> &nodes) {
  for (const auto &cnode : nodes) {
    auto inputs = cnode->inputs();
    if (IsGraphInput(cnode) || opt::IsSpecialType(cnode)) {
      continue;
    }
    // Infer quant param with forward (output->input).
    auto curr_quant_holder = GetCNodeQuantHolder(cnode);
    CHECK_NULL_RETURN(curr_quant_holder);
    auto curr_input_quant_params = curr_quant_holder->get_input_quant_params();
    for (size_t i = 0; i < curr_input_quant_params.size(); ++i) {
      auto quant_param = curr_input_quant_params.at(i);
      if (!quant_param.empty() && quant_param.at(0).inited) {
        continue;
      }
      auto index = i + kPrimOffset;
      if (!cnode->input(index)->isa<mindspore::CNode>()) {
        continue;
      }

      // Expand Nodes.
      auto origin_inputs = cnode->inputs();
      auto ret = RemoveIfDepend(cnode);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << cnode->fullname_with_scope() << " RemoveIfDepend failed.";
        return ret;
      }
      ret = RemoveIfMakeTuple(cnode);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << cnode->fullname_with_scope() << " RemoveIfMakeTuple failed.";
        return ret;
      }
      opt::RemoveIfMonad(cnode);
      auto before_cnode_map = opt::GetRealCertainVarInput(cnode, index);
      cnode->set_inputs(origin_inputs);

      auto before_cnode = before_cnode_map.first;
      size_t before_out_index = before_cnode_map.second;
      auto before_quant_holder = GetCNodeQuantHolder(before_cnode);
      CHECK_NULL_RETURN(before_quant_holder);
      auto before_output_quant_param = before_quant_holder->get_output_quant_params();
      if (before_output_quant_param.size() > before_out_index) {
        MS_LOG(INFO) << before_cnode->fullname_with_scope() << " forward propagate to " << cnode->fullname_with_scope();
        curr_quant_holder->set_input_quant_param(i, before_output_quant_param.at(before_out_index));
      }
    }
    // Infer quant param with self.
    auto ret = PropagateSelf(cnode, true);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " propagate self failed.";
      return ret;
    }
  }
  return RET_OK;
}

int PropagateQuantParamPass::BackwardPropagate(const std::list<CNodePtr> &nodes) {
  auto ret = FindNodeDepends(nodes, &node_depends_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Find node depends failed.";
    return ret;
  }
  for (auto iter = nodes.rbegin(); iter != nodes.rend(); iter++) {
    auto cnode = *iter;
    auto inputs = cnode->inputs();
    if (IsGraphInput(cnode) || opt::IsSpecialType(cnode)) {
      continue;
    }
    // Infer quant param with forward (output<-input).
    auto curr_quant_holder = GetCNodeQuantHolder(cnode);
    CHECK_NULL_RETURN(curr_quant_holder);
    auto curr_output_quant_params = curr_quant_holder->get_output_quant_params();
    for (size_t i = 0; i < curr_output_quant_params.size(); ++i) {
      auto quant_param = curr_output_quant_params.at(i);
      if (!quant_param.empty() && quant_param.at(0).inited) {
        continue;
      }
      // output<-input
      auto depend_iter = node_depends_.find(cnode);
      if (depend_iter == node_depends_.end()) {
        MS_LOG(WARNING) << cnode->fullname_with_scope() << " find depend failed.";
        continue;
      }
      if (depend_iter->second.backwards.size() == 1) {
        auto input_cnode = depend_iter->second.backwards.at(0);
        // find input index
        size_t index = 0;
        for (size_t j = 1; j < input_cnode->inputs().size(); j++) {
          if (input_cnode->input(j) == cnode) {
            index = j - kPrimOffset;
          }
        }
        auto input_quant_holder = GetCNodeQuantHolder(input_cnode);
        auto input_quant_params = input_quant_holder->get_input_quant_params();
        if (input_quant_params.size() > index && !input_quant_params.at(index).empty() &&
            input_quant_params.at(index).at(0).inited) {
          MS_LOG(INFO) << input_cnode->fullname_with_scope() << " backward propagate to "
                       << cnode->fullname_with_scope();
          curr_quant_holder->set_output_quant_param(i, input_quant_params.at(index));
        }
      } else if (depend_iter->second.backwards.size() > 1) {
        if (cnode->isa<abstract::AbstractTuple>() && cnode->cast<abstract::AbstractTuplePtr>()->size() > 1) {
          MS_LOG(ERROR) << "Single output, multiple references.";
        } else {
          MS_LOG(ERROR) << "Support for multi output.";
        }
        MS_LOG(ERROR) << cnode->fullname_with_scope()
                      << " Only Support single output. output size is:" << depend_iter->second.backwards.size();
        return RET_ERROR;
      }
    }
    // Infer quant param with self.
    ret = PropagateSelf(cnode, true);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " propagate self failed.";
      return ret;
    }
  }
  return RET_OK;
}

int PropagateQuantParamPass::FindNodeDepends(const std::list<CNodePtr> &nodes,
                                             std::map<CNodePtr, DependNodes> *node_depends) {
  for (const auto &cnode : nodes) {
    if (opt::IsSpecialType(cnode)) {
      continue;
    }
    // Expand Nodes.
    auto origin_inputs = cnode->inputs();
    auto ret = RemoveIfDepend(cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " RemoveIfDepend failed.";
      return ret;
    }
    ret = RemoveIfMakeTuple(cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " RemoveIfMakeTuple failed.";
      return ret;
    }
    opt::RemoveIfMonad(cnode);

    for (size_t i = 1; i < cnode->size(); i++) {
      // Associate the input node with the forward output
      if (!cnode->input(i)->isa<mindspore::CNode>()) {
        continue;
      }
      auto input_cnode = cnode->input(i)->cast<CNodePtr>();
      auto iter = node_depends->find(input_cnode);
      if (iter != node_depends->end()) {
        iter->second.backwards.push_back(cnode);
      }
    }
    node_depends->insert({cnode, {cnode->inputs(), {}}});
    cnode->set_inputs(origin_inputs);
  }
  return RET_OK;
}

int PropagateQuantParamPass::Propagate() {
  CHECK_NULL_RETURN(func_graph_);
  auto nodes = func_graph_->GetOrderedCnodes();

  auto ret = ForwardPropagate(nodes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Propagate forward failed.";
    return ret;
  }
  ret = BackwardPropagate(nodes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Propagate backed failed.";
    return ret;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
