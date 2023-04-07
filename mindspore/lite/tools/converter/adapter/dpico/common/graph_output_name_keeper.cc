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

#include "common/graph_output_name_keeper.h"
#include <map>
#include <string>
#include <vector>
#include "common/anf_util.h"
#include "common/check_base.h"
#include "common/op_enum.h"
#include "ops/make_tuple.h"
#include "ops/depend.h"
#include "include/registry/converter_context.h"

namespace mindspore {
namespace dpico {
GraphOutputNameKeeper *GraphOutputNameKeeper::GetInstance() {
  static GraphOutputNameKeeper instance;
  return &instance;
}

int GraphOutputNameKeeper::SaveOriginalOutputs(const api::FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_ERROR, "func's input parameter is a nullptr.");
  auto return_cnode = func_graph->get_return();
  MS_CHECK_TRUE_MSG(return_cnode != nullptr, RET_ERROR, "func_graph's return node is invalid.");
  original_outputs_ = return_cnode->inputs();
  original_outputs_.erase(original_outputs_.begin());
  bool change{true};
  while (change) {
    change = false;
    std::vector<api::AnfNodePtr> outputs_tmp;
    for (size_t j = 0; j < original_outputs_.size(); ++j) {
      auto output_node = original_outputs_[j];
      MS_CHECK_TRUE_MSG(output_node != nullptr, RET_ERROR, "existing node is a nullptr.");
      if (dpico::CheckPrimitiveType(output_node, api::MakeShared<ops::MakeTuple>())) {
        auto make_tuple_cnode = output_node->cast<api::CNodePtr>();
        MS_CHECK_TRUE_MSG(make_tuple_cnode != nullptr, RET_ERROR, "make tuple node is invalid.");
        auto make_tuple_inputs = make_tuple_cnode->inputs();
        outputs_tmp.insert(outputs_tmp.end(), make_tuple_inputs.begin() + 1, make_tuple_inputs.end());
        change = true;
        continue;
      }
      if (dpico::CheckPrimitiveType(output_node, api::MakeShared<ops::Depend>())) {
        auto depend_cnode = output_node->cast<api::CNodePtr>();
        MS_CHECK_TRUE_MSG(depend_cnode != nullptr, RET_ERROR, "depend node is invalid.");
        MS_CHECK_TRUE_MSG(depend_cnode->size() == kInputIndex3, RET_ERROR, "depend node's input size should be 3.");
        outputs_tmp.push_back(depend_cnode->input(1));
        change = true;
        continue;
      }
      outputs_tmp.push_back(output_node);
    }
    original_outputs_ = outputs_tmp;
  }

  auto origin_outputs_name = converter::ConverterContext::GetGraphOutputTensorNames();
  for (auto &output_name : origin_outputs_name) {
    const std::string top_name_suffix = "duplicate";
    const size_t max_loop = 1000;
    for (size_t i = 0; i < max_loop; i++) {
      std::string top_name_tmp = output_name + "_" + top_name_suffix + std::to_string(i);
      auto attr = func_graph->get_attr(top_name_tmp);
      if (attr != nullptr) {
        auto op_name = api::GetValue<std::string>(attr);
        ori_output_info_[op_name] = output_name;
      } else {
        break;
      }
    }
  }
  return RET_OK;
}

int GraphOutputNameKeeper::DetermineOmOpInputName(const api::AnfNodePtr &in_node, std::string *input_name) {
  if (original_outputs_.empty()) {
    return RET_OK;
  }
  MS_CHECK_TRUE_MSG(in_node != nullptr && input_name != nullptr, RET_ERROR, "func's input parameter is a nullptr.");
  if (ori_output_info_.find(in_node->fullname_with_scope()) != ori_output_info_.end()) {
    *input_name = ori_output_info_[in_node->fullname_with_scope()];
    return RET_OK;
  }
  auto iter = std::find(original_outputs_.begin(), original_outputs_.end(), in_node);
  if (iter == original_outputs_.end()) {
    return RET_OK;
  }
  auto index = iter - original_outputs_.begin();
  auto origin_outputs_name = converter::ConverterContext::GetGraphOutputTensorNames();
  if (origin_outputs_name.size() <= static_cast<size_t>(index)) {
    return RET_OK;
  }
  input_name->swap(origin_outputs_name[index]);
  return RET_OK;
}

int GraphOutputNameKeeper::DetermineOmOpOutputName(const api::AnfNodePtr &node, std::string *output_name,
                                                   bool is_subgraph_input) {
  MS_CHECK_TRUE_MSG(node != nullptr && output_name != nullptr, RET_ERROR, "func's input parameter is a nullptr.");
  MS_CHECK_TRUE_MSG(!original_outputs_.empty(), RET_ERROR, "has no outputs.");
  if (ori_output_info_.find(node->fullname_with_scope()) != ori_output_info_.end()) {
    *output_name = ori_output_info_[node->fullname_with_scope()];
    return RET_OK;
  }
  auto iter = std::find(original_outputs_.begin(), original_outputs_.end(), node);
  if (iter == original_outputs_.end()) {
    return RET_OK;
  }
  auto index = iter - original_outputs_.begin();
  auto origin_outputs_name = converter::ConverterContext::GetGraphOutputTensorNames();
  if (origin_outputs_name.size() <= static_cast<size_t>(index)) {
    return RET_OK;
  }
  if (!is_subgraph_input) {
    MS_CHECK_TRUE_MSG(om_to_anf_mapper_.find(origin_outputs_name[index]) == om_to_anf_mapper_.end(), RET_ERROR,
                      "find the output has been existed.");
    om_to_anf_mapper_.emplace(origin_outputs_name[index], *output_name);
  }
  output_name->swap(origin_outputs_name[index]);
  return RET_OK;
}

bool GraphOutputNameKeeper::CanKeepOutputNames(const std::vector<std::string> &om_outputs) {
  size_t has_find{0};
  for (const auto &output : om_outputs) {
    if (om_to_anf_mapper_.find(output) != om_to_anf_mapper_.end()) {
      ++has_find;
    }
  }
  return has_find == om_to_anf_mapper_.size();
}

std::string GraphOutputNameKeeper::GetAnfOutputNameFromOm(const std::string &om_out_name) {
  if (om_to_anf_mapper_.find(om_out_name) != om_to_anf_mapper_.end()) {
    return om_to_anf_mapper_[om_out_name];
  }
  return om_out_name;
}
}  // namespace dpico
}  // namespace mindspore
