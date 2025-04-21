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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_DPICO_COMMON_GRAPH_OUTPUT_NAME_KEEPER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_DPICO_COMMON_GRAPH_OUTPUT_NAME_KEEPER_H

#include <map>
#include <string>
#include <vector>
#include "mindapi/ir/func_graph.h"

namespace mindspore {
namespace dpico {
class GraphOutputNameKeeper {
 public:
  static GraphOutputNameKeeper *GetInstance();
  int SaveOriginalOutputs(const api::FuncGraphPtr &func_graph);
  void RecycleResource() {
    original_outputs_.clear();
    om_to_anf_mapper_.clear();
  }
  void ResetOutputNameMapper() { om_to_anf_mapper_.clear(); }
  int DetermineOmOpInputName(const api::AnfNodePtr &in_node, std::string *input_name);
  int DetermineOmOpOutputName(const api::AnfNodePtr &node, std::string *output_name, bool is_subgraph_input = false);
  bool CanKeepOutputNames(const std::vector<std::string> &om_outputs);
  std::string GetAnfOutputNameFromOm(const std::string &om_out_name);

 private:
  GraphOutputNameKeeper() = default;
  ~GraphOutputNameKeeper() = default;
  api::AnfNodePtrList original_outputs_;
  std::map<std::string, std::string> om_to_anf_mapper_;
  std::map<std::string, std::string> ori_output_info_;
};
}  // namespace dpico
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_DPICO_COMMON_GRAPH_OUTPUT_NAME_KEEPER_H
