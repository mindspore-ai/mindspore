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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_CUSTOM_CREATOR_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_CUSTOM_CREATOR_H_

#include <memory>
#include <string>
#include <vector>
#include "mindapi/ir/func_graph.h"
#include "ops/custom.h"
#include "src/graph_split_info.h"
#include "./op_enum_public.h"
#include "include/errorcode.h"

using mindspore::lite::STATUS;
namespace mindspore {
namespace dpico {
using ModelCoreInfoPtr = std::shared_ptr<mapper::ModelCoreInfo>;
class CustomOpCreator {
 public:
  explicit CustomOpCreator(int custom_id = 0, bool has_unsupported = false)
      : custom_id_(custom_id), has_unsupported_(has_unsupported) {}
  int GetCustomId() const { return custom_id_; }
  ~CustomOpCreator() = default;
  api::CNodePtr CreateCustomOp(const api::FuncGraphPtr &func_graph, Subgraph *subgraph,
                               const ModelCoreInfoPtr &om_model_info);

 private:
  api::ParameterPtr CreateOmParameter(const api::FuncGraphPtr &func_graph, const ModelCoreInfoPtr &om_model_info);
  STATUS SetSubgraphInputOutputDims(Subgraph *subgraph, const api::FuncGraphPtr &func_graph,
                                    const ModelCoreInfoPtr &om_model_info);
  STATUS SetCustomAttrs(const Subgraph &subgraph, const api::FuncGraphPtr &func_graph,
                        const api::SharedPtr<ops::Custom> &prim);
  STATUS SetCustomOutputs(const api::FuncGraphPtr &func_graph, Subgraph *subgraph, const api::CNodePtr &custom_cnode,
                          const ModelCoreInfoPtr &om_model_info);
  STATUS SetCustomSingleOutput(const api::FuncGraphPtr &func_graph, Subgraph *subgraph,
                               const api::CNodePtr &custom_cnode, const ModelCoreInfoPtr &om_model_info,
                               std::vector<std::string> *output_names);
  STATUS SetCustomMultiOutput(const api::FuncGraphPtr &func_graph, Subgraph *subgraph,
                              const api::CNodePtr &custom_cnode, const ModelCoreInfoPtr &om_model_info,
                              std::vector<std::string> *output_names);
  int custom_id_;
  bool has_unsupported_;
};
}  // namespace dpico
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_CUSTOM_CREATOR_H_
