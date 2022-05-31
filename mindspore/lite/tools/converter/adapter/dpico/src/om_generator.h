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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_OM_GENERATOR_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_OM_GENERATOR_H_

#include <utility>
#include <string>
#include <map>
#include <memory>
#include <vector>
#include "mindapi/ir/common.h"
#include "src/graph_split_info.h"
#include "mapper/op_mapper.h"
#include "op/base_operator.h"

namespace mindspore {
namespace dpico {
class OmGenerator {
 public:
  OmGenerator() = default;
  ~OmGenerator() = default;
  int Run(const api::FuncGraphPtr &func_graph, const Subgraph &sub_graph, int custom_id,
          mapper::ModelCoreInfo *om_model_info, bool use_origin_config);

 private:
  int GenerateAippConfig(const std::string &aipp_cfg_path, const api::AnfNodePtrList &subgraph_inputs);
  int GenerateMapperConfig(const api::FuncGraphPtr &func_graph, const Subgraph &sub_graph, int custom_id,
                           const std::string &cfg);
  int TransformSubGraphInputs(const api::AnfNodePtrList &nodes, std::vector<BaseOperatorPtr> *base_operators);
  int TransformSubGraphCNodes(const api::FuncGraphManagerPtr &manager, const api::CNodePtrList &Cnodes,
                              std::vector<BaseOperatorPtr> *base_operators);
};
}  // namespace dpico
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_OM_GENERATOR_H_
