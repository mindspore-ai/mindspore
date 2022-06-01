/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_GRAPH_SPLIT_INFO_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_GRAPH_SPLIT_INFO_H_

#include <utility>
#include <vector>
#include <map>
#include "mindapi/ir/common.h"

using ShapeVector = std::vector<int64_t>;
namespace mindspore {
namespace dpico {
enum OmNetType : int { kCnn = 0, kRoi = 1, kRecurrent = 2 };

struct Subgraph {
  int32_t graph_id;
  bool is_supported;
  OmNetType om_net_type;
  api::CNodePtrList cnodes;
  std::vector<ShapeVector> inputs_dims;
  std::vector<ShapeVector> outputs_dims;
  std::vector<int> outputs_format;
  Subgraph(size_t input_id, bool input_flag, OmNetType input_type, api::CNodePtrList input_cnodes)
      : graph_id(input_id), is_supported(input_flag), om_net_type(input_type), cnodes(std::move(input_cnodes)) {}
};

struct GraphSplitInfo {
  size_t num_of_segments{0};
  std::map<api::FuncGraphPtr, std::vector<Subgraph>> subgraphs_map;
};
}  // namespace dpico
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_GRAPH_SPLIT_INFO_H_
