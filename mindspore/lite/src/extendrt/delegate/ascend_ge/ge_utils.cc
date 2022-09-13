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

#include "extendrt/delegate/ascend_ge/ge_utils.h"
#include <set>
#include "ir/manager.h"
#include "tools/converter/adapter/acl/mapper/spatial_node_adapter.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "mindspore/core/ops/op_name.h"

namespace mindspore {
std::string AdjustCnodeName(const PrimitivePtr &prim) {
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return "";
  }
  std::string name = prim->name();
  const std::set<std::string> kAdjustCnodeName = {"Resize", "Conv2dTransposeFusion", "Concat"};
  if (kAdjustCnodeName.find(name) != kAdjustCnodeName.end()) {
    auto val_ptr = prim->GetAttr(ops::kOriginalOpName);
    if (val_ptr != nullptr) {
      auto origin_name = GetValue<std::string>(val_ptr);
      MS_LOG(DEBUG) << "Change old name " << name << " to new name " << origin_name;
      name = origin_name;
    }
  }
  return name;
}

Status RunPrimitiveMapper(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "Deparser graph start.";
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is nullptr.";
    return kCoreFailed;
  }
  std::set<FuncGraphPtr> all_func_graphs = {};
  lite::GetAllFuncGraph(func_graph, &all_func_graphs);
  for (auto graph : all_func_graphs) {
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (!utils::isa<CNodePtr>(node)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr) {
        MS_LOG(ERROR) << "cnode is nullptr.";
        return kCoreFailed;
      }
      auto prim = GetCNodePrimitive(cnode);
      if (prim == nullptr) {
        MS_LOG(ERROR) << "prim is nullptr.";
        return kCoreFailed;
      }
      std::string name = AdjustCnodeName(prim);
      auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(name);
      if (mapper == nullptr) {
        MS_LOG(DEBUG) << "Name: " << name << " not need to mapper.";
        continue;
      }
      MS_LOG(INFO) << "Deparser cnode: " << name;
      auto status = mapper->Mapper(cnode);
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "Deparser primitive failed.";
        return kCoreFailed;
      }
    }
  }
  return kSuccess;
}

Status AdaptGraph(const FuncGraphPtr &func_graph) {
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return kCoreFailed;
  }
  if (RunPrimitiveMapper(func_graph) != kSuccess) {
    MS_LOG(ERROR) << "Run mapper primitive failed.";
    return kCoreFailed;
  }
  if (lite::AdapteSpatialNode(func_graph, manager) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adapter spatial node failed.";
    return kCoreFailed;
  }
  if (lite::acl::DelRedundantParameter(func_graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Delete redundant parameter failed.";
    return kCoreFailed;
  }
  return kSuccess;
}
}  // namespace mindspore
