/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_HETEROGENEOUS_INSERT_MOVE_TO_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_HETEROGENEOUS_INSERT_MOVE_TO_

#include <map>
#include <string>
#include <vector>

#include "include/backend/common/pass_manager/optimizer.h"

namespace mindspore {
namespace opt {
// Insert MoveAssign and MoveTo node according to parameter info
class InsertMoveTo : public Pass {
  struct OffloadParamInfo {
    CNodePtr user_node_;
    size_t input_index_;
    size_t first_execution_order_;
    size_t last_side_effect_execution_order_;
    bool side_effect_;
    std::string offload_device_;
  };

 public:
  InsertMoveTo() : Pass("insert_move_to") {}
  ~InsertMoveTo() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  void Init(const FuncGraphPtr &graph);

  bool HandleParameter();
  void CollectOffloadedParameter();
  CNodePtr InsertParamMoveTo(const ParameterPtr &parameter, const OffloadParamInfo &info) const;
  void InsertParamMoveAssign(const ParameterPtr &parameter, const OffloadParamInfo &info,
                             const CNodePtr &move_to) const;
  bool BackendInlineNode(const CNodePtr &node);

  std::map<ParameterPtr, std::vector<OffloadParamInfo>> offloaded_parameters_;
  FuncGraphPtr func_graph_{nullptr};
  KernelGraphPtr kernel_graph_{nullptr};
  FuncGraphManagerPtr manager_{nullptr};
  size_t load_lead_dh_{1};
  size_t load_lead_hf_{1};
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_HETEROGENEOUS_INSERT_MOVE_TO_
