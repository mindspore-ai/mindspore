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

#include "plugin/device/ascend/optimizer/ir_fusion/parameter_and_transop_fusion.h"
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/core_ops.h"
#include "include/backend/kernel_info.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
const AnfNodePtr ParamTransRoad(const FuncGraphPtr &func_graph, const AnfNodePtr &node, bool first_flag,
                                std::vector<CNodePtr> *trans_road) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    auto op_name = common::AnfAlgo::GetCNodeName(cnode);
    auto manager = func_graph->manager();
    if (manager == nullptr) {
      return nullptr;
    }
    if (op_name == prim::kPrimCast->name() || op_name == prim::kPrimTranspose->name() ||
        op_name == prim::kPrimTransposeD->name() || op_name == prim::kPrimReshape->name() ||
        op_name == prim::kPrimTransData->name()) {
      auto users = manager->node_users()[node];
      if (users.size() > 1 && !first_flag) {
        return nullptr;
      }
      MS_EXCEPTION_IF_NULL(trans_road);
      trans_road->push_back(cnode);
      first_flag = false;
      auto next_node = common::AnfAlgo::GetInputNode(cnode, 0);
      MS_EXCEPTION_IF_NULL(next_node);
      if (next_node->isa<Parameter>() || next_node->isa<ValueNode>()) {
        return next_node;
      }
      return ParamTransRoad(func_graph, next_node, first_flag, trans_road);
    }
  } else if (node->isa<Parameter>() || node->isa<ValueNode>()) {
    return node;
  }
  return nullptr;
}

kernel::KernelBuildInfoPtr GetKernelBuildInfo(const CNodePtr &cast, const string &format, TypeId input_type,
                                              TypeId output_type) {
  MS_EXCEPTION_IF_NULL(cast);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(cast->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto cast_build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(cast_build_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetOutputsFormat({format});
  builder.SetInputsFormat({format});
  builder.SetInputsDeviceType({input_type});
  builder.SetOutputsDeviceType({output_type});
  builder.SetKernelType(cast_build_info->kernel_type());
  builder.SetFusionType(cast_build_info->fusion_type());
  builder.SetProcessor(cast_build_info->processor());
  return builder.Build();
}
}  // namespace

bool ParameterTransOpFusion::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Func graph is nullptr";
    return false;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    return false;
  }
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  bool changed = false;
  constexpr size_t kTransRoadSize = 3;
  for (const auto &node : node_list) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto node_name = common::AnfAlgo::GetCNodeName(cnode);
    if (node_name == prim::kPrimCast->name() || node_name == prim::kPrimTranspose->name() ||
        node_name == prim::kPrimTransposeD->name() || node_name == prim::kPrimReshape->name() ||
        node_name == prim::kPrimTransData->name()) {
      MS_LOG(DEBUG) << "Skip trans op";
      continue;
    }
    size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
    for (size_t input_index = 0; input_index < input_num; input_index++) {
      std::vector<CNodePtr> trans_road;
      bool first_flag = true;
      auto final_node =
        ParamTransRoad(func_graph, common::AnfAlgo::GetInputNode(cnode, input_index), first_flag, &trans_road);
      if (final_node != nullptr && trans_road.size() == kTransRoadSize &&
          common::AnfAlgo::GetCNodeName(trans_road[kIndex0]) == kTransDataOpName &&
          common::AnfAlgo::GetCNodeName(trans_road[kIndex1]) == prim::kPrimCast->name() &&
          common::AnfAlgo::GetCNodeName(trans_road[kIndex2]) == kTransDataOpName) {
        auto cur_transop = trans_road[kIndex0];
        auto format = AnfAlgo::GetOutputFormat(cur_transop, 0);
        auto dtype = AnfAlgo::GetOutputDeviceDataType(cur_transop, 0);
        auto param_format = AnfAlgo::GetOutputFormat(final_node, 0);
        auto param_dtype = AnfAlgo::GetOutputDeviceDataType(final_node, 0);

        auto cast = trans_road[kIndex1];
        if (param_format == format && param_dtype != dtype) {
          AnfAlgo::SetSelectKernelBuildInfo(GetKernelBuildInfo(cast, format, param_dtype, dtype), cast.get());
          (void)manager->Replace(trans_road[kIndex2], final_node);
          (void)manager->Replace(cur_transop, cast);
        }
        changed = true;
      }
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
