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

#include "backend/optimizer/ascend/ir_fusion/parameter_and_transop_fusion.h"
#include <memory>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "base/core_ops.h"
#include "runtime/device/kernel_info.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
const AnfNodePtr ParamTransRoad(const FuncGraphPtr &func_graph, const AnfNodePtr &node, bool first_flag,
                                std::vector<CNodePtr> *trans_road) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "nullptr";
    return nullptr;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    auto op_name = AnfAlgo::GetCNodeName(cnode);
    auto manager = func_graph->manager();
    if (manager == nullptr) {
      return nullptr;
    }
    if (op_name == prim::kPrimCast->name() || op_name == prim::kPrimTranspose->name() ||
        op_name == prim::kPrimReshape->name() || op_name == kTransDataOpName) {
      auto users = manager->node_users()[node];
      if (users.size() > 1 && !first_flag) {
        return nullptr;
      }
      trans_road->push_back(cnode);
      first_flag = false;
      auto next_node = AnfAlgo::GetInputNode(cnode, 0);
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
  for (auto node : node_list) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto node_name = AnfAlgo::GetCNodeName(cnode);
    if (node_name == prim::kPrimCast->name() || node_name == prim::kPrimTranspose->name() ||
        node_name == prim::kPrimReshape->name() || node_name == kTransDataOpName) {
      MS_LOG(DEBUG) << "Skip trans op";
      continue;
    }
    size_t input_num = AnfAlgo::GetInputTensorNum(cnode);
    for (size_t input_index = 0; input_index < input_num; input_index++) {
      std::vector<CNodePtr> trans_road;
      bool first_flag = true;
      auto final_node = ParamTransRoad(func_graph, AnfAlgo::GetInputNode(cnode, input_index), first_flag, &trans_road);
      if (final_node != nullptr && trans_road.size() == 3 && AnfAlgo::GetCNodeName(trans_road[0]) == kTransDataOpName &&
          AnfAlgo::GetCNodeName(trans_road[1]) == prim::kPrimCast->name() &&
          AnfAlgo::GetCNodeName(trans_road[2]) == kTransDataOpName) {
        auto cur_transop = trans_road[0];
        auto format = AnfAlgo::GetOutputFormat(cur_transop, 0);
        auto dtype = AnfAlgo::GetOutputDeviceDataType(cur_transop, 0);
        auto param_format = AnfAlgo::GetOutputFormat(final_node, 0);
        auto param_dtype = AnfAlgo::GetOutputDeviceDataType(final_node, 0);

        auto cast = trans_road[1];
        if (param_format == format && param_dtype != dtype) {
          AnfAlgo::SetSelectKernelBuildInfo(GetKernelBuildInfo(cast, format, param_dtype, dtype), cast.get());
          manager->Replace(trans_road[2], final_node);
          manager->Replace(cur_transop, cast);
        }
        changed = true;
      }
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
