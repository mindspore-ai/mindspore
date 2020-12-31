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

#include "backend/optimizer/graph_kernel/reorder_ops.h"
#include <memory>
#include <vector>
#include <string>
#include "base/core_ops.h"
#include "utils/utils.h"
#include "utils/log_adapter.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
namespace {
bool CanReorder(const FuncGraphManagerPtr &mng, const CNodePtr &transdata_node, const CNodePtr &cast_node) {
  auto transdata_input_type = AnfAlgo::GetInputDeviceDataType(transdata_node, 0);
  auto transdata_output_type = AnfAlgo::GetOutputDeviceDataType(transdata_node, 0);
  auto cast_input_type = AnfAlgo::GetInputDeviceDataType(cast_node, 0);
  auto cast_output_type = AnfAlgo::GetOutputDeviceDataType(cast_node, 0);
  // Conditions of reordering transdata_cast to cast_transdata:
  //   1) current transdata is only used by cast
  //   2) transdata works on float32 (transdata supports float16/float32;
  //                                  transdata performances better on float16 due to less data to process)
  //   3) cast works on float32 -> float16
  if (mng->node_users()[transdata_node].size() == 1 && transdata_input_type == kNumberTypeFloat32 &&
      transdata_output_type == transdata_input_type && cast_input_type == transdata_output_type &&
      cast_output_type == kNumberTypeFloat16) {
    return true;
  }
  return false;
}

void SetNodeInfo(const CNodePtr &transdata_node, const CNodePtr &cast_node, const CNodePtr &node) {
  // Initial
  //   TransData: (type0, format0) -> (type0, format1)
  //   Cast:      (type0, format1) -> (type1, format1)
  // After reorder
  //   Cast:      (type0, format0) -> (type1, format0)
  //   TransData: (type1, format0) -> (type1, format1)
  auto type0 = AnfAlgo::GetInputDeviceDataType(transdata_node, 0);
  auto type1 = AnfAlgo::GetOutputDeviceDataType(cast_node, 0);
  auto format0 = AnfAlgo::GetInputFormat(transdata_node, 0);
  auto format1 = AnfAlgo::GetOutputFormat(transdata_node, 0);

  auto abstract = transdata_node->abstract();
  auto scope = cast_node->scope();
  std::vector<std::string> inputs_format;
  std::vector<std::string> outputs_format;
  std::vector<TypeId> inputs_device_type;
  std::vector<TypeId> outputs_device_type;
  auto kernel_type = AnfAlgo::GetKernelType(cast_node);
  auto op_pattern = AnfAlgo::GetOpPattern(cast_node);
  auto fusion_type = AnfAlgo::GetFusionType(cast_node);
  auto processor = AnfAlgo::GetProcessor(cast_node);

  auto node_name = AnfAlgo::GetCNodeName(node);
  if (node_name == "Cast") {
    inputs_format.push_back(format0);
    outputs_format.push_back(format0);
    inputs_device_type.push_back(type0);
    outputs_device_type.push_back(type1);
    // Set attrs
    AnfAlgo::CopyNodeAttrs(cast_node, node);
  } else if (node_name == "TransData") {
    abstract = cast_node->abstract();
    scope = transdata_node->scope();
    inputs_format.push_back(format0);
    outputs_format.push_back(format1);
    inputs_device_type.push_back(type1);
    outputs_device_type.push_back(type1);
    kernel_type = AnfAlgo::GetKernelType(transdata_node);
    op_pattern = AnfAlgo::GetOpPattern(transdata_node);
    fusion_type = AnfAlgo::GetFusionType(transdata_node);
    processor = AnfAlgo::GetProcessor(transdata_node);
    // Set attrs
    AnfAlgo::CopyNodeAttrs(transdata_node, node);
  } else {
    MS_LOG(EXCEPTION) << "Node must be Cast or TransData";
  }

  // Set abstract info
  node->set_abstract(abstract);
  // Set scope info
  node->set_scope(scope);
  // Set kernel build info
  node->set_kernel_info(std::make_shared<device::KernelInfo>());
  kernel::KernelBuildInfo::KernelBuildInfoBuilder info_builder;
  info_builder.SetInputsFormat(inputs_format);
  info_builder.SetInputsDeviceType(inputs_device_type);
  info_builder.SetOutputsFormat(outputs_format);
  info_builder.SetOutputsDeviceType(outputs_device_type);
  info_builder.SetKernelType(kernel_type);
  info_builder.SetOpPattern(op_pattern);
  info_builder.SetFusionType(fusion_type);
  info_builder.SetProcessor(processor);
  AnfAlgo::SetSelectKernelBuildInfo(info_builder.Build(), node.get());
}

bool ReorderTransDataCast(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  for (const auto &anf_node : todos) {
    // Find cast node.
    auto cast_node = anf_node->cast<CNodePtr>();
    if (cast_node == nullptr || !AnfAlgo::CheckPrimitiveType(cast_node, prim::kPrimCast)) {
      continue;
    }

    // Find transdata node before cast node.
    auto cast_input = AnfAlgo::GetInputNode(cast_node, 0);
    auto transdata_node = cast_input->cast<CNodePtr>();
    if (transdata_node == nullptr || !AnfAlgo::CheckPrimitiveType(transdata_node, prim::KPrimTransData)) {
      continue;
    }

    // Reorder transdata_cast to cast_transdata if possible.
    if (!CanReorder(mng, transdata_node, cast_node)) {
      continue;
    }

    MS_LOG(INFO) << "Reorder " << transdata_node->fullname_with_scope() << ", " << cast_node->fullname_with_scope();

    auto new_cast_node = func_graph->NewCNode({NewValueNode(prim::kPrimCast), transdata_node->inputs()[1]});
    SetNodeInfo(transdata_node, cast_node, new_cast_node);

    auto new_transdata_node = func_graph->NewCNode({NewValueNode(prim::KPrimTransData), new_cast_node});
    SetNodeInfo(transdata_node, cast_node, new_transdata_node);

    (void)mng->Replace(cast_node, new_transdata_node);
    changed = true;
  }

  return changed;
}
}  // namespace

bool ReorderOps::Run(const FuncGraphPtr &func_graph) { return ReorderTransDataCast(func_graph); }
}  // namespace opt
}  // namespace mindspore
