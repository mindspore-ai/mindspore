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

#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include "mindspore/core/ops/core_ops.h"
#include "backend/common/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/decrease_compute_precision.h"

namespace mindspore::graphkernel {
// Add CastCNode
CNodePtr AddCastCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const std::string &format,
                      const TypeId &input_type, const TypeId &output_type, const ShapeVector &origin_shape,
                      const TypeId &origin_type) {
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr cast = func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())), input});
  MS_EXCEPTION_IF_NULL(cast);
  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({format});
  builder.SetOutputsFormat({format});
  builder.SetInputsDeviceType({input_type});
  builder.SetOutputsDeviceType({output_type});
  builder.SetFusionType(kernel::kPatternOpaque);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::AKG_KERNEL);
  if (cast->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    cast->set_kernel_info(kernel_info);
  }
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cast.get());
  common::AnfAlgo::SetOutputInferTypeAndShape({origin_type}, {origin_shape}, cast.get());
  common::AnfAlgo::SetNodeAttr(kAttrDstType, TypeIdToType(output_type), cast);
  common::AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(true), cast);
  common::AnfAlgo::SetNodeAttr(kAttrDatadumpOriginalNames, MakeValue<std::vector<std::string>>({}), cast);
  return cast;
}

// Update Output Abatract and BuildInfo as Input Changed
void UpdateOutputInfo(const AnfNodePtr &cnode) {
  if (!common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimMakeTuple)) {
    ShapeVector out_shape = GetShape(cnode);
    auto abs_shape_ptr = std::make_shared<abstract::Shape>(abstract::Shape(out_shape));
    auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(TypeId::kNumberTypeFloat16), abs_shape_ptr);
    cnode->set_abstract(abstract);
    std::vector<std::string> input_formats = AnfAlgo::GetAllInputFormats(cnode);
    std::vector<TypeId> input_types = AnfAlgo::GetAllInputDeviceTypes(cnode);
    for (size_t i = 0; i < input_types.size(); i++) {
      input_types[i] = TypeId::kNumberTypeFloat16;
    }
    std::vector<std::string> output_formats = AnfAlgo::GetAllOutputFormats(cnode);
    std::vector<TypeId> output_types = {TypeId::kNumberTypeFloat16};
    auto graph_sel_info = BuildSelectKernelBuildInfo(input_formats, input_types, output_formats, output_types,
                                                     AnfAlgo::GetProcessor(cnode));
    AnfAlgo::SetSelectKernelBuildInfo(graph_sel_info, cnode.get());
  }
}

CNodePtr InsertCastForGraphKernel(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto mng = func_graph->manager();
  size_t in_num = common::AnfAlgo::GetInputNum(cnode);  // include monads.
  for (size_t input_index = 0; input_index < in_num; ++input_index) {
    auto cur_input = common::AnfAlgo::GetInputNode(cnode, input_index);
    if (HasAbstractMonad(cur_input)) {
      continue;
    }
    auto prev_node = common::AnfAlgo::GetPrevNodeOutput(cnode, input_index);
    auto in_node = prev_node.first;
    auto in_index = prev_node.second;
    auto ori_shape = AnfAlgo::GetOutputDeviceShape(in_node, in_index);
    auto ori_dtype = AnfAlgo::GetOutputDeviceDataType(in_node, in_index);
    const std::string dev_fmt = AnfAlgo::GetInputFormat(cnode, input_index);
    if (cur_input->isa<ValueNode>()) {
      ori_dtype = cur_input->cast<ValueNodePtr>()->value()->cast<tensor::TensorPtr>()->data_type();
    }
    auto new_dtype = TypeId::kNumberTypeFloat16;
    if (ori_dtype == TypeId::kNumberTypeFloat32) {
      if (cur_input->isa<ValueNode>()) {
        auto valuePtr = cur_input->cast<ValueNodePtr>();
        auto itensor = std::make_shared<tensor::Tensor>(
          TypeId::kNumberTypeFloat16, valuePtr->value()->cast<tensor::TensorPtr>()->shape(),
          valuePtr->value()->cast<tensor::TensorPtr>()->data_c(), TypeId::kNumberTypeFloat32);
        auto value_node = std::make_shared<ValueNode>(itensor);
        value_node->set_abstract(itensor->ToAbstract());
        (void)mng->Replace(cur_input, value_node);
      }
      auto cast = AddCastCNode(func_graph, cur_input, dev_fmt, ori_dtype, new_dtype, ori_shape, new_dtype);
      MS_EXCEPTION_IF_NULL(cast);
      cast->set_scope(cnode->scope());
      ShapeVector out_shape = GetShape(cur_input);
      auto abs_shape_ptr = std::make_shared<abstract::Shape>(out_shape);

      auto abstract =
        std::make_shared<abstract::AbstractTensor>(TypeIdToType(TypeId::kNumberTypeFloat16), abs_shape_ptr);
      cast->set_abstract(abstract);
      common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), cast);
      (void)mng->Replace(cur_input, cast);
    }
  }
  CNodePtr new_node = nullptr;
  new_node = std::make_shared<CNode>(*cnode);
  MS_EXCEPTION_IF_NULL(new_node);
  UpdateOutputInfo(new_node);
  return new_node;
}

bool DecreaseComputePrecision::Process(const FuncGraphPtr &func_graph) const {
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto todos = TopoSort(func_graph->get_return());
  bool changed = false;
  // Cast Down CNODES
  for (auto node : todos) {
    if (node->isa<CNode>() && !common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
      auto cnode = node->cast<CNodePtr>();
      if (IsPrimitiveCNode(cnode, prim::kPrimCast)) {
        if (AnfAlgo::GetOutputDeviceDataType(cnode->input(1), 0) == kNumberTypeFloat16) {
          auto in_node = cnode->input(1);
          (void)mng->Replace(node, in_node);
          changed = true;
          continue;
        }
        if (AnfAlgo::GetOutputDeviceDataType(cnode->input(1), 0) == kNumberTypeFloat32 &&
            AnfAlgo::GetOutputDeviceDataType(cnode, 0) == kNumberTypeFloat16) {
          continue;
        }
      }
      auto new_node = InsertCastForGraphKernel(func_graph, cnode);
      (void)mng->Replace(node, new_node);
      changed = true;
    }
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }

  // Cast Up Outputs
  auto old_output = func_graph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(old_output);
  auto add_cast = [&func_graph](const CNodePtr &old_cnode, bool is_output, std::vector<AnfNodePtr> &new_inputs) {
    AnfNodePtrList inputs1 = {NewValueNode(prim::kPrimCast), old_cnode};
    auto cnode1 = func_graph->NewCNode(inputs1);
    func_graph->AddNode(cnode1);
    ShapeVector cast_shape = GetShape(old_cnode);
    auto shape_ptr = std::make_shared<abstract::Shape>(abstract::Shape(cast_shape));
    auto new_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(TypeId::kNumberTypeFloat32), shape_ptr);
    cnode1->set_abstract(new_abstract);
    cnode1->set_scope(old_cnode->scope());
    SetNodeAttrSafely(kAttrDstType, kFloat32, cnode1);
    MS_EXCEPTION_IF_NULL(cnode1);
    cnode1->set_kernel_info(std::make_shared<device::KernelInfo>());
    std::vector<std::string> cnode_input_format = {GetFormat(old_cnode)};
    std::vector<TypeId> cnode_input_type = {kNumberTypeFloat16};
    std::vector<std::string> cnode_output_format = {GetFormat(old_cnode)};
    std::vector<TypeId> cnode_output_type = {kNumberTypeFloat32};
    kernel::KernelBuildInfo::KernelBuildInfoBuilder graph_info_builder;
    graph_info_builder.SetInputsFormat(cnode_input_format);
    graph_info_builder.SetInputsDeviceType(cnode_input_type);
    graph_info_builder.SetOutputsFormat(cnode_output_format);
    graph_info_builder.SetOutputsDeviceType(cnode_output_type);
    graph_info_builder.SetProcessor(kernel::GetProcessorFromContext());
    graph_info_builder.SetKernelType(KernelType::AKG_KERNEL);
    graph_info_builder.SetFusionType(kernel::kPatternOpaque);
    auto info_1 = graph_info_builder.Build();
    AnfAlgo::SetSelectKernelBuildInfo(info_1, cnode1.get());
    if (is_output) {
      func_graph->set_output(cnode1);
    } else {
      (void)new_inputs.emplace_back(cnode1);
    }
  };

  std::vector<AnfNodePtr> new_inputs;
  if (common::AnfAlgo::CheckPrimitiveType(old_output, prim::kPrimMakeTuple)) {
    (void)new_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    auto all_out = common::AnfAlgo::GetAllOutput(old_output);
    for (const auto &out : all_out) {
      auto c_out = out->cast<CNodePtr>();
      if (c_out) {
        add_cast(c_out, false, new_inputs);
      }
    }
    old_output->set_inputs(new_inputs);
  } else {
    add_cast(old_output, true, new_inputs);
  }
  return changed;
}

bool IsCastUnAware(const FuncGraphPtr &func_graph) {
  std::vector<PrimitivePtr> cast_aware_list = {prim::kPrimReduceSum, prim::kPrimReduceMean, prim::kPrimReduceAll};
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }

  auto graph_name = GetValue<std::string>(func_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
  if (graph_name.find("atomic") != std::string::npos) {
    return false;
  }

  auto todos = TopoSort(func_graph->get_return());
  for (auto node : todos) {
    if (node->isa<CNode>()) {
      if (std::find(cast_aware_list.begin(), cast_aware_list.end(), common::AnfAlgo::GetCNodePrimitive(node)) !=
          cast_aware_list.end()) {
        return false;
      }
      auto itype_id = AnfAlgo::GetOutputDeviceDataType(node, 0);
      if (itype_id != TypeId::kNumberTypeFloat16 && itype_id != TypeId::kNumberTypeFloat32) {
        return false;
      }
    }
  }
  return true;
}

bool DecreaseComputePrecision::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto todos = TopoSort(func_graph->get_return());
  bool changed = false;
  for (const auto &node : todos) {
    if (common::AnfAlgo::IsGraphKernel(node)) {
      auto sub_func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      MS_ERROR_IF_NULL(sub_func_graph);
      if (IsCastUnAware(sub_func_graph)) {
        changed = Process(sub_func_graph) || changed;
      }
    }
  }
  if (changed) {
    GkUtils::UpdateFuncGraphManager(mng, func_graph);
  }
  return changed;
}
}  // namespace mindspore::graphkernel
