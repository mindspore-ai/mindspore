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

#include "plugin/device/cpu/optimizer/insert_cast_cpu.h"

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include "backend/common/optimizer/helper.h"
#include "kernel/kernel_build_info.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_graph.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace opt {
namespace {
constexpr unsigned int kLstmReserveIndex = 3;
AnfNodePtr AddCastOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const std::string &format,
                                const TypeId &input_type, const TypeId &output_type,
                                const abstract::BaseShapePtr &origin_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(origin_shape);
  std::string input_format = format;
  std::string output_format = format;
  CNodePtr cast = func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())), input});
  MS_EXCEPTION_IF_NULL(cast);
  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({input_format});
  builder.SetOutputsFormat({output_format});
  builder.SetInputsDeviceType({input_type});
  builder.SetOutputsDeviceType({output_type});
  if (cast->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    cast->set_kernel_info(kernel_info);
  }
  if (origin_shape->IsDynamic()) {
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), cast);
    common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), cast);
  }
  common::AnfAlgo::SetNodeAttr("dst_type", TypeIdToType(output_type), cast);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cast.get());
  common::AnfAlgo::SetOutputTypeAndDetailShape({output_type}, {origin_shape}, cast.get());
  common::AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(true), cast);
  std::shared_ptr<kernel::NativeCpuKernelMod> cpu_kernel =
    kernel::Factory<kernel::NativeCpuKernelMod>::Instance().Create(kCastOpName);
  if (cpu_kernel == nullptr) {
    MS_LOG(EXCEPTION) << "Operator[Cast] " << cast->kernel_info() << " is not support.";
  }

  auto kernel_attrs = cpu_kernel->GetOpSupport();
  kernel::SetCpuRefMapToKernelInfo(cast, kernel_attrs);
  auto thread_pool = kernel::GetActorMgrInnerThreadPool();
  cpu_kernel->SetThreadPool(thread_pool);
  auto args = kernel::AbstractArgsFromCNode(cast);
  auto ret = cpu_kernel->Init(args.op, args.inputs, args.outputs);
  if (!ret) {
    MS_LOG(EXCEPTION) << trace::DumpSourceLines(cast);
  }
  if (cpu_kernel->Resize(args.op, args.inputs, args.outputs, kernel::GetKernelDepends(cast)) ==
      kernel::KRET_RESIZE_FAILED) {
    MS_LOG(EXCEPTION) << "CPU kernel op [" << cast->fullname_with_scope() << "] Resize failed.";
  }
  AnfAlgo::SetKernelMod(cpu_kernel, cast.get());
  return cast;
}

std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetNodeUserList(const FuncGraphPtr &graph,
                                                                         const AnfNodePtr &node) {
  auto output_node_list = std::make_shared<std::vector<std::pair<AnfNodePtr, int>>>();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto iter = manager->node_users().find(node);
  if (iter == manager->node_users().end()) {
    return output_node_list;
  }
  auto output_info_list = iter->second;
  (void)std::copy(output_info_list.begin(), output_info_list.end(), std::back_inserter(*output_node_list));
  return output_node_list;
}

void SyncWeightNodeWithCast(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const AnfNodePtr &cur_input,
                            const AnfNodePtr &cast, const std::string &format, const TypeId &device_type,
                            const TypeId &origin_type, const abstract::BaseShapePtr &origin_shape,
                            std::vector<AnfNodePtr> *make_tuple_inputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cast);
  MS_EXCEPTION_IF_NULL(make_tuple_inputs);
  auto first_depend_node =
    func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), cast, cnode});
  MS_EXCEPTION_IF_NULL(first_depend_node);
  first_depend_node->set_abstract(cast->abstract());
  auto post_cast = AddCastOpNodeToGraph(func_graph, first_depend_node, format, device_type, origin_type, origin_shape);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->AddRefCorrespondPairs(std::make_pair(post_cast, 0), common::AnfAlgo::VisitKernel(cur_input, 0));
  make_tuple_inputs->push_back(post_cast);
}

void InsertCast(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  size_t in_num = common::AnfAlgo::GetInputTensorNum(cnode);
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  for (size_t input_index = 0; input_index < in_num; ++input_index) {
    auto prev_node = common::AnfAlgo::GetPrevNodeOutput(cnode, input_index);
    auto origin_type = AnfAlgo::GetOutputDeviceDataType(prev_node.first, prev_node.second);
    if (origin_type == kTypeUnknown) {
      origin_type = common::AnfAlgo::GetOutputInferDataType(prev_node.first, prev_node.second);
    }
    auto cur_input = common::AnfAlgo::GetInputNode(cnode, input_index);
    MS_EXCEPTION_IF_NULL(cur_input);
    const std::string dev_fmt = AnfAlgo::GetInputFormat(cnode, input_index);
    const abstract::BaseShapePtr origin_shape =
      common::AnfAlgo::GetOutputDetailShape(prev_node.first, prev_node.second);
    TypeId device_type = AnfAlgo::GetInputDeviceDataType(cnode, input_index);
    if (origin_type != device_type && origin_type != kTypeUnknown && device_type != kTypeUnknown) {
      auto cast = AddCastOpNodeToGraph(func_graph, cur_input, dev_fmt, origin_type, device_type, origin_shape);
      MS_EXCEPTION_IF_NULL(cast);
      cast->set_scope(cnode->scope());
      cnode->set_input(input_index + 1, cast);
      auto real_input = common::AnfAlgo::VisitKernel(cur_input, 0).first;
      MS_EXCEPTION_IF_NULL(real_input);
      if (common::AnfAlgo::IsUpdateParameterKernel(cnode) && real_input->isa<Parameter>() &&
          common::AnfAlgo::IsParameterWeight(real_input->cast<ParameterPtr>())) {
        SyncWeightNodeWithCast(func_graph, cnode, cur_input, cast, dev_fmt, device_type, origin_type, origin_shape,
                               &make_tuple_inputs);
      }
    }
    if (make_tuple_inputs.size() > 1) {
      auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
      auto second_depend_node =
        func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), cnode, make_tuple});
      MS_EXCEPTION_IF_NULL(second_depend_node);
      second_depend_node->set_abstract(cnode->abstract());
      auto used_node_list = GetRealNodeUsedList(func_graph, cnode);
      if (used_node_list != nullptr && used_node_list->empty()) {
        used_node_list = GetNodeUserList(func_graph, cnode);
      }
      for (size_t j = 0; j < used_node_list->size(); j++) {
        auto used_node = used_node_list->at(j).first;
        MS_EXCEPTION_IF_NULL(used_node);
        if (!used_node->isa<CNode>()) {
          continue;
        }
        utils::cast<CNodePtr>(used_node)->set_input(IntToSize(used_node_list->at(j).second), second_depend_node);
      }
    }
  }
}

void InsertCastForGraphOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &func_output) {
  MS_EXCEPTION_IF_NULL(func_output);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(func_output);
  if (!func_output->isa<CNode>()) {
    return;
  }
  auto func_output_node = func_output->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(func_output_node);
  for (size_t i = 0; i < input_num; i++) {
    auto input_node = common::AnfAlgo::GetInputNode(func_output_node, i);
    MS_EXCEPTION_IF_NULL(input_node);
    auto abstract = input_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    if (!abstract->isa<abstract::AbstractTensor>()) {
      MS_LOG(INFO) << "The " << i << "th output of graph is not a tensor type, skipping insert cast.";
      continue;
    }
    if (!input_node->isa<CNode>()) {
      MS_LOG(INFO) << "The " << i << "th output of graph is not a CNode, skipping insert cast.";
      continue;
    }
    auto kernel_node = common::AnfAlgo::VisitKernel(input_node, 0).first;
    MS_EXCEPTION_IF_NULL(kernel_node);
    auto infer_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(func_output, i);
    auto device_type = AnfAlgo::GetPrevNodeOutputDeviceDataType(func_output, i);
    const std::string dev_fmt = AnfAlgo::GetPrevNodeOutputFormat(func_output, i);
    if (infer_type != device_type && device_type != kTypeUnknown) {
      const abstract::BaseShapePtr origin_shape = common::AnfAlgo::GetPrevNodeOutputDetailShape(func_output_node, i);
      auto cast = AddCastOpNodeToGraph(func_graph, input_node, dev_fmt, device_type, infer_type, origin_shape);
      MS_EXCEPTION_IF_NULL(cast);
      cast->set_scope(func_output->scope());
      func_output_node->set_input(i + 1, cast);
      auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
      if (kernel_graph != nullptr) {
        MS_LOG(INFO) << "Replace internal output from:" << kernel_node->DebugString() << " to:" << cast->DebugString()
                     << " for graph:" << kernel_graph->ToString();
        kernel_graph->ReplaceInternalOutput(kernel_node, cast);
      }
    }
  }
}
}  // namespace

bool InsertCastCPU::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  for (auto node : node_list) {
    if (node != nullptr && node->isa<CNode>() && AnfUtils::IsRealKernel(node)) {
      CNodePtr cnode = node->cast<CNodePtr>();
      InsertCast(func_graph, cnode);
    }
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto func_output = func_graph->output();
  InsertCastForGraphOutput(func_graph, func_output);
  return true;
}
}  // namespace opt
}  // namespace mindspore
