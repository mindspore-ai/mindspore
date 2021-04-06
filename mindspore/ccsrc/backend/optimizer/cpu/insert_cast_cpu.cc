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

#include "backend/optimizer/cpu/insert_cast_cpu.h"

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "backend/kernel_compiler/kernel_build_info.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"
#include "utils/utils.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr AddCastOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const std::string &format,
                                const TypeId &input_type, const TypeId &output_type,
                                const std::vector<size_t> &origin_shape, const TypeId &origin_type) {
  MS_EXCEPTION_IF_NULL(func_graph);
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

  // if kernel info is null , it remarks this function is running ut
  if (cast->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    cast->set_kernel_info(kernel_info);
  }
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cast.get());
  AnfAlgo::SetOutputInferTypeAndShape({origin_type}, {origin_shape}, cast.get());
  AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(true), cast);
  return cast;
}

AnfNodePtr InsertCastForMultipleOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                       const std::vector<bool> &need_insert_cast) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  size_t out_num = AnfAlgo::GetOutputTensorNum(cnode);
  for (size_t output_idx = 0; output_idx < out_num; ++output_idx) {
    AnfNodePtr replace_node = nullptr;
    const auto origin_shape = AnfAlgo::GetOutputInferShape(cnode, output_idx);
    const auto infer_type = AnfAlgo::GetOutputInferDataType(cnode, output_idx);
    auto idx = NewValueNode(SizeToLong(output_idx));
    MS_EXCEPTION_IF_NULL(idx);
    auto imm = std::make_shared<Int64Imm>(output_idx);
    idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
    auto getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), cnode, idx});
    AnfAlgo::SetOutputInferTypeAndShape({infer_type}, {origin_shape}, getitem.get());
    if (need_insert_cast[output_idx]) {
      const auto dev_fmt = AnfAlgo::GetOutputFormat(cnode, output_idx);
      const auto device_type = AnfAlgo::GetOutputDeviceDataType(cnode, output_idx);
      if (infer_type != device_type) {
        replace_node =
          AddCastOpNodeToGraph(func_graph, getitem, dev_fmt, device_type, infer_type, origin_shape, infer_type);
        MS_EXCEPTION_IF_NULL(replace_node);
        replace_node->set_scope(cnode->scope());
        AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), replace_node);
        if (kernel_graph != nullptr && kernel_graph->IsInternalOutput(cnode, output_idx)) {
          kernel_graph->ReplaceInternalOutput(cnode, replace_node, output_idx, 0);
        }
      }
    }
  }
  return cnode;
}

void InsertCastForInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  size_t in_num = AnfAlgo::GetInputTensorNum(cnode);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  auto mng = kernel_graph->manager();
  for (size_t input_index = 0; input_index < in_num; ++input_index) {
    auto prev_node = AnfAlgo::GetPrevNodeOutput(cnode, input_index);
    const auto infer_type = AnfAlgo::GetOutputInferDataType(prev_node.first, prev_node.second);
    auto cur_input = AnfAlgo::GetInputNode(cnode, input_index);

    const std::string dev_fmt = AnfAlgo::GetInputFormat(cnode, input_index);
    const std::vector<size_t> origin_shape = AnfAlgo::GetOutputInferShape(prev_node.first, prev_node.second);

    if (TypeId device_type = AnfAlgo::GetInputDeviceDataType(cnode, input_index); infer_type != device_type) {
      auto cast =
        AddCastOpNodeToGraph(func_graph, cur_input, dev_fmt, infer_type, device_type, origin_shape, device_type);
      MS_EXCEPTION_IF_NULL(cast);
      cast->set_scope(cnode->scope());
      AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), cast);
      mng->Replace(cur_input, cast);
    }
  }
}

AnfNodePtr InsertCastForOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                               const std::vector<bool> &need_insert_cast) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetOutputTensorNum(cnode) == 0) {
    return cnode;
  }
  MS_EXCEPTION_IF_NULL(cnode->Type());
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  // Single output
  if (!cnode->Type()->isa<Tuple>()) {
    if (!need_insert_cast[0]) {
      return cnode;
    }
    const std::string dev_fmt = AnfAlgo::GetOutputFormat(cnode, 0);
    std::vector<size_t> origin_shape = AnfAlgo::GetOutputInferShape(cnode, 0);
    const auto infer_type = AnfAlgo::GetOutputInferDataType(cnode, 0);

    const TypeId device_type = AnfAlgo::GetOutputDeviceDataType(cnode, 0);
    AnfNodePtr replace_node = cnode;
    if (infer_type != device_type) {
      replace_node =
        AddCastOpNodeToGraph(func_graph, cnode, dev_fmt, device_type, infer_type, origin_shape, infer_type);
      MS_EXCEPTION_IF_NULL(replace_node);
      replace_node->set_scope(cnode->scope());
      AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), replace_node);
      if (kernel_graph != nullptr && kernel_graph->IsInternalOutput(cnode, 0)) {
        kernel_graph->ReplaceInternalOutput(cnode, replace_node);
      }
    }
    return replace_node;
  }
  // Multiple output
  return InsertCastForMultipleOutput(func_graph, cnode, need_insert_cast);
}
}  // namespace

const BaseRef InsertCastCPU::DefinePattern() const {
  VarPtr V = std::make_shared<CondVar>(UnVisited);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr InsertCastCPU::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::IsRealCNodeKernel(node) || func_graph == nullptr) {
    return nullptr;
  }
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  // process input
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  InsertCastForInput(func_graph, cnode);
  // process output
  return InsertCastForOutput(func_graph, cnode, std::vector<bool>(AnfAlgo::GetOutputTensorNum(cnode), true));
}
}  // namespace opt
}  // namespace mindspore
