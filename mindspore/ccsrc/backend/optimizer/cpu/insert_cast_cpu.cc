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
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
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
  if (cast->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    cast->set_kernel_info(kernel_info);
  }
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cast.get());
  AnfAlgo::SetOutputInferTypeAndShape({origin_type}, {origin_shape}, cast.get());
  AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(true), cast);
  std::shared_ptr<kernel::CPUKernel> cpu_kernel = kernel::CPUKernelFactory::GetInstance().Create(kCastOpName, cast);
  if (cpu_kernel == nullptr) {
    MS_LOG(EXCEPTION) << "Operator[Cast] " << cast->kernel_info() << " is not support.";
  }
  try {
    cpu_kernel->Init(cast);
  } catch (std::exception &e) {
    MS_LOG(EXCEPTION) << e.what() << "\nTrace: " << trace::DumpSourceLines(cast);
  }
  AnfAlgo::SetKernelMod(cpu_kernel, cast.get());
  return cast;
}

void InsertCast(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  size_t in_num = AnfAlgo::GetInputTensorNum(cnode);
  for (size_t input_index = 0; input_index < in_num; ++input_index) {
    auto prev_node = AnfAlgo::GetPrevNodeOutput(cnode, input_index);
    auto origin_type = AnfAlgo::GetOutputDeviceDataType(prev_node.first, prev_node.second);
    if (origin_type == kTypeUnknown) {
      origin_type = AnfAlgo::GetOutputInferDataType(prev_node.first, prev_node.second);
    }
    auto cur_input = AnfAlgo::GetInputNode(cnode, input_index);
    if (cur_input->isa<Parameter>() && AnfAlgo::IsParameterWeight(cur_input->cast<ParameterPtr>())) {
      continue;
    }
    const std::string dev_fmt = AnfAlgo::GetInputFormat(cnode, input_index);
    const std::vector<size_t> origin_shape = AnfAlgo::GetOutputInferShape(prev_node.first, prev_node.second);

    if (TypeId device_type = AnfAlgo::GetInputDeviceDataType(cnode, input_index); origin_type != device_type) {
      auto cast =
        AddCastOpNodeToGraph(func_graph, cur_input, dev_fmt, origin_type, device_type, origin_shape, device_type);
      MS_EXCEPTION_IF_NULL(cast);
      cast->set_scope(cnode->scope());
      cnode->set_input(input_index + 1, cast);
    }
  }
}
}  // namespace

bool InsertCastCPU::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  for (auto node : node_list) {
    if (node != nullptr && node->isa<CNode>() && AnfAlgo::IsRealKernel(node)) {
      CNodePtr cnode = node->cast<CNodePtr>();
      InsertCast(func_graph, cnode);
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
