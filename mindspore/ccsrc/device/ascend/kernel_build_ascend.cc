/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "device/ascend/kernel_build_ascend.h"

#include <vector>
#include <string>
#include <memory>
#include <functional>

#include "device/ascend/kernel_select_ascend.h"
#include "device/kernel_info.h"
#include "kernel/kernel.h"
#include "kernel/tbe/tbe_kernel_build.h"
#include "kernel/tbe/tbe_kernel_parallel_build.h"
#include "kernel/aicpu/aicpu_kernel_build.h"
#include "kernel/hccl/hccl_kernel_build.h"
#include "kernel/mng/rt_kernel_build.h"
#include "kernel/tbe/tbe_utils.h"
#include "operator/ops.h"
#include "session/anf_runtime_algorithm.h"
#include "./common.h"

namespace mindspore {
namespace device {
namespace ascend {
using mindspore::kernel::tbe::TbeUtils;
using std::make_shared;
static kernel::KernelModPtr SerialCompileImpl(const AnfNodePtr &anf_node) {
  kernel::KernelModPtr kernel_mod_ptr = nullptr;
  KernelType kernel_type = AnfAlgo::GetKernelType(anf_node);
  switch (kernel_type) {
    case KernelType::AICPU_KERNEL: {
      kernel_mod_ptr = kernel::AicpuOpBuild(anf_node);
      break;
    }
    case KernelType::RT_KERNEL: {
      kernel_mod_ptr = kernel::RtOpBuild(anf_node);
      break;
    }
    case KernelType::HCCL_KERNEL: {
      kernel_mod_ptr = kernel::HcclOpBuild(anf_node);
      break;
    }
    default: {
      MS_LOG(EXCEPTION) << "node [" << anf_node->DebugString() << "] Unsupported kernel_type:" << kernel_type;
    }
  }
  return kernel_mod_ptr;
}

static bool KernelBuildParallelCompile(const mindspore::session::KernelGraph *kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  std::vector<AnfNodePtr> tbe_nodes;
  std::vector<AnfNodePtr> other_nodes;
  for (const auto &anf_node : kernel_graph_ptr->execution_order()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    if (!AnfAlgo::IsRealKernel(anf_node)) {
      continue;
    }
    KernelType kernel_type = AnfAlgo::GetKernelType(anf_node);
    switch (kernel_type) {
      case KernelType::TBE_KERNEL: {
        if (AnfAlgo::GetKernelMod(anf_node) == nullptr) {
          tbe_nodes.push_back(anf_node);
        }
        break;
      }
      default: {
        other_nodes.push_back(anf_node);
        break;
      }
    }
  }
  bool ret = kernel::TbeOpParallelBuild(tbe_nodes);
  for (const auto &anf_node : other_nodes) {
    kernel::KernelModPtr kernel_mod_ptr = SerialCompileImpl(anf_node);
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, anf_node.get());
  }
  return ret;
}

static vector<int> CalCleanZerosSize(const CNodePtr &pre_node) {
  MS_EXCEPTION_IF_NULL(pre_node);
  std::vector<int> clean_size_list;
  // clean output
  if (AnfAlgo::HasNodeAttr(kAttrAutomicOutputIndexs, pre_node)) {
    auto clean_output_indexs = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAutomicOutputIndexs);
    for (auto index : clean_output_indexs) {
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(pre_node, index);
      size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
      std::vector<size_t> shape = AnfAlgo::GetOutputDeviceShape(pre_node, index);
      auto size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
      clean_size_list.push_back((size + kMemAlignSize + 31) / kMemAlignSize * kMemAlignSize);
    }
  }
  // clean workspace
  auto workspaces_size = 0;
  if (AnfAlgo::HasNodeAttr(kAttrAutomicWorkspaceSize, pre_node)) {
    workspaces_size = AnfAlgo::GetNodeAttr<int>(pre_node, kAttrAutomicWorkspaceSize);
    clean_size_list.push_back(workspaces_size);
  }
  MS_LOG(INFO) << "clear output size:" << clean_size_list.size() << ", workspace size:" << workspaces_size
               << ",pre_node:" << pre_node->fullname_with_scope();
  return clean_size_list;
}

static void AddTbeClearZeroNode(mindspore::session::KernelGraph *const kernel_graph,
                                const mindspore::CNodePtr &pre_node, std::vector<mindspore::CNodePtr> *new_nodes) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(pre_node);
  MS_EXCEPTION_IF_NULL(new_nodes);
  auto clear_zero_prim = std::make_shared<Primitive>(kAtomicAddrCleanOpName);
  MS_EXCEPTION_IF_NULL(clear_zero_prim);
  auto new_value_node = NewValueNode(clear_zero_prim);
  MS_EXCEPTION_IF_NULL(new_value_node);
  std::vector<AnfNodePtr> inputs = {new_value_node};
  inputs.push_back(pre_node);
  CNodePtr clear_zero = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(clear_zero);
  AbstractBasePtr abstract = std::make_shared<abstract::AbstractNone>();
  MS_EXCEPTION_IF_NULL(abstract);
  clear_zero->set_abstract(abstract);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  builder->SetKernelType(KernelType::TBE_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), clear_zero.get());
  auto clean_size = CalCleanZerosSize(pre_node);
  AnfAlgo::SetNodeAttr(kAttrAutomicAddMemSize, MakeValue(clean_size), clear_zero);
  AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(pre_node.get()), clear_zero.get());
  new_nodes->push_back(clear_zero);
}

bool IsAtomicNode(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto parameters_indexs = kernel_mod->GenParameters();
  if (parameters_indexs.empty()) {
    return false;
  }
  auto atomic_flag = false;
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  auto workspace_size_list = kernel_mod->GetWorkspaceSizeList();
  size_t workspace_num = kernel_mod->GetWorkspaceSizeList().size();
  if (input_num + workspace_num + output_num > parameters_indexs.size()) {
    size_t lossNum = (input_num + workspace_num + output_num) - parameters_indexs.size();
    for (size_t i = 0; i < lossNum; i++) {
      parameters_indexs.push_back(0);
    }
  }
  std::vector<size_t> clean_output_indexs;
  // in parameters data sort as input->workspace->output
  size_t index = 0;
  while (index < output_num) {
    if (parameters_indexs[input_num + workspace_num + index] == 1) {
      atomic_flag = true;
      clean_output_indexs.push_back(index);
    }
    index++;
  }
  if (atomic_flag) {
    AnfAlgo::SetNodeAttr(kAttrAutomicOutputIndexs, MakeValue(clean_output_indexs), kernel_node);
  }
  for (size_t i = 0; i < workspace_num; ++i) {
    if (parameters_indexs[input_num + i] == 1) {
      atomic_flag = true;
      AnfAlgo::SetNodeAttr(kAttrAutomicWorkspaceSize,
                           MakeValue(std::accumulate(workspace_size_list.begin(), workspace_size_list.end(), 0)),
                           kernel_node);
      break;
    }
  }
  return atomic_flag;
}

bool KernelBuild(const mindspore::session::KernelGraph *kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  TbeUtils::LoadCache();
  bool ret;
  ret = device::ascend::KernelBuildParallelCompile(kernel_graph_ptr);
  return ret;
}

void KernelBuildPreprocess(mindspore::session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<CNodePtr> new_nodes;
  for (const auto &anf_node : kernel_graph->execution_order()) {
    std::string apply_function_name = AnfAlgo::GetCNodeName(anf_node);
    if (apply_function_name == prim::kPrimMaxPoolGrad->name() &&
        AnfAlgo::GetKernelType(anf_node) == KernelType::AUTO_DIFF_KERNEL) {
      auto clear_zero_prim = std::make_shared<Primitive>(kClearZeroOpName);
      MS_EXCEPTION_IF_NULL(clear_zero_prim);
      auto new_value_node = NewValueNode(clear_zero_prim);
      MS_EXCEPTION_IF_NULL(new_value_node);
      std::vector<AnfNodePtr> inputs = {new_value_node};
      inputs.push_back(anf_node);
      CNodePtr clear_zero = kernel_graph->NewCNode(inputs);
      MS_EXCEPTION_IF_NULL(clear_zero);
      auto kernel_info = std::make_shared<device::KernelInfo>();
      MS_EXCEPTION_IF_NULL(kernel_info);
      clear_zero->set_kernel_info(kernel_info);
      AbstractBasePtr abstract = std::make_shared<abstract::AbstractNone>();
      MS_EXCEPTION_IF_NULL(abstract);
      AnfAlgo::SetNodeAttr("input_names", MakeValue(std::vector<std::string>({"x"})), clear_zero);
      SelectKernelInfo(clear_zero);
      // set the distinction label of clear same with anf
      AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(anf_node.get()), clear_zero.get());
      new_nodes.push_back(clear_zero);
    } else if (AnfAlgo::GetKernelType(anf_node) == KernelType::TBE_KERNEL) {
      if (IsAtomicNode(anf_node)) {
        AddTbeClearZeroNode(kernel_graph, anf_node, &new_nodes);
      }
    }
    new_nodes.push_back(anf_node);
  }
  kernel_graph->set_execution_order(new_nodes);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
