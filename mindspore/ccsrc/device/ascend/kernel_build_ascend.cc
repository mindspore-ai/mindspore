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
#include "kernel/akg/ascend/akg_ascend_kernel_build.h"
#include "kernel/aicpu/aicpu_kernel_build.h"
#include "kernel/hccl/hccl_kernel_build.h"
#include "kernel/rts/rt_kernel_build.h"
#include "kernel/tbe/tbe_utils.h"
#include "kernel/common_utils.h"
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

static bool KernelPreBuildParallelCompile(const mindspore::session::KernelGraph *kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  std::vector<AnfNodePtr> tbe_nodes;
  for (const auto &anf_node : kernel_graph_ptr->execution_order()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    if (!AnfAlgo::IsRealKernel(anf_node)) {
      continue;
    }
    KernelType kernel_type = AnfAlgo::GetKernelType(anf_node);
    switch (kernel_type) {
      case KernelType::TBE_KERNEL: {
        if (AnfAlgo::GetKernelMod(anf_node) == nullptr &&
            AnfAlgo::GetFusionType(anf_node) == kernel::FusionType::DYNAMIC) {
          tbe_nodes.push_back(anf_node);
        }
        break;
      }
      default: {
        break;
      }
    }
  }
  bool ret = kernel::TbeOpParallelPreBuild(tbe_nodes);
  return ret;
}

static bool KernelBuildParallelCompile(const mindspore::session::KernelGraph *kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  std::vector<AnfNodePtr> tbe_nodes;
  std::vector<AnfNodePtr> akg_nodes;
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
      case KernelType::AKG_KERNEL: {
        akg_nodes.push_back(anf_node);
        break;
      }
      default: {
        other_nodes.push_back(anf_node);
        break;
      }
    }
  }
  bool tbe_ret = kernel::TbeOpParallelBuild(tbe_nodes);
  bool akg_ret = kernel::AkgAscendKernelParallelBuild(akg_nodes);
  auto bin_map = kernel::tbe::KernelMeta::GetInstance();
  (void)bin_map->ReadIndex(kernel::kCceKernelMeta);
  for (const auto &anf_node : other_nodes) {
    kernel::KernelModPtr kernel_mod_ptr = SerialCompileImpl(anf_node);
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, anf_node.get());
  }
  return tbe_ret && akg_ret;
}

static std::vector<size_t> CalCleanZerosSize(const CNodePtr &pre_node) {
  MS_EXCEPTION_IF_NULL(pre_node);
  auto kernel_mod = AnfAlgo::GetKernelMod(pre_node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  std::vector<size_t> clean_size_list;
  // clean output
  if (AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {
    auto output_indexs = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
    auto output_men_size = kernel_mod->GetOutputSizeList();
    for (auto index : output_indexs) {
      auto clean_item = (output_men_size.at(index) + kMemAlignSize + 31) / kMemAlignSize * kMemAlignSize;
      clean_size_list.emplace_back(clean_item);
    }
  }
  // clean workspace
  if (AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {
    auto workspace_indexs = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
    auto workspace_men_sizes = kernel_mod->GetWorkspaceSizeList();
    for (const auto &index : workspace_indexs) {
      auto clean_item = (workspace_men_sizes.at(index) + kMemAlignSize + 31) / kMemAlignSize * kMemAlignSize;
      clean_size_list.emplace_back(clean_item);
    }
  }
  MS_LOG(INFO) << "clear output size:" << clean_size_list.size() << ",pre_node:" << pre_node->fullname_with_scope();
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
  AnfAlgo::SetNodeAttr(kAttrAtomicAddMemSize, MakeValue(clean_size), clear_zero);
  AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(pre_node.get()), clear_zero.get());
  new_nodes->push_back(clear_zero);
}

static bool IsAtomicNode(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto parameters_indexs = kernel_mod->GenParameters();
  if (parameters_indexs.empty()) {
    return false;
  }
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  size_t workspace_num = kernel_mod->GetWorkspaceSizeList().size();
  size_t param_num = parameters_indexs.size();
  size_t total_num = input_num + workspace_num + output_num;
  MS_LOG(INFO) << "parameters size: " << param_num << ", input & workspace & output num: " << total_num;
  size_t pad_index = param_num;
  for (; pad_index < total_num; ++pad_index) {
    parameters_indexs.emplace_back(0);
  }
  // process input
  for (size_t j = 0; j < input_num; ++j) {
    if (parameters_indexs.at(j) == 1) {
      MS_LOG(EXCEPTION) << "Atomic addr clean does't support clean input address, input index: " << j;
    }
  }
  // process output
  std::vector<size_t> output_indexs = {};
  for (size_t i = 0; i < output_num; ++i) {
    auto param_output = parameters_indexs.at(input_num + workspace_num + i);
    if (param_output == 1) {
      output_indexs.emplace_back(i);
      MS_LOG(INFO) << "Atomic clear output index: " << i;
    }
  }
  if (!output_indexs.empty()) {
    AnfAlgo::SetNodeAttr(kAttrAtomicOutputIndexs, MakeValue(output_indexs), kernel_node);
  }
  // process workspace
  std::vector<size_t> workspace_indexs = {};
  for (size_t k = 0; k < workspace_num; ++k) {
    auto param_workspace = parameters_indexs.at(input_num + k);
    if (param_workspace == 1) {
      workspace_indexs.emplace_back(k);
      MS_LOG(INFO) << "Atomic clear workspace index: " << k;
    }
  }
  if (!workspace_indexs.empty()) {
    AnfAlgo::SetNodeAttr(kAttrAtomicWorkspaceIndexs, MakeValue(workspace_indexs), kernel_node);
  }
  return !(workspace_indexs.empty() && output_indexs.empty());
}

bool KernelPreBuild(const mindspore::session::KernelGraph *kernel_graph_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_graph_ptr);
  bool ret = device::ascend::KernelPreBuildParallelCompile(kernel_graph_ptr);
  return ret;
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
        AnfAlgo::GetKernelType(anf_node) == KernelType::AKG_KERNEL) {
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
