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

#include "pre_activate/pass/add_atomic_clean.h"
#include <memory>
#include <vector>
#include <functional>
#include "operator/ops.h"
#include "utils/utils.h"
#include "utils/graph_utils.h"
#include "utils/log_adapter.h"
#include "session/anf_runtime_algorithm.h"
#include "session/kernel_graph.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
namespace {

static std::vector<size_t> g_output_idx;

bool HasAtomic(const AnfNodePtr &input) {
  if (IsPrimitiveCNode(input)) {
    const auto &cnode = input->cast<CNodePtr>();
    const auto &prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    return prim->HasAttr("atomic_add");
  }
  return false;
}

std::vector<int> CalCleanSize(const CNodePtr &pre_node) {
  MS_EXCEPTION_IF_NULL(pre_node);
  std::vector<int> clean_size_list;
  // clean output
  for (auto &index : g_output_idx) {
    TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(pre_node, index);
    size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
    std::vector<size_t> shape = AnfAlgo::GetOutputDeviceShape(pre_node, index);
    auto size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    clean_size_list.push_back((size + kMemAlignSize + 31) / kMemAlignSize * kMemAlignSize);
  }
  MS_LOG(DEBUG) << "Clear output size: " << clean_size_list.size() << ", pre_node: " << pre_node->fullname_with_scope();
  return clean_size_list;
}

CNodePtr CreateTbeAtomicCleanNode(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                                  const mindspore::CNodePtr &pre_node) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(pre_node);
  auto clean_zero_prim = std::make_shared<Primitive>(kAtomicAddrCleanOpName);
  auto new_value_node = NewValueNode(clean_zero_prim);
  std::vector<AnfNodePtr> inputs = {new_value_node};
  CNodePtr clean_zero = kernel_graph->NewCNode(inputs);
  AbstractBasePtr abstract = std::make_shared<abstract::AbstractNone>();
  clean_zero->set_abstract(abstract);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  builder->SetKernelType(KernelType::TBE_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), clean_zero.get());
  auto clean_size = CalCleanSize(pre_node);
  AnfAlgo::SetNodeAttr(kAttrAutomicAddMemSize, MakeValue(clean_size), clean_zero);
  AnfAlgo::SetNodeAttr(kAttrAutomicOutputIndexs, MakeValue(g_output_idx), clean_zero);
  AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(pre_node.get()), clean_zero.get());
  return clean_zero;
}
}  // namespace

void AddAtomicClean(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }
  auto &todos = kernel_graph->execution_order();
  for (auto iter = todos.cbegin(); iter != todos.end(); ++iter) {
    auto node = *iter;
    if (AnfAlgo::IsGraphKernel(node) && kernel_graph->nodes().contains(node)) {
      auto fg = GetValueNode<FuncGraphPtr>(node->input(kAnfPrimitiveIndex));
      MS_EXCEPTION_IF_NULL(fg);
      auto input = fg->get_return()->input(1);
      if (IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
        const auto &cnode = input->cast<CNodePtr>();
        for (size_t i = 0; i < cnode->inputs().size(); ++i) {
          if (HasAtomic(cnode->input(i))) {
            g_output_idx.push_back(i - 1);
          }
        }
      } else if (HasAtomic(input)) {
        g_output_idx.push_back(0);
      }

      if (!g_output_idx.empty()) {
        auto zero_node = CreateTbeAtomicCleanNode(kernel_graph, node);
        auto depend = kernel_graph->NewCNode({NewValueNode(prim::kPrimDepend), node->input(1), zero_node});
        std::vector<AnfNodePtr> new_input = node->inputs();
        new_input[1] = depend;
        auto new_cnode = std::make_shared<CNode>(new_input, kernel_graph);
        // Set abstract
        new_cnode->set_abstract(node->abstract());
        // Set kernel info
        new_cnode->set_kernel_info(node->kernel_info_ptr());
        mng->Replace(node, new_cnode);
        g_output_idx.clear();
      }
    }
  }
}
}  // namespace opt
}  // namespace mindspore
