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

#include "pre_activate/mem_reuse/mem_reuse.h"
#include <algorithm>
#include <memory>
#include "pre_activate/mem_reuse/mem_reuse_checker.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace memreuse {
bool MemReuseUtil::InitDynamicOutputKernelRef() {
  int index = util_index_;
  auto kernel_cnodes = graph_->execution_order();
  if (kernel_cnodes.empty()) {
    return true;
  }
  int kernel_out_ref_num = 0;
  for (auto &kernel_cnode : kernel_cnodes) {
#ifdef MEM_REUSE_DEBUG
    MemReuseChecker::GetInstance().CheckSignalOps(kernel_cnode);
#endif
    if (kernel_cnode == nullptr) {
      return false;
    }
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel_cnode);
    if (kernel_mod == nullptr) {
      return false;
    }
    auto key = kernel_cnode.get();
    // for every apply_kernel to set new output
    auto iter = kernel_output_refs_.find(key);
    if (iter == kernel_output_refs_.end()) {
      auto output_sizes = kernel_mod->GetOutputSizeList();
      KernelRefCountPtrList kernel_refs;
      for (auto size : output_sizes) {
        total_dy_size_ += size;
        // do not MallocDynamicMem just record this
        KernelRefCountPtr kernel_ref = std::make_shared<KernelRefCount>();
        index++;
        auto curr_stream_id = AnfAlgo::GetStreamId(kernel_cnode);
        kernel_ref->stream_id_ = curr_stream_id;
        kernel_ref->SetKernelRefCountInfo(index, size, kDynamicRefCount);
        kernel_refs.push_back(kernel_ref);
        kernel_out_ref_num++;
        total_refs_list_.push_back(kernel_ref);
      }
      if (!kernel_refs.empty()) {
        kernel_output_refs_[key] = kernel_refs;
      }
    }
  }
  return true;
}

bool MemReuseUtil::InitDynamicWorkspaceKernelRef() {
  int WkIndex = util_index_;
  auto kernel_cnodes = graph_->execution_order();
  if (kernel_cnodes.empty()) {
    return true;
  }
  for (auto &kernel_cnode : kernel_cnodes) {
    if (kernel_cnode == nullptr) {
      return false;
    }
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel_cnode);
    if (kernel_mod == nullptr) {
      return false;
    }
    auto key = kernel_cnode.get();
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    KernelRefCountPtrList workspace_kernel_refs;
    for (auto size : workspace_sizes) {
      total_workspace_size_ += size;
      ++WkIndex;
      KernelRefCountPtr workspace_ref = std::make_shared<KernelRefCount>();
      workspace_ref->SetKernelRefCountInfo(WkIndex, size, kDynamicRefCount);
      workspace_kernel_refs.push_back(workspace_ref);
      // total wk ref
      total_wk_ref_list_.push_back(workspace_ref);
    }
    if (!workspace_kernel_refs.empty()) {
      // every key index wk_refs
      kernel_workspace_refs_[key] = workspace_kernel_refs;
    }
  }
  return true;
}

bool MemReuseUtil::InitDynamicKernelRef(const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  graph_ = graph;
  is_all_nop_node_ = opt::IsAllNopNode(graph);
  if (!InitDynamicOutputKernelRef()) {
    MS_LOG(INFO) << "InitDynamicOutputKernelRef fail";
    return false;
  }
  if (!InitDynamicWorkspaceKernelRef()) {
    MS_LOG(INFO) << "InitDynamicWorkspaceKernelRef fail";
    return false;
  }
  return true;
}

// set longest worspace list && largest workspace sizes
void MemReuseUtil::SetWorkSpaceList() {
  int max_list_size = 0;
  std::vector<size_t> total_sizes;
  std::vector<size_t> max_list;
  auto kernel_cnodes = graph_->execution_order();
  for (auto &kernel_cnode : kernel_cnodes) {
    MS_EXCEPTION_IF_NULL(kernel_cnode);
    auto cnode_key = kernel_cnode.get();
    auto cnode_iter = kernel_workspace_refs_.find(cnode_key);
    if (cnode_iter != kernel_workspace_refs_.end()) {
      auto kernel_refs = cnode_iter->second;
      std::vector<size_t> current_list;
      for (size_t i = 0; i < kernel_refs.size(); ++i) {
        auto size = kernel_refs[i]->size_;
        current_list.push_back(size);
      }
      if (max_list_size < SizeToInt(current_list.size())) {
        max_list_size = SizeToInt(current_list.size());
      }
      (void)std::copy(current_list.begin(), current_list.end(), std::back_inserter(total_sizes));
    }
  }
  sort(total_sizes.rbegin(), total_sizes.rend());
  max_list.resize(IntToSize(max_list_size));
  if (SizeToInt(total_sizes.size()) < max_list_size) {
    MS_LOG(EXCEPTION) << "total workspace size is less than required max list size";
  }
  max_list.assign(total_sizes.begin(), total_sizes.begin() + max_list_size);
  for (auto &ma : max_list) {
    total_reuseworkspace_size_ += ma;
  }
  max_workspace_size_ = max_list_size;
  max_workspace_list_ = max_list;
}

void MemReuseUtil::SetInputMap(const CNodePtr &kernel, KernelDef *kernel_def_ptr) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_def_ptr);
  auto key = kernel.get();
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
    auto ref_ptr = GetKernelInputRef(kernel, i);
    if (ref_ptr != nullptr) {
      if (ref_ptr->reftype() == kStaticRefCount) {
        continue;
      } else if (ref_ptr->reftype() == kDynamicRefCount) {
        auto iter = kernel_def_ptr->inputs_.find(key);
        if (iter == kernel_def_ptr->inputs_.end()) {
          kernel_def_ptr->inputs_[key].push_back(ref_ptr);
        } else {
          iter->second.push_back(ref_ptr);
        }
      }
    }
  }
}

void MemReuseUtil::SetOutputMap(const CNodePtr &kernel, KernelDef *kernel_def_ptr) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_def_ptr);
  auto key = kernel.get();
  auto iter = kernel_def_ptr->outputs_.find(key);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  for (size_t k = 0; k < kernel_mod->GetOutputSizeList().size(); ++k) {
    KernelRefCountPtr kernel_ref = kernel_output_refs_[key][k];
    if (iter == kernel_def_ptr->outputs_.end()) {
      kernel_def_ptr->outputs_[key].push_back(kernel_ref);
    } else {
      iter->second.push_back(kernel_ref);
    }
  }
}

void MemReuseUtil::SetWkMap(const CNodePtr &kernel, KernelDef *kernel_def_ptr) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_def_ptr);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto key = kernel.get();
  for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
    if (kernel_workspace_refs_.find(key) != kernel_workspace_refs_.end()) {
      auto wk_refs = kernel_workspace_refs_[key];
      if (i < wk_refs.size()) {
        auto wk_ref = wk_refs[i];
        kernel_def_ptr->wk_space_[key].push_back(wk_ref);
      } else {
        MS_LOG(EXCEPTION) << "current index: " << i << " larger than wk_refs size " << wk_refs.size();
      }
    } else {
      MS_LOG(EXCEPTION) << "kernel_workspace_refs_ init error";
    }
  }
}

KernelRefCountPtr MemReuseUtil::GetRef(const AnfNodePtr &node, int output_idx) {
  if (node == nullptr) {
    MS_LOG(EXCEPTION) << "The node pointer is a nullptr.";
  }
  if (node->isa<CNode>()) {
    auto ak_node = node->cast<CNodePtr>();
    auto key = ak_node.get();
    MemReuseChecker::GetInstance().CheckOutRef(kernel_output_refs_, ak_node, IntToSize(output_idx));
    return kernel_output_refs_[key][IntToSize(output_idx)];
  }
  return nullptr;
}

KernelRefCountPtr MemReuseUtil::GetKernelInputRef(const CNodePtr &kernel, size_t input_idx) {
  if (input_idx >= AnfAlgo::GetInputTensorNum(kernel)) {
    MS_LOG(EXCEPTION) << "Input index " << input_idx << " is larger than input number "
                      << AnfAlgo::GetInputTensorNum(kernel);
  }
  auto input_node = kernel->input(input_idx + 1);
  // Graph may be all nop nodes and not remove nop node, so this can not skip nop node.
  session::KernelWithIndex kernel_input;
  if (is_all_nop_node_) {
    // The graph does not remove the nop node.
    kernel_input = AnfAlgo::VisitKernelWithReturnType(input_node, 0, false);
  } else {
    // The graph removes the nop node.
    kernel_input = AnfAlgo::VisitKernelWithReturnType(input_node, 0, true);
  }
  if (IsPrimitive(kernel_input.first, prim::kPrimMakeTuple)) {
    MS_LOG(EXCEPTION) << "Input node [" << input_node->DebugString() << "]'s input " << input_idx << " is MakeTuple";
  }
  auto result = GetRef(kernel_input.first, SizeToInt(kernel_input.second));
  return result;
}

void MemReuseUtil::SetKernelDefMap() {
  auto kernel_cnodes = graph_->execution_order();
  for (auto &kernel : kernel_cnodes) {
    KernelDefPtr kernel_def_ptr = std::make_shared<KernelDef>();
    kernel_def_ptr->set_kernel_name(AnfAlgo::GetCNodeName(kernel));
    kernel_def_ptr->set_scope_full_name(kernel->fullname_with_scope());
    kernel_def_ptr->set_stream_id(AnfAlgo::GetStreamId(kernel));
    SetInputMap(kernel, kernel_def_ptr.get());
    SetOutputMap(kernel, kernel_def_ptr.get());
    SetWkMap(kernel, kernel_def_ptr.get());
    auto key = kernel.get();
    kernel_def_ptr->set_input_refs(kernel_def_ptr->inputs_[key]);
    kernel_def_ptr->set_output_refs(kernel_def_ptr->outputs_[key]);
    kernel_def_ptr_list_.push_back(kernel_def_ptr);
    kernel_map_[key] = kernel_def_ptr;
  }
  SetKernelDefInputs();
}

void MemReuseUtil::SetKernelDefInputs() {
  for (const auto &kernel : graph_->execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto key = kernel.get();
    // find kernel_def according to cnode addr
    auto iter = kernel_map_.find(key);
    if (iter == kernel_map_.end()) {
      MS_LOG(EXCEPTION) << "kernel [" << kernel->fullname_with_scope() << "] is not init.";
    }
    auto kernel_def = iter->second;
    for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
      auto ref_ptr = GetKernelInputRef(kernel, i);
      if (ref_ptr != nullptr) {
        // set the inputs of this kernel_def
        auto input_node = AnfAlgo::GetInputNode(kernel, i);
        // Graph may be all nop nodes and not remove nop node, so this can not skip nop node.
        session::KernelWithIndex input;
        if (is_all_nop_node_) {
          // The graph does not remove the nop node.
          input = AnfAlgo::VisitKernelWithReturnType(input_node, 0, false);
        } else {
          // The graph removes the nop node.
          input = AnfAlgo::VisitKernelWithReturnType(input_node, 0, true);
        }
        if (IsPrimitive(input.first, prim::kPrimMakeTuple)) {
          MS_LOG(EXCEPTION) << "Input node [" << input_node->DebugString() << "]'s input " << i << " is MakeTuple";
        }
        auto input_key = (input.first).get();
        auto input_iter = kernel_map_.find(input_key);
        if (input_iter == kernel_map_.end()) {
          MS_LOG(EXCEPTION) << "kernel [" << (input.first)->fullname_with_scope() << "] is not init.";
        }
        kernel_def->InsertInputKernel(input_iter->second);
      }
    }
  }
}

void MemReuseUtil::SetReuseRefCount() {
  auto kernels = graph_->execution_order();
  for (auto &kernel : kernels) {
    auto key = kernel.get();
    for (auto &def : kernel_def_ptr_list_) {
      auto iter = def->inputs_.find(key);
      if (iter != def->inputs_.end()) {
        for (auto &input : iter->second) {
          input->ref_count_++;
          input->ref_count_dynamic_use_++;
        }
      }
    }
  }
}

void MemReuseUtil::SetSummaryNodesRefCount() {
  bool summary_exist = graph_->summary_node_exist();
  if (!summary_exist) {
    return;
  }

  auto summary_nodes = graph_->summary_nodes();
  if (summary_nodes.empty()) {
    return;
  }

  for (auto &node_item : summary_nodes) {
    auto node = node_item.second.first;
    size_t index = IntToSize(node_item.second.second);
    MS_LOG(INFO) << "set summary node's ref count, node: " << node->fullname_with_scope() << " index: " << index;
    if (kernel_output_refs_.find(node.get()) != kernel_output_refs_.end()) {
      KernelRefCountPtr kernel_ref = kernel_output_refs_[node.get()][index];
      kernel_ref->ref_count_ = kMaxRefCount;
      kernel_ref->ref_count_dynamic_use_ = kMaxRefCount;
    } else {
      MS_LOG(WARNING) << "can't find summary node's kernel_def " << node->fullname_with_scope();
    }
  }
#ifdef MEM_REUSE_DEBUG
  auto graph = *graph_;
  MemReuseChecker::GetInstance().CheckMemReuseIR(total_refs_list_, kernel_def_ptr_list_, &graph);
#endif
}

void MemReuseUtil::SetGraphOutputRefCount() {
  auto nodes = AnfAlgo::GetAllOutput(graph_->output(), {prim::kPrimTupleGetItem});
  for (const auto &node : nodes) {
    session::KernelWithIndex kernel_input;
    if (is_all_nop_node_) {
      // The graph does not remove the nop node.
      kernel_input = AnfAlgo::VisitKernelWithReturnType(node, 0, false);
    } else {
      // The graph removes the nop node.
      kernel_input = AnfAlgo::VisitKernelWithReturnType(node, 0, true);
    }
    MS_EXCEPTION_IF_NULL(kernel_input.first);
    if (!kernel_input.first->isa<CNode>() || !AnfAlgo::IsRealKernel(kernel_input.first)) {
      continue;
    }
    auto ak_node = kernel_input.first->cast<CNodePtr>();
    auto key = ak_node.get();
    auto iter = kernel_output_refs_.find(key);
    if ((iter != kernel_output_refs_.end()) && (kernel_input.second < iter->second.size())) {
      auto kernel_ref_count_ptr = kernel_output_refs_[key][kernel_input.second];
      MS_EXCEPTION_IF_NULL(kernel_ref_count_ptr);
      kernel_ref_count_ptr->ref_count_ = kMaxRefCount;
      kernel_ref_count_ptr->ref_count_dynamic_use_ = kMaxRefCount;
    }
  }
#ifdef MEM_REUSE_DEBUG
  auto graph = *graph_;
  MemReuseChecker::GetInstance().CheckMemReuseIR(total_refs_list_, kernel_def_ptr_list_, &graph);
#endif
}

void MemReuseUtil::ResetDynamicUsedRefCount() {
  for (auto iter = kernel_output_refs_.begin(); iter != kernel_output_refs_.end(); ++iter) {
    for (auto &ref_count : iter->second) {
      MS_EXCEPTION_IF_NULL(ref_count);
      ref_count->ref_count_dynamic_use_ = ref_count->ref_count_;
    }
  }
}

void MemReuseUtil::SetAllInfo(KernelGraph *graph) {
  if (!InitDynamicKernelRef(graph)) {
    MS_LOG(EXCEPTION) << "Init ReuseAssignDynamicMemory Fault";
  }
  SetKernelDefMap();
  SetReuseRefCount();
  SetSummaryNodesRefCount();
  SetWorkSpaceList();
#ifdef MEM_REUSE_DEBUG
  MemReuseChecker::GetInstance().CheckMemReuseIR(total_refs_list_, kernel_def_ptr_list_, graph);
#endif
}

uint8_t *MemReuseUtil::GetNodeOutputPtr(const AnfNodePtr &node, size_t index) const {
  auto key = node.get();
  auto iter = kernel_output_refs_.find(key);
  uint8_t *ptr = nullptr;
  if (iter != kernel_output_refs_.end()) {
    if (index >= iter->second.size()) {
      MS_LOG(EXCEPTION) << "index:[" << index << "] is larger than it's workspace size:[" << iter->second.size() << "]";
    }
    auto output_ref = iter->second[index];
    ptr = mem_base_ + output_ref->offset_;
  } else {
    MS_LOG(EXCEPTION) << "node [" << AnfAlgo::GetCNodeName(node) << "] don't exist in kernel_output_refs";
  }
  return ptr;
}

uint8_t *MemReuseUtil::GetNodeWorkSpacePtr(const AnfNodePtr &node, size_t index) const {
  auto key = node.get();
  auto iter = kernel_workspace_refs_.find(key);
  uint8_t *ptr = nullptr;
  if (iter != kernel_workspace_refs_.end()) {
    if (index >= iter->second.size()) {
      MS_LOG(EXCEPTION) << "index:[" << index << "] is larger than it's workspace size:[" << iter->second.size() << "]";
    }
    auto wk_ref = iter->second[index];
    ptr = mem_base_ + wk_ref->offset_;
  }
  return ptr;
}
}  // namespace memreuse
}  // namespace mindspore
