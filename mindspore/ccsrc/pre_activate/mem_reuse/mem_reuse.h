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

#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_REUSE_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_REUSE_H_
#include <map>
#include <memory>
#include <vector>
#include "pre_activate/mem_reuse/kernel_refcount.h"
#include "session/anf_runtime_algorithm.h"
#include "session/kernel_graph.h"
#include "kernel/tbe/tbe_utils.h"
using mindspore::kernel::tbe::TbeUtils;
namespace mindspore {
namespace memreuse {
static constexpr int kMaxRefCount = 9999;
static constexpr size_t kDefaultMemAlignSize = 512;
static constexpr size_t kAttAlignSize = 31;
static constexpr int kInvalidIndex = -2;

using KernelDefPtrMaps = std::vector<mindspore::memreuse::KernelDefPtr>;
using KernelRefs = std::map<KernelKey, KernelRefCountPtrList>;

using KernelGraph = mindspore::session::KernelGraph;

class MemReuseUtil {
 public:
  KernelRefs kernel_output_refs_;
  KernelRefCountPtrList total_refs_list_;
  KernelRefCountPtrList total_wk_ref_list_;
  KernelRefs kernel_workspace_refs_;
  MemReuseUtil() : util_index_(kInitIndex), graph_(nullptr) {}
  ~MemReuseUtil() {
    if (graph_ != nullptr) {
      graph_ = nullptr;
    }
    MS_LOG(INFO) << "Total Dynamic Memory Size:  " << total_dy_size_;
    MS_LOG(INFO) << "Total WorkSpace Memory Size: " << total_workspace_size_;
    MS_LOG(INFO) << "Total Reused WorkSpafce Memory Size: " << total_reuseworkspace_size_;
  }

  void SetAllInfo(KernelGraph *graph);
  bool InitDynamicOutputKernelRef();
  bool InitDynamicWorkspaceKernelRef();
  bool InitDynamicKernelRef(const KernelGraph *graph);
  void SetWorkSpaceList();
  void SetKernelDefMap();
  void SetInputMap(const CNodePtr &kernel, KernelDef *kernel_def_ptr);
  void SetOutputMap(const CNodePtr &kernel, KernelDef *kernel_def_ptr);
  void SetWkMap(const CNodePtr &kernel, KernelDef *kernel_def_ptr);
  void SetReuseRefCount();
  // Set the reference count of graph output specially.
  void SetGraphOutputRefCount();

  KernelRefCountPtr GetRef(const AnfNodePtr &node, int output_idx);
  KernelRefCountPtr GetKernelInputRef(const CNodePtr &kernel, size_t input_idx);
  KernelRefCountPtrList total_refs_list() const { return total_refs_list_; }
  KernelRefCountPtrList total_wk_ref_list() const { return total_wk_ref_list_; }
  KernelDefPtrMaps kernel_def_ptr_list() const { return kernel_def_ptr_list_; }
  int max_workspace_size() const { return max_workspace_size_; }
  std::vector<size_t> max_workspace_list() const { return max_workspace_list_; }
  void set_total_refs_list(const KernelRefCountPtrList &total_refs_list) { total_refs_list_ = total_refs_list; }
  void set_kernel_def_ptr_list(const KernelDefPtrMaps &kernel_def_ptr_list) {
    kernel_def_ptr_list_ = kernel_def_ptr_list;
  }
  void set_mem_base(uint8_t *mem_base) { mem_base_ = mem_base; }
  uint8_t *GetNodeOutputPtr(const AnfNodePtr &node, size_t index) const;
  uint8_t *GetNodeWorkSpacePtr(const AnfNodePtr &node, size_t index) const;

 private:
  int util_index_;
  const KernelGraph *graph_;
  KernelRefCountPtrList ref_list_;
  KernelDefPtrMaps kernel_def_ptr_list_;
  KernelRefCountPtrList last_ref_list_;
  int max_workspace_size_ = 0;
  std::vector<size_t> max_workspace_list_;
  size_t total_dy_size_ = 0;
  size_t total_workspace_size_ = 0;
  size_t total_reuseworkspace_size_ = 0;
  uint8_t *mem_base_{nullptr};
};
using MemReuseUtilPtr = std::shared_ptr<MemReuseUtil>;
}  // namespace memreuse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_REUSE_H_
