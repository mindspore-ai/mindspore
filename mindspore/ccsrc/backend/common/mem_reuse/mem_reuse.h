/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_REUSE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_REUSE_H_
#include <map>
#include <memory>
#include <vector>
#include "utils/hash_map.h"
#include "backend/common/mem_reuse/kernel_refcount.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_graph.h"
#include "utils/ms_context.h"
namespace mindspore {
namespace memreuse {
static constexpr int kMaxRefCount = 9999;
static constexpr size_t kDefaultMemAlignSize = 512;
static constexpr size_t kAttAlignSize = 31;
static constexpr int kInvalidIndex = -2;

using KernelDefPtrMaps = std::vector<mindspore::memreuse::KernelDefPtr>;
using KernelRefs = std::map<KernelKey, KernelRefCountPtrList>;

using KernelGraph = mindspore::session::KernelGraph;

class BACKEND_EXPORT MemReuseUtil {
 public:
  KernelRefCountPtrList total_refs_list_;
  KernelRefCountPtrList total_wk_ref_list_;
  MemReuseUtil() : util_index_(kInitIndex), graph_(nullptr), is_all_nop_node_(false) {}
  ~MemReuseUtil() {
    if (graph_ != nullptr) {
      graph_ = nullptr;
    }
    MS_LOG(INFO) << "Total Dynamic Memory Size:  " << total_dy_size_;
    MS_LOG(INFO) << "Total WorkSpace Memory Size: " << total_workspace_size_;
    MS_LOG(INFO) << "Total Reused WorkSpace Memory Size: " << total_reuseworkspace_size_;
  }

  void SetAllInfo(const KernelGraph *graph);
  bool InitDynamicOutputKernelRef();
  bool InitDynamicWorkspaceKernelRef();
  bool InitDynamicKernelRef(const KernelGraph *graph);
  void SetWorkSpaceList();
  void SetKernelDefMap();
  void SetInputMap(const CNodePtr &kernel, KernelDef *kernel_def_ptr);
  void SetOutputMap(const CNodePtr &kernel, KernelDef *kernel_def_ptr);
  void SetWkMap(const CNodePtr &kernel, KernelDef *kernel_def_ptr);
  void SetKernelDefInputs();
  void SetReuseRefCount();
#ifndef ENABLE_SECURITY
  void SetSummaryNodesRefCount();
#endif
  void SetRefNodesInputRefCount();
  // Set the reference count of graph output specially.
  void SetGraphOutputRefCount();
  // Reset the dynamic used reference count by ref_count_.
  void ResetDynamicUsedRefCount();

  KernelRefCountPtr GetRef(const AnfNodePtr &node, size_t output_idx);
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
  bool is_all_nop_node() const { return is_all_nop_node_; }
  session::KernelWithIndex VisitKernelWithReturnType(const AnfNodePtr &node, size_t i, bool skip_nop_node);

 private:
  KernelRefs kernel_output_refs_;
  KernelRefs kernel_workspace_refs_;
  int util_index_;
  const KernelGraph *graph_;
  bool is_all_nop_node_;
  KernelRefCountPtrList ref_list_;
  KernelDefPtrMaps kernel_def_ptr_list_;
  KernelRefCountPtrList last_ref_list_;
  int max_workspace_size_ = 0;
  std::vector<size_t> max_workspace_list_;
  size_t total_dy_size_ = 0;
  size_t total_workspace_size_ = 0;
  size_t total_reuseworkspace_size_ = 0;
  uint8_t *mem_base_{nullptr};
  // kernel_map_: key is the AnfNodePtr addr, value is the KernelDef
  std::map<KernelKey, KernelDefPtr> kernel_map_;

  bool enable_visit_kernel_cache_{false};

  mindspore::HashMap<AnfNodePtr, session::KernelWithIndex> visit_kernel_with_return_type_in0pos_cache_;
  mindspore::HashMap<AnfNodePtr, session::KernelWithIndex> visit_kernel_with_return_type_in0pos_skip_nop_cache_;
};
using MemReuseUtilPtr = std::shared_ptr<MemReuseUtil>;

enum Status { kUnused, kReused };
enum MemType { kNew, kInStreamReuse, kBetweenStreamReuse, kKernelDependenceReuse };
class Membuf {
 public:
  Membuf() = default;
  Membuf(Status status, size_t size, size_t offset, int index, MemType type, const KernelDefPtr &used_kernel)
      : status_(status), size_(size), offset_(offset), index_(index), type_(type), used_kernel_(used_kernel) {}
  ~Membuf() = default;
  // Memory block status flags
  Status status_ = kUnused;
  size_t size_{0};
  size_t offset_{0};
  // Store the tensor index stored in this memory block at a certain moment
  int index_{0};
  MemType type_{kNew};
  KernelDefPtr used_kernel_;
};
using MembufPtr = std::shared_ptr<Membuf>;

}  // namespace memreuse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_REUSE_H_
