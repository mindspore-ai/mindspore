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

#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_REUSE_ALLOCATOR_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_REUSE_ALLOCATOR_H_
#include <cmath>
#include <map>
#include <list>
#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <utility>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <queue>
#include "pre_activate/mem_reuse/kernel_refcount.h"
#include "pre_activate/mem_reuse/mem_reuse.h"

namespace mindspore {
namespace memreuse {
static constexpr int kWorkspaceIndexFactor = -1000;
static constexpr int kDynamicMem = -1;
static constexpr int kWorkspaceMem = 1;
static constexpr size_t kTotalSize = 0;
enum Status { kUnused, kReused };
class Membuf {
 public:
  Membuf() = default;
  Membuf(Status status, size_t size, size_t offset, int index, const KernelDefPtr &used_kernel)
      : status_(status), size_(size), offset_(offset), index_(index), used_kernel_(used_kernel) {}
  ~Membuf() = default;
  // Memory block status flags
  Status status_ = kUnused;
  size_t size_{0};
  size_t offset_{0};
  // Store the tensor index stored in this memory block at a certain moment
  int index_{0};
  KernelDefPtr used_kernel_;
};
using MembufPtr = std::shared_ptr<Membuf>;

class BestFitMemReuse {
 public:
  BestFitMemReuse() = default;
  ~BestFitMemReuse() { membuf_ptr_list_.clear(); }
  /**
   * Init all information need by memory reuse
   * @param mem_reuse_util_ptr, initialize in the memreuse.cc
   */
  void InitMemReuseInfo(const MemReuseUtil *mem_reuse_util_ptr);
  void CheckMembufIndx(size_t check_idx) const;
  void AssignNodeWorkspaceOffset();
  void ReleasePreNodeWorkspace(const KernelDef *kernel_def_ptr);
  /**
   * Assign output tensor memory offset of current kernel
   */
  void AssignNodeOutputOffset();
  /**
   * Update input tensor's status of current kernel, and the status of membuf used by current kernel
   */
  void UpdateNodeInputAndMembuf();
  /**
   * Check whether to release the kernel output tensor which refcount is equal to zero
   */
  void ReleaseNodeUnusedOutput();
  /**
   * Reuse the exist membuf if possible
   * @param tensor_desc, the output tensor of current kernel
   * @param membuf_index, the index of membuf to be reused
   * @param flag
   */
  void ReuseExistMembuf(KernelRefCount *tensor_desc, size_t membuf_index, int flag);
  /**
   * Get the membuf that can be reused
   * @param tensor_size, the size of the tensor ready to assign memory offset
   * @return membuf map, key: the membuf size, value: the membuf index
   */
  std::map<size_t, size_t> GetReusableMembufMap(size_t tensor_size);
  /**
   * Update the status of the reused memory block
   * @param tensor_desc, the tensor ready to assign memory
   * @param membuf, the membuf to be reused
   * @param flag, distinguish dynamic memory and workspace
   */
  void UpdateMembufInfo(KernelRefCount *tensor_desc, Membuf *membuf, int flag);
  // If the size of the memory block is greater than the size of the tensor, split the extra memory
  void SplitMembuf(const KernelRefCount *tensor_desc, size_t membuf_index);
  // Determine if the memory block needs to be split
  bool IsSplit(size_t tensor_size, size_t membuf_size) const;
  // If there is no memory block that can be reused, add a new memory block at the end
  void AddNewMembufPtr(KernelRefCount *tensor_desc, int flag);
  // Merge unused membuf
  void ReleaseMembuf(size_t tensor_index, int flag);
  // Memory address alignment 512
  size_t AlignMemorySize(size_t size) const;
  int GetRealIndex(size_t index, int flag = kDynamicMem) const;
  size_t GetTensorIndex(int index) const;
  size_t GetWorkspaceIndex(int index) const;
  // Memory reuse main program entry
  void Reuse(const MemReuseUtil *mem_reuse_util_ptr);
  // Get the total memory that needs to be applied eventually
  size_t GetAllocatedSize();
  // return false, when the node output cannot be released
  bool IsRelease();
  /**
   * determine if the kernel_curr can reuse the output tensor add of kernel_prev
   * @param kernel_curr, current kernel
   * @param kernel_prev, the membuf used by this kernel
   * @return bool
   */
  bool IsUsable(const KernelDefPtr &kernel_curr, const KernelDefPtr &kernel_prev);
  /**
   * init the dependence of all kernels in the graph
   */
  void InitKernelDependence();
  // set tensor_def and op_def
  void set_tensor_ptr_list(const std::vector<KernelRefCountPtr> &tensor_ptr_list) {
    tensor_ptr_list_ = tensor_ptr_list;
  }
  void set_workspace_ptr_list(const std::vector<KernelRefCountPtr> &workspace_ptr_list) {
    wk_tensor_list_ = workspace_ptr_list;
  }
  void set_op_ptr_list(const std::vector<KernelDefPtr> &op_ptr_list) { op_ptr_list_ = op_ptr_list; }

 private:
  KernelDefPtr current_kernel_;
  // Save all tensor information
  std::vector<KernelRefCountPtr> tensor_ptr_list_;
  std::vector<KernelRefCountPtr> wk_tensor_list_;
  // Save all op information, including input and output tensor index
  std::vector<KernelDefPtr> op_ptr_list_;
  // Memory block information sequence, temporary variables
  std::vector<MembufPtr> membuf_ptr_list_;
  // kernel_front_map_, key: the kernel_def, value: kernels before this kernel_def
  std::map<KernelDefPtr, std::set<KernelDefPtr>> kernel_front_map_;
};
}  // namespace memreuse
}  // namespace mindspore
#endif  // #define MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_REUSE_ALLOCATOR_H_
