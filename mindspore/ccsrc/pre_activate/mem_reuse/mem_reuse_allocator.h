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
#include "pre_activate/mem_reuse/kernel_refcount.h"
#include "pre_activate/mem_reuse/mem_reuse.h"
#include "pre_activate/mem_reuse/stream_reuse.h"

namespace mindspore {
namespace memreuse {
static constexpr int kWkIndexFactor = -1000;
static constexpr int kDyFac = -1;
static constexpr int kWkFac = 1;
static constexpr size_t kTotalSize = 0;
enum Status { kUnused, kReused };
class Membuf {
 public:
  Membuf() = default;
  Membuf(uint32_t stream_id, Status status, size_t size, size_t offset, int index)
      : stream_id_(stream_id), status_(status), size_(size), offset_(offset), index_(index) {}
  ~Membuf() = default;
  // Memory block status flags
  std::set<uint32_t> called_stream_ids_;
  uint32_t stream_id_{0};
  Status status_ = kUnused;
  size_t size_{0};
  size_t offset_{0};
  // Store the tensor index stored in this memory block at a certain moment
  int index_{0};
};
using MembufPtr = std::shared_ptr<Membuf>;

class BestFitMemReuse {
 public:
  BestFitMemReuse() = default;
  ~BestFitMemReuse() { membuf_ptr_list_.clear(); }
  // Init all information need by memory reuse
  void InitMemReuseInfo(const MemReuseUtil *mem_reuse_util_ptr);
  bool CheckMembufIndx(const std::vector<MembufPtr> &membuf_ptr_list, size_t check_idx) const;
  bool IsMembufListEmpty(const std::vector<MembufPtr> &membuf_ptr_list) const;
  void AssignNodeWkOffset(const KernelDef *kernel_def_ptr);
  void ReleasePreNodeWkSpace(const KernelDef *kernel_def_ptr);
  // void assign node output tensor memory offset
  void AssignNodeOutputOffset(const KernelDef *kernel_def_ptr);
  void ReleaseParallStream();
  // update node input tensor refcount, and membuf list status
  void UpdateNodeInputAndMembuf(const KernelDef *kernel_def_ptr);
  // check node output tensor which refcount is equal to zero
  void ReleaseNodeUnusedOutput(const KernelDef *kernel_def_ptr);
  // If there are memory blocks that can be reused
  void ReuseExistMembuf(KernelRefCount *tensor_desc, size_t membuf_index, int flag);
  // Save memory blocks that can be reused to the map
  std::map<size_t, size_t> GetReusableMembufMap(size_t tensor_size);
  // Update the status of the reused memory block
  void UpdateMembufInfo(KernelRefCount *tensor_desc, Membuf *membuf, int flag);
  // If the size of the memory block is greater than the size of the tensor, split the extra memory
  void SplitMembuf(const KernelRefCount *tensor_desc, size_t membuf_index);
  // Determine if the memory block needs to be split
  bool IsSplit(size_t tensor_size, size_t membuf_size) const;
  // If there is no memory block that can be reused, add a new memory block at the end
  void AddNewMembufPtr(KernelRefCount *tensor_desc, int flag);
  // Merge unused membuf
  void ReleaseMembuf(size_t tensor_index, int flag);
  bool HasParallelId(const std::set<uint32_t> &called_ids, uint32_t curr_id);
  void MergeCalledIds(const Membuf *membuf_target, Membuf *membuf);
  // Memory address alignment 512
  size_t AlignMemorySize(size_t size) const;
  int GetFacIdx(size_t real_idx, int flag = kDyFac) const;
  int GetRealIdx(int fac_idx, int flag = kDyFac) const;
  size_t FindIndx(const std::vector<MembufPtr> &membuf_ptr_list, int fac_idx) const;
  void CheckTensorIndex(int tensor_index) const;
  // Memory reuse main program entry
  void Reuse(const MemReuseUtil *mem_reuse_util_ptr);
  // Get the total memory that needs to be applied eventually
  size_t GetAllocatedSize();
  // If the target stream can be reused by current stream
  bool IsReusableStream(uint32_t curr_stream_id, uint32_t target_stream_id);
  // set tensor_def and op_def
  void set_tensor_ptr_list(const std::vector<KernelRefCountPtr> &tensor_ptr_list) {
    tensor_ptr_list_ = tensor_ptr_list;
  }
  void set_op_ptr_list(const std::vector<KernelDefPtr> &op_ptr_list) { op_ptr_list_ = op_ptr_list; }

 private:
  uint32_t current_stream_id_{0};
  // Save all tensor information
  std::vector<KernelRefCountPtr> tensor_ptr_list_;
  std::vector<KernelRefCountPtr> wk_tensor_list_;
  // Save all op information, including input and output tensor index
  std::vector<KernelDefPtr> op_ptr_list_;
  // Memory block information sequence, temporary variables
  std::vector<MembufPtr> membuf_ptr_list_;
  std::unordered_map<uint32_t, std::unordered_set<uint32_t>> parallel_streams_map_;
};
}  // namespace memreuse
}  // namespace mindspore
#endif  // #define MINDSPORE_CCSRC_PRE_ACTIVATE_MEM_REUSE_MEM_REUSE_ALLOCATOR_H_
