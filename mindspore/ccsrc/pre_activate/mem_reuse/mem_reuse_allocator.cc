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

#include "pre_activate/mem_reuse/mem_reuse_allocator.h"
#include <memory>
#include <algorithm>
#include <set>
#include "pre_activate/mem_reuse/mem_reuse.h"
#include "pre_activate/mem_reuse/mem_reuse_checker.h"

namespace mindspore {
namespace memreuse {
void BestFitMemReuse::InitMemReuseInfo(const MemReuseUtil *mem_reuse_util_ptr) {
  MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr);
  tensor_ptr_list_ = mem_reuse_util_ptr->total_refs_list();
  wk_tensor_list_ = mem_reuse_util_ptr->total_wk_ref_list();
  op_ptr_list_ = mem_reuse_util_ptr->kernel_def_ptr_list();
  // check info Correctness
  for (auto &tensor : tensor_ptr_list_) {
    tensor->size_ = AlignMemorySize(tensor->size_);
  }
  // align wk size to 512 && refcount == 1
  for (auto &wk : wk_tensor_list_) {
    wk->size_ = AlignMemorySize(wk->size_);
    wk->ref_count_ = 1;
  }
  auto stream_reuse = std::make_shared<StreamReuse>();
  stream_reuse->SetStreamReuseResource();
  parallel_streams_map_ = stream_reuse->parallel_streams_map();
}

bool BestFitMemReuse::CheckMembufIndx(const std::vector<MembufPtr> &membuf_ptr_list, size_t check_idx) const {
  return check_idx < membuf_ptr_list.size();
}

bool BestFitMemReuse::IsMembufListEmpty(const std::vector<MembufPtr> &membuf_ptr_list) const {
  return membuf_ptr_list.empty();
}

int BestFitMemReuse::GetFacIdx(size_t real_idx, int flag) const {
  if (flag == kDyFac) {
    return SizeToInt(real_idx);
  } else if (flag == kWkFac) {
    auto wk_fac_idx = kWkIndexFactor * SizeToInt(real_idx + 1);
    return wk_fac_idx;
  } else {
    MS_LOG(EXCEPTION) << "flag " << flag << " is invalid";
  }
}

int BestFitMemReuse::GetRealIdx(int fac_idx, int flag) const {
  // membuf index maybe invalid_index
  if (fac_idx == kInvalidIndex) {
    MS_LOG(EXCEPTION) << "this membuf index is invalid";
  }
  if (flag == kDyFac) {
    return fac_idx;
  } else if (flag == kWkFac) {
    if (fac_idx % 10 == 0) {
      auto wk_fac_idx = fac_idx / kWkIndexFactor + 1;
      return wk_fac_idx;
    } else {
      MS_LOG(EXCEPTION) << "fac_idx: " << fac_idx << "is invalid";
    }
  } else {
    MS_LOG(EXCEPTION) << "flag: " << flag << " is invalid";
  }
}

void BestFitMemReuse::AssignNodeOutputOffset(const KernelDef *kernel_def_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_def_ptr);
  for (auto &tensor_idx : kernel_def_ptr->GetOutputRefIndexs()) {
    CheckTensorIndex(tensor_idx);
    auto tensor_desc = tensor_ptr_list_[IntToSize(tensor_idx)];
    MS_EXCEPTION_IF_NULL(tensor_desc);
    auto reusable_membuf_map = GetReusableMembufMap(tensor_desc->size_);
    if (!reusable_membuf_map.empty()) {
      auto membuf_index = reusable_membuf_map.begin()->second;
      // find the best suitable membuf in membuf list, and reuse it
      ReuseExistMembuf(tensor_desc.get(), membuf_index, kDyFac);
    } else {
      // no membuf can reuse, add new membuf after the membuf_ptr_list
      AddNewMembufPtr(tensor_desc.get(), kDyFac);
#ifdef MEM_REUSE_DEBUG
      MemReuseChecker::GetInstance().IsAddNewMembuf_ = true;
#endif
    }
  }
}

void BestFitMemReuse::AssignNodeWkOffset(const KernelDef *kernel_def_ptr) {
  MS_EXCEPTION_IF_NULL(kernel_def_ptr);
  for (auto &wk_idx : kernel_def_ptr->GetWkRefIndexs()) {
    if (IntToSize(wk_idx) >= wk_tensor_list_.size()) {
      MS_LOG(EXCEPTION) << "wk_idx: " << wk_idx << " is invalid";
    }
    auto wk_ref = wk_tensor_list_[IntToSize(wk_idx)];
    MS_EXCEPTION_IF_NULL(wk_ref);
    auto re_wk_membuf_map = GetReusableMembufMap(wk_ref->size_);
    if (!re_wk_membuf_map.empty()) {
      auto membuf_index = re_wk_membuf_map.begin()->second;
      ReuseExistMembuf(wk_ref.get(), membuf_index, kWkFac);
    } else {
      AddNewMembufPtr(wk_ref.get(), kWkFac);
    }
  }
}
// releas pre node wk
void BestFitMemReuse::ReleasePreNodeWkSpace(const KernelDef *kernel_def_ptr) {
  for (auto &wk_idx : kernel_def_ptr->GetWkRefIndexs()) {
    auto wk_index = IntToSize(wk_idx);
    if (wk_index >= wk_tensor_list_.size()) {
      MS_LOG(EXCEPTION) << "wk_index: " << wk_index << " is larger than wk_tensor_list size" << wk_tensor_list_.size();
    }
    auto wk_tensor = wk_tensor_list_[wk_index];
    wk_tensor->ref_count_--;
    if (wk_tensor->ref_count_ == 0) {
      ReleaseMembuf(wk_index, kWkFac);
    }
  }
}

void BestFitMemReuse::ReuseExistMembuf(KernelRefCount *tensor_desc, size_t membuf_index, int flag) {
  MS_EXCEPTION_IF_NULL(tensor_desc);
  if (!CheckMembufIndx(membuf_ptr_list_, membuf_index)) {
    return;
  }
  auto membuf = membuf_ptr_list_[membuf_index];
  MS_EXCEPTION_IF_NULL(membuf);
  // first to split && then update membuf_info
  if (IsSplit(tensor_desc->size_, membuf->size_)) {
    // split the membuf, and insert a new membuf after this membuf
    SplitMembuf(tensor_desc, membuf_index);
  }
  // update membuf status, and set tensor offset
  UpdateMembufInfo(tensor_desc, membuf.get(), flag);
}

std::map<size_t, size_t> BestFitMemReuse::GetReusableMembufMap(size_t tensor_size) {
  std::map<size_t, size_t> size_map;
  for (size_t i = 0; i < membuf_ptr_list_.size(); ++i) {
    auto membuf = membuf_ptr_list_[i];
    auto called_ids = membuf->called_stream_ids_;
    auto index = i;
    bool IsMembufOk = membuf->status_ == kUnused && membuf->size_ >= tensor_size;
    bool has_parallel_id = HasParallelId(called_ids, current_stream_id_);
    if (IsMembufOk && !has_parallel_id) {
      (void)size_map.insert(std::make_pair(membuf->size_, index));
      break;
    }
  }
  return size_map;
}

void BestFitMemReuse::UpdateMembufInfo(KernelRefCount *tensor_desc, Membuf *membuf, int flag) {
  MS_EXCEPTION_IF_NULL(tensor_desc);
  MS_EXCEPTION_IF_NULL(membuf);
  auto fac_idx = GetFacIdx(IntToSize(tensor_desc->index_), flag);
  membuf->status_ = kReused;
  membuf->stream_id_ = current_stream_id_;
  // clear before called_ids
  membuf->called_stream_ids_.clear();
  (void)membuf->called_stream_ids_.insert(current_stream_id_);
  membuf->index_ = fac_idx;
  tensor_desc->offset_ = membuf->offset_;
}

bool BestFitMemReuse::IsSplit(size_t tensor_size, size_t membuf_size) const { return tensor_size < membuf_size; }

void BestFitMemReuse::SplitMembuf(const KernelRefCount *tensor_desc, size_t membuf_index) {
  MS_EXCEPTION_IF_NULL(tensor_desc);
  if (!CheckMembufIndx(membuf_ptr_list_, membuf_index)) {
    return;
  }
  auto membuf = membuf_ptr_list_[membuf_index];
  MS_EXCEPTION_IF_NULL(membuf);
  auto bias = membuf->size_ - tensor_desc->size_;
  membuf->size_ = tensor_desc->size_;
  // to check if spilt membuf can be merge
  auto new_membuf =
    std::make_shared<Membuf>(current_stream_id_, kUnused, bias, membuf->offset_ + membuf->size_, kInvalidIndex);
  (void)membuf_ptr_list_.insert(membuf_ptr_list_.begin() + SizeToInt(membuf_index + 1), new_membuf);
  MergeCalledIds(membuf.get(), new_membuf.get());
}

void BestFitMemReuse::AddNewMembufPtr(KernelRefCount *tensor_desc, int flag) {
  MS_EXCEPTION_IF_NULL(tensor_desc);
  size_t membuf_offset = std::accumulate(membuf_ptr_list_.begin(), membuf_ptr_list_.end(), IntToSize(0),
                                         [](size_t sum, MembufPtr &membuf) { return sum + membuf->size_; });
  size_t membuf_size = tensor_desc->size_;
  auto fac_idx = GetFacIdx(IntToSize(tensor_desc->index_), flag);
  auto membuf = std::make_shared<Membuf>(current_stream_id_, kReused, membuf_size, membuf_offset, fac_idx);
  membuf_ptr_list_.push_back(membuf);
  tensor_desc->offset_ = membuf_offset;
  (void)membuf->called_stream_ids_.insert(current_stream_id_);
}

void BestFitMemReuse::UpdateNodeInputAndMembuf(const KernelDef *kernel_def_ptr) {
  // process node input tensor
  for (const auto &tensor_idx : kernel_def_ptr->GetInputRefIndexs()) {
    auto tensor_index = IntToSize(tensor_idx);
    CheckTensorIndex(tensor_idx);
    auto tensor_desc = tensor_ptr_list_[tensor_index];
    auto fac_idx = GetFacIdx(tensor_index, kDyFac);
    MS_EXCEPTION_IF_NULL(tensor_desc);
    tensor_desc->ref_count_--;
    // find tensor_index -> membuf update it's called_ids
    for (size_t i = 0; i < membuf_ptr_list_.size(); ++i) {
      auto membuf = membuf_ptr_list_[i];
      // find it
      if (membuf->index_ == fac_idx) {
        (void)membuf->called_stream_ids_.insert(current_stream_id_);
        break;
      }
    }
    if (tensor_desc->ref_count_ == 0) {
      ReleaseMembuf(tensor_index, kDyFac);
    } else if (tensor_desc->ref_count_ < 0) {
      MS_LOG(EXCEPTION) << "tensor: " << tensor_desc->index_ << " refcount: " << tensor_desc->ref_count_
                        << " check error";
    }
  }
}

void BestFitMemReuse::ReleaseNodeUnusedOutput(const KernelDef *kernel_def_ptr) {
  for (auto &tensor_idx : kernel_def_ptr->GetOutputRefIndexs()) {
    auto tensor_index = IntToSize(tensor_idx);
    CheckTensorIndex(tensor_idx);
    auto tensor_desc = tensor_ptr_list_[tensor_index];
    MS_EXCEPTION_IF_NULL(tensor_desc);
    if (tensor_desc->ref_count_ == 0) {
      ReleaseMembuf(tensor_index, kDyFac);
    } else if (tensor_desc->ref_count_ < 0) {
      MS_LOG(EXCEPTION) << "tensor: " << tensor_desc->index_ << " refcount: " << tensor_desc->ref_count_
                        << " check error";
    }
  }
}

size_t BestFitMemReuse::FindIndx(const std::vector<MembufPtr> &membuf_ptr_list, int fac_idx) const {
  size_t membuf_index = membuf_ptr_list.size();
  for (size_t n = 0; n < membuf_ptr_list.size(); ++n) {
    auto membuf = membuf_ptr_list[n];
    MS_EXCEPTION_IF_NULL(membuf);
    if (membuf->index_ == fac_idx) {
      membuf_index = n;
      break;
    }
  }
  return membuf_index;
}

void BestFitMemReuse::ReleaseMembuf(size_t tensor_index, int flag) {
  auto fac_idex = GetFacIdx(tensor_index, flag);
  auto membuf_index = FindIndx(membuf_ptr_list_, fac_idex);
  if (!CheckMembufIndx(membuf_ptr_list_, membuf_index)) {
    return;
  }
  auto membuf = membuf_ptr_list_[membuf_index];
  MS_EXCEPTION_IF_NULL(membuf);
  membuf->status_ = kUnused;
  if (membuf_index != (membuf_ptr_list_.size() - 1)) {
    auto membuf_next = membuf_ptr_list_[membuf_index + 1];
    MS_EXCEPTION_IF_NULL(membuf_next);
    bool has_parallel_id = false;
    for (auto &cal_id : membuf->called_stream_ids_) {
      has_parallel_id = HasParallelId(membuf_next->called_stream_ids_, cal_id);
      if (has_parallel_id) {
        break;
      }
    }
    if (membuf_next->status_ == kUnused && !has_parallel_id) {
      membuf->size_ += membuf_next->size_;
      MergeCalledIds(membuf_next.get(), membuf.get());
      auto it = membuf_ptr_list_.begin() + SizeToInt(membuf_index + 1);
      (void)membuf_ptr_list_.erase(it);
    }
  }
  if (membuf_index != 0) {
    if (!CheckMembufIndx(membuf_ptr_list_, membuf_index - 1)) {
      return;
    }
    auto membuf_prev = membuf_ptr_list_[membuf_index - 1];
    MS_EXCEPTION_IF_NULL(membuf_prev);
    bool has_parallel_id = false;
    for (auto &cal_id : membuf->called_stream_ids_) {
      has_parallel_id = HasParallelId(membuf_prev->called_stream_ids_, cal_id);
      if (has_parallel_id) {
        break;
      }
    }
    if (membuf_prev->status_ == kUnused && !has_parallel_id) {
      membuf->size_ += membuf_prev->size_;
      membuf->offset_ = membuf_prev->offset_;
      MergeCalledIds(membuf_prev.get(), membuf.get());
      auto it = membuf_ptr_list_.begin() + SizeToInt(membuf_index - 1);
      (void)membuf_ptr_list_.erase(it);
    }
  }
}

bool BestFitMemReuse::HasParallelId(const std::set<uint32_t> &called_ids, uint32_t curr_id) {
  if (called_ids.empty()) {
    MS_LOG(EXCEPTION) << "There is a invalid WkMembuf,called_ids is empty";
  }
  for (auto item : called_ids) {
    if (!IsReusableStream(curr_id, item)) {
      return true;
    }
  }
  return false;
}

void BestFitMemReuse::MergeCalledIds(const Membuf *membuf_target, Membuf *membuf) {
  MS_EXCEPTION_IF_NULL(membuf_target);
  MS_EXCEPTION_IF_NULL(membuf);
  for (auto target : membuf_target->called_stream_ids_) {
    (void)membuf->called_stream_ids_.insert(target);
  }
}

void BestFitMemReuse::ReleaseParallStream() {
  std::vector<size_t> target_relea_idxs;
  for (size_t i = 0; i < membuf_ptr_list_.size(); ++i) {
    auto membuf = membuf_ptr_list_[i];
    if (membuf->status_ == kReused) {
      continue;
    }
    // for begin to end, so no need merge pre_membuf
    if (i != (membuf_ptr_list_.size() - 1)) {
      auto membuf_next = membuf_ptr_list_[i + 1];
      if (membuf_next->status_ == kReused) {
        continue;
      }
      MS_EXCEPTION_IF_NULL(membuf_next);
      // judge current id no parallel fro membuf && membuf_next
      bool has_parallel_id_crr = HasParallelId(membuf->called_stream_ids_, current_stream_id_);
      bool has_parallel_id_next = HasParallelId(membuf_next->called_stream_ids_, current_stream_id_);
      if (membuf->status_ == kUnused && membuf_next->status_ == kUnused && !has_parallel_id_crr &&
          !has_parallel_id_next) {
        membuf->size_ += membuf_next->size_;
        MergeCalledIds(membuf_next.get(), membuf.get());
        target_relea_idxs.push_back(i + 1);
      }
    }
  }
  // erase all target membuf
  std::vector<MembufPtr> membuf_ptr_list_tmp;
  for (size_t j = 0; j < membuf_ptr_list_.size(); ++j) {
    for (auto idx : target_relea_idxs) {
      if (j != idx) {
        membuf_ptr_list_tmp.push_back(membuf_ptr_list_[j]);
      }
    }
  }
  membuf_ptr_list_.clear();
  (void)std::copy(membuf_ptr_list_tmp.begin(), membuf_ptr_list_tmp.end(), back_inserter(membuf_ptr_list_));
}

size_t BestFitMemReuse::AlignMemorySize(size_t size) const {
  // memory size 512 align
  return (size + kDefaultMemAlignSize + kAttAlignSize) / kDefaultMemAlignSize * kDefaultMemAlignSize;
}

size_t BestFitMemReuse::GetAllocatedSize() {
  size_t AllocatedSize = kTotalSize;
  if (membuf_ptr_list_.empty()) {
    return AllocatedSize;
  }
  AllocatedSize = (*membuf_ptr_list_.rbegin())->offset_ + (*membuf_ptr_list_.rbegin())->size_;
  MS_LOG(INFO) << "MemReuse Allocated Dynamic Size: " << AllocatedSize;
  return AllocatedSize;
}

/**
 * parallel_streams_map: key, current_stream_id; value,  streams parallel to current stream
 * @param curr_stream_id
 * @param target_stream_id
 * @return bool, if the target stream can be reused by current stream
 */
bool BestFitMemReuse::IsReusableStream(uint32_t curr_stream_id, uint32_t target_stream_id) {
  auto iter_parall = parallel_streams_map_.find(curr_stream_id);
  if (parallel_streams_map_.empty() || (iter_parall == parallel_streams_map_.end())) {
    // no parallel stream exists
    return true;
  }
  auto curr_parallel_set = iter_parall->second;
  return curr_parallel_set.find(target_stream_id) == curr_parallel_set.end();
}

void BestFitMemReuse::CheckTensorIndex(int tensor_index) const {
  if (tensor_index < 0) {
    MS_LOG(EXCEPTION) << "warning, please check tensor info.";
  }
  if (IntToSize(tensor_index) >= tensor_ptr_list_.size()) {
    MS_LOG(EXCEPTION) << "invalid tensor index";
  }
}

void BestFitMemReuse::Reuse(const MemReuseUtil *mem_reuse_util_ptr) {
  MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr);
  InitMemReuseInfo(mem_reuse_util_ptr);
  KernelDefPtr pre_op = nullptr;
#ifdef MEM_REUSE_DEBUG
  size_t op_num = 0;
#endif
  for (const auto &op_def_ptr : op_ptr_list_) {
    current_stream_id_ = op_def_ptr->stream_id();
    // releas pre_op_def
    if (pre_op != nullptr) {
      ReleasePreNodeWkSpace(pre_op.get());
    }
    MemReuseChecker::GetInstance().IsAddNewMembuf_ = false;
    // process node output tensor
    AssignNodeOutputOffset(op_def_ptr.get());
#ifdef MEM_REUSE_DEBUG
    if (MemReuseChecker::GetInstance().IsAddNewMembuf_) {
      MemReuseChecker::GetInstance().SetAddNewMembuInfos(op_def_ptr.get(), membuf_ptr_list_, op_num);
    }
#endif
    // deal with current op'workspace
    AssignNodeWkOffset(op_def_ptr.get());
    pre_op = op_def_ptr;
    // update node input tensor refcount, and membuf list status
    UpdateNodeInputAndMembuf(op_def_ptr.get());
    // check node output tensor which refcount is equal to zero
#ifdef MEM_REUSE_DEBUG
    MemReuseChecker::GetInstance().SetMembuInfos(op_def_ptr.get(), membuf_ptr_list_);
    ++op_num;
#endif
  }
#ifdef MEM_REUSE_DEBUG
  MemReuseChecker::GetInstance().ExportMembufInfoIR();
  MemReuseChecker::GetInstance().ExportAddNewMmebufIR();
#endif
}
}  // namespace memreuse
}  // namespace mindspore
