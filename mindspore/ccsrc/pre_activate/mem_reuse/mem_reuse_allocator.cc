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
#include "pre_activate/mem_reuse/mem_reuse.h"
#include "pre_activate/mem_reuse/mem_reuse_checker.h"

namespace mindspore {
namespace memreuse {
void BestFitMemReuse::InitMemReuseInfo(const MemReuseUtil *mem_reuse_util_ptr) {
  MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr);
  set_tensor_ptr_list(mem_reuse_util_ptr->total_refs_list());
  set_workspace_ptr_list(mem_reuse_util_ptr->total_wk_ref_list());
  set_op_ptr_list(mem_reuse_util_ptr->kernel_def_ptr_list());
  // check info Correctness
  for (auto &tensor : tensor_ptr_list_) {
    tensor->size_ = AlignMemorySize(tensor->size_);
  }
  // align wk size to 512 && refcount == 1
  for (auto &wk : wk_tensor_list_) {
    wk->size_ = AlignMemorySize(wk->size_);
    wk->ref_count_ = 1;
  }
}

void BestFitMemReuse::InitKernelDependence() {
  for (const auto &kernel : op_ptr_list_) {
    std::set<KernelDefPtr> front;
    std::queue<KernelDefPtr> to_visit;
    to_visit.push(kernel);
    // find all kernels before current kernel
    while (!to_visit.empty()) {
      auto curr = to_visit.front();
      to_visit.pop();
      if (front.count(curr)) {
        continue;
      }
      front.insert(curr);
      auto iter = kernel_front_map_.find(curr);
      if (iter != kernel_front_map_.end()) {
        auto visited_front = iter->second;
        front.insert(visited_front.begin(), visited_front.end());
        continue;
      }
      for (const auto &input : curr->input_kernels()) {
        to_visit.push(input);
      }
    }
    kernel_front_map_[kernel] = front;
  }
}

bool BestFitMemReuse::IsUsable(const KernelDefPtr &kernel_curr, const KernelDefPtr &kernel_prev) {
  // determine whether the kernel_curr can reuse kernel_prev's output tensor membuf
  MS_EXCEPTION_IF_NULL(kernel_curr);
  MS_EXCEPTION_IF_NULL(kernel_prev);
  auto curr_stream_id = kernel_curr->stream_id();
  auto prev_stream_id = kernel_prev->stream_id();
  if (curr_stream_id == prev_stream_id) {
    return true;
  }
  auto iter = kernel_front_map_.find(kernel_curr);
  if (iter == kernel_front_map_.end()) {
    MS_LOG(EXCEPTION) << kernel_curr->scope_full_name() << " is not init.";
  }
  auto kernel_curr_front = iter->second;
  return kernel_curr_front.count(kernel_prev);
}

void BestFitMemReuse::AssignNodeOutputOffset() {
  for (auto &tensor_idx : current_kernel_->GetOutputRefIndexs()) {
    size_t index = GetTensorIndex(tensor_idx);
    auto tensor_desc = tensor_ptr_list_[index];
    MS_EXCEPTION_IF_NULL(tensor_desc);
    auto reusable_membuf_map = GetReusableMembufMap(tensor_desc->size_);
    if (!reusable_membuf_map.empty()) {
      auto membuf_index = reusable_membuf_map.begin()->second;
      // find the best suitable membuf in membuf list, and reuse it
      ReuseExistMembuf(tensor_desc.get(), membuf_index, kDynamicMem);
    } else {
      // no membuf can reuse, add new membuf after the membuf_ptr_list
      AddNewMembufPtr(tensor_desc.get(), kDynamicMem);
#ifdef MEM_REUSE_DEBUG
      MemReuseChecker::GetInstance().IsAddNewMembuf_ = true;
#endif
    }
  }
}

void BestFitMemReuse::AssignNodeWorkspaceOffset() {
  for (auto &wk_idx : current_kernel_->GetWorkspaceRefIndexs()) {
    size_t index = GetWorkspaceIndex(wk_idx);
    auto wk_ref = wk_tensor_list_[index];
    MS_EXCEPTION_IF_NULL(wk_ref);
    auto re_wk_membuf_map = GetReusableMembufMap(wk_ref->size_);
    if (!re_wk_membuf_map.empty()) {
      auto membuf_index = re_wk_membuf_map.begin()->second;
      ReuseExistMembuf(wk_ref.get(), membuf_index, kWorkspaceMem);
    } else {
      AddNewMembufPtr(wk_ref.get(), kWorkspaceMem);
    }
  }
}

void BestFitMemReuse::ReuseExistMembuf(KernelRefCount *tensor_desc, size_t membuf_index, int flag) {
  MS_EXCEPTION_IF_NULL(tensor_desc);
  CheckMembufIndx(membuf_index);
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
    auto index = i;
    bool is_membuf_ok = membuf->status_ == kUnused && membuf->size_ >= tensor_size;
    if (is_membuf_ok && IsUsable(current_kernel_, membuf->used_kernel_)) {
      (void)size_map.insert(std::make_pair(membuf->size_, index));
      break;
    }
  }
  return size_map;
}

void BestFitMemReuse::UpdateMembufInfo(KernelRefCount *tensor_desc, Membuf *membuf, int flag) {
  MS_EXCEPTION_IF_NULL(tensor_desc);
  MS_EXCEPTION_IF_NULL(membuf);
  auto real_index = GetRealIndex(IntToSize(tensor_desc->index_), flag);
  membuf->status_ = kReused;
  membuf->index_ = real_index;
  membuf->used_kernel_ = current_kernel_;
  tensor_desc->offset_ = membuf->offset_;
}

bool BestFitMemReuse::IsSplit(size_t tensor_size, size_t membuf_size) const { return tensor_size < membuf_size; }

void BestFitMemReuse::SplitMembuf(const KernelRefCount *tensor_desc, size_t membuf_index) {
  MS_EXCEPTION_IF_NULL(tensor_desc);
  CheckMembufIndx(membuf_index);
  auto membuf = membuf_ptr_list_[membuf_index];
  MS_EXCEPTION_IF_NULL(membuf);
  auto bias = membuf->size_ - tensor_desc->size_;
  membuf->size_ = tensor_desc->size_;
  // to check if spilt membuf can be merge
  auto new_membuf =
    std::make_shared<Membuf>(kUnused, bias, membuf->offset_ + membuf->size_, kInvalidIndex, current_kernel_);
  (void)membuf_ptr_list_.insert(membuf_ptr_list_.begin() + SizeToInt(membuf_index + 1), new_membuf);
}

void BestFitMemReuse::AddNewMembufPtr(KernelRefCount *tensor_desc, int flag) {
  MS_EXCEPTION_IF_NULL(tensor_desc);
  size_t membuf_offset = 0;
  if (!membuf_ptr_list_.empty()) {
    membuf_offset = membuf_ptr_list_.back()->offset_ + membuf_ptr_list_.back()->size_;
  }
  auto membuf_size = tensor_desc->size_;
  auto real_index = GetRealIndex(IntToSize(tensor_desc->index_), flag);
  auto membuf = std::make_shared<Membuf>(kReused, membuf_size, membuf_offset, real_index, current_kernel_);
  membuf_ptr_list_.push_back(membuf);
  tensor_desc->offset_ = membuf_offset;
}

void BestFitMemReuse::UpdateNodeInputAndMembuf() {
  // process node input tensor
  for (const auto &tensor_idx : current_kernel_->GetInputRefIndexs()) {
    size_t tensor_index = GetTensorIndex(tensor_idx);
    auto tensor_desc = tensor_ptr_list_[tensor_index];
    MS_EXCEPTION_IF_NULL(tensor_desc);
    tensor_desc->ref_count_--;
    if (tensor_desc->ref_count_ == 0) {
      ReleaseMembuf(tensor_index, kDynamicMem);
    } else if (tensor_desc->ref_count_ < 0) {
      MS_LOG(EXCEPTION) << "tensor: " << tensor_desc->index_ << " refcount: " << tensor_desc->ref_count_
                        << " check error";
    }
  }
}

void BestFitMemReuse::ReleaseNodeUnusedOutput() {
  for (auto &tensor_idx : current_kernel_->GetOutputRefIndexs()) {
    size_t tensor_index = GetTensorIndex(tensor_idx);
    auto tensor_desc = tensor_ptr_list_[tensor_index];
    MS_EXCEPTION_IF_NULL(tensor_desc);
    if (tensor_desc->ref_count_ == 0) {
      ReleaseMembuf(tensor_index, kDynamicMem);
    } else if (tensor_desc->ref_count_ < 0) {
      MS_LOG(EXCEPTION) << "tensor: " << tensor_desc->index_ << " refcount: " << tensor_desc->ref_count_
                        << " check error";
    }
  }
}

void BestFitMemReuse::ReleasePreNodeWorkspace(const KernelDef *kernel_def_ptr) {
  for (auto &workspace_index : kernel_def_ptr->GetWorkspaceRefIndexs()) {
    size_t index = GetWorkspaceIndex(workspace_index);
    auto wk_tensor = wk_tensor_list_[index];
    wk_tensor->ref_count_--;
    if (wk_tensor->ref_count_ == 0) {
      ReleaseMembuf(index, kWorkspaceMem);
    } else if (wk_tensor->ref_count_ < 0) {
      MS_LOG(EXCEPTION) << "tensor: " << wk_tensor->index_ << " refcount: " << wk_tensor->ref_count_ << " check error";
    }
  }
}

void BestFitMemReuse::ReleaseMembuf(size_t tensor_index, int flag) {
  if (membuf_ptr_list_.empty()) {
    return;
  }
  auto real_index = GetRealIndex(tensor_index, flag);
  auto membuf_iter = std::find_if(membuf_ptr_list_.begin(), membuf_ptr_list_.end(),
                                  [real_index](const MembufPtr &membuf) { return membuf->index_ == real_index; });
  if (membuf_iter == membuf_ptr_list_.end()) {
    return;
  }
  auto membuf = (*membuf_iter);
  MS_EXCEPTION_IF_NULL(membuf);
  membuf->status_ = kUnused;
  if (membuf_iter != membuf_ptr_list_.end() - 1) {
    auto next_iter = membuf_iter + 1;
    auto membuf_next = (*next_iter);
    MS_EXCEPTION_IF_NULL(membuf_next);
    if (membuf_next->status_ == kUnused) {
      bool is_merge = IsUsable(current_kernel_, membuf_next->used_kernel_);
      if (is_merge) {
        membuf->size_ += membuf_next->size_;
        (void)membuf_ptr_list_.erase(next_iter);
      }
    }
  }
  if (membuf_iter != membuf_ptr_list_.begin()) {
    auto prev_iter = membuf_iter - 1;
    auto membuf_prev = (*prev_iter);
    MS_EXCEPTION_IF_NULL(membuf_prev);
    if (membuf_prev->status_ == kUnused) {
      bool is_merge = IsUsable(current_kernel_, membuf_prev->used_kernel_);
      if (is_merge) {
        membuf->size_ += membuf_prev->size_;
        membuf->offset_ = membuf_prev->offset_;
        (void)membuf_ptr_list_.erase(prev_iter);
      }
    }
  }
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
  AllocatedSize = membuf_ptr_list_.back()->offset_ + membuf_ptr_list_.back()->size_;
  MS_LOG(INFO) << "MemReuse Allocated Dynamic Size: " << AllocatedSize;
  return AllocatedSize;
}

bool BestFitMemReuse::IsRelease() {
  // unable_used_node include the node type that output tensor cannot be released,
  // even if its refcount is equal to zero.
  std::unordered_set<std::string> unable_used_node = {prim::kPrimBatchNorm->name(), prim::kPrimBatchNormGrad->name(),
                                                      prim::kPrimFusedBatchNorm->name(),
                                                      prim::kPrimFusedBatchNormGrad->name()};
  return unable_used_node.find(current_kernel_->kernel_name()) == unable_used_node.end();
}

size_t BestFitMemReuse::GetTensorIndex(int index) const {
  if (index < 0 || IntToSize(index) >= tensor_ptr_list_.size()) {
    MS_LOG(WARNING) << "current cnode: " << current_kernel_->scope_full_name();
    MS_LOG(EXCEPTION) << "invalid tensor index";
  }
  return IntToSize(index);
}

size_t BestFitMemReuse::GetWorkspaceIndex(int index) const {
  if (index < 0 || IntToSize(index) >= wk_tensor_list_.size()) {
    MS_LOG(WARNING) << "current cnode: " << current_kernel_->scope_full_name();
    MS_LOG(EXCEPTION) << "invalid tensor index";
  }
  return IntToSize(index);
}

int BestFitMemReuse::GetRealIndex(size_t index, int flag) const {
  if (flag == kDynamicMem) {
    return SizeToInt(index);
  } else if (flag == kWorkspaceMem) {
    return kWorkspaceIndexFactor * SizeToInt(index + 1);
  } else {
    MS_LOG(EXCEPTION) << "flag " << flag << " is invalid";
  }
}

void BestFitMemReuse::CheckMembufIndx(size_t membuf_index) const {
  if (membuf_index >= membuf_ptr_list_.size()) {
    MS_LOG(WARNING) << "current cnode: " << current_kernel_->scope_full_name();
    MS_LOG(EXCEPTION) << "invalid membuf index: " << membuf_index << ", real size: " << membuf_ptr_list_.size();
  }
}

void BestFitMemReuse::Reuse(const MemReuseUtil *mem_reuse_util_ptr) {
  MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr);
  InitMemReuseInfo(mem_reuse_util_ptr);
  InitKernelDependence();
  KernelDefPtr pre_op = nullptr;
#ifdef MEM_REUSE_DEBUG
  size_t op_num = 0;
#endif
  for (const auto &op_def_ptr : op_ptr_list_) {
    current_kernel_ = op_def_ptr;
    // releas pre_op_def
    if (pre_op != nullptr) {
      ReleasePreNodeWorkspace(pre_op.get());
    }
    MemReuseChecker::GetInstance().IsAddNewMembuf_ = false;
    // process node output tensor
    AssignNodeOutputOffset();
#ifdef MEM_REUSE_DEBUG
    if (MemReuseChecker::GetInstance().IsAddNewMembuf_) {
      MemReuseChecker::GetInstance().SetAddNewMembuInfos(op_def_ptr.get(), membuf_ptr_list_, op_num);
    }
#endif
    // deal with current op'workspace
    AssignNodeWorkspaceOffset();
    pre_op = op_def_ptr;
    // update node input tensor refcount, and membuf list status
    UpdateNodeInputAndMembuf();
    // check node output tensor which refcount is equal to zero
    if (IsRelease()) {
      ReleaseNodeUnusedOutput();
    }
#ifdef MEM_REUSE_DEBUG
    MemReuseChecker::GetInstance().SetMembuInfos(op_def_ptr.get(), membuf_ptr_list_);
    ++op_num;
#endif
  }
#ifdef MEM_REUSE_DEBUG
  MemReuseChecker::GetInstance().ExportMembufInfoIR();
  MemReuseChecker::GetInstance().ExportAddNewMmebufIR();
  MemReuseChecker::GetInstance().set_kernel_front_map(kernel_front_map_);
  MemReuseChecker::GetInstance().ExportKernelDependence();
#endif
}
}  // namespace memreuse
}  // namespace mindspore
