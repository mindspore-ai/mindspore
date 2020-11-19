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

#include "backend/kernel_compiler/cpu/cache_swap_hashmap_cpu_kernel.h"
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void Compress(HashmapEntry<T> *entry_p, const size_t &length, T entry) {
  T i = (entry + 1) % length, off = 1;
  for (; !entry_p[i].IsEmpty(); i = (i + 1) % length, off++) {
    if (entry_p[i].tag > off) {
      entry_p[entry].key = entry_p[i].key;
      entry_p[entry].value = entry_p[i].value;
      entry_p[entry].step = entry_p[i].step;
      entry_p[entry].tag = entry_p[i].tag - off;
      entry_p[i].SetEmpty();
      off = 0;
      entry = i;
    }
  }
}

void CacheSwapHashmapCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  auto hashmap_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto emb_idx_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);

  if (hashmap_shape.size() != 2) {
    MS_LOG(EXCEPTION) << "Dimension of HashMap must be 2, (n, 4)";
  }

  for (size_t i = 0; i < emb_idx_shape.size(); ++i) {
    batch_size_ *= emb_idx_shape[i];
  }

  hashmap_length_ = hashmap_shape[0];
  if (hashmap_length_ <= 0) {
    MS_LOG(EXCEPTION) << "Hashmap length must > 0";
  }
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

bool CacheSwapHashmapCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> & /*workspace*/,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "Only support int32, int64";
    return false;
  }
  return true;
}

template <typename T>
void CacheSwapHashmapCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  HashmapEntry<T> *hashmap = reinterpret_cast<HashmapEntry<T> *>(inputs[0]->addr);
  auto miss_emb_idx = reinterpret_cast<T *>(inputs[1]->addr);
  step_ = *reinterpret_cast<T *>(inputs[2]->addr);
  auto swap_cache_idx = reinterpret_cast<T *>(outputs[0]->addr);
  auto old_emb_idx = reinterpret_cast<T *>(outputs[1]->addr);

  for (size_t i = 0; i < batch_size_; ++i) {
    if (miss_emb_idx[i] < 0) {
      swap_cache_idx[i] = -1;
      old_emb_idx[i] = -1;
    } else {
      T emb_idx = miss_emb_idx[i];
      T entry = HashFunc(emb_idx, hashmap_length_);
      T tag_count = 1;
      while (!hashmap[entry].IsEmpty()) {
        entry = (entry + 1) % hashmap_length_;
        tag_count++;
      }

      hashmap[entry].key = emb_idx;
      hashmap[entry].step = step_;
      hashmap[entry].tag = tag_count;

      T tmp_entry = (entry + 1) % hashmap_length_;

      while (hashmap[tmp_entry].IsEmpty() || hashmap[tmp_entry].IsUsing(step_)) {
        tmp_entry = (tmp_entry + 1) % hashmap_length_;
      }

      swap_cache_idx[i] = hashmap[tmp_entry].value;
      old_emb_idx[i] = hashmap[tmp_entry].key;
      hashmap[entry].value = swap_cache_idx[i];
      hashmap[tmp_entry].SetEmpty();
      Compress(hashmap, hashmap_length_, tmp_entry);
    }
  }
}
}  // namespace kernel
}  // namespace mindspore
