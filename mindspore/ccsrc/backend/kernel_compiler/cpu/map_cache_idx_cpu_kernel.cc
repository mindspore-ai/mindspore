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

#include "backend/kernel_compiler/cpu/map_cache_idx_cpu_kernel.h"
#include <string>
#include <memory>
#include <vector>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {

template <typename T>
struct HashmapEntry {
  T key;
  T value;
  T step;
  T tag;

  bool IsEmpty() {
    if (this->tag == NULLTAG)
      return true;
    else
      return false;
  }

  bool IsUsing(const T &train_step) {
    if (this->step >= (train_step - 1))
      return true;
    else
      return false;
  }

  bool IsKey(const T &emb_idx) {
    if (this->key == emb_idx)
      return true;
    else
      return false;
  }

  void SetEmpty() { this->tag = NULLTAG; }
};

template <typename T>
T HashFunc(const T &key, const size_t &m) {
  return (T)(((0.6180339 * key) - floor(0.6180339 * key)) * m);
}

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

void MapCacheIdxCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  auto hashmap_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto emb_idx_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);

  if (hashmap_shape.size() != 2) {
    MS_LOG(EXCEPTION) << "Dimension of HashMap must be 2, (n, 4)";
  }

  for (size_t i = 0; i < emb_idx_shape.size(); ++i) {
    batch_size_ *= emb_idx_shape[i];
  }

  hashmap_length_ = hashmap_shape[0];
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

bool MapCacheIdxCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
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
void MapCacheIdxCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  HashmapEntry<T> *hashmap = reinterpret_cast<HashmapEntry<T> *>(inputs[0]->addr);
  auto input_indices = reinterpret_cast<T *>(inputs[1]->addr);
  T *step_ = reinterpret_cast<T *>(inputs[2]->addr);
  T emb_max_num = *reinterpret_cast<T *>(inputs[3]->addr);
  T cache_max_num = *reinterpret_cast<T *>(inputs[4]->addr);
  auto output_cache_idx = reinterpret_cast<T *>(outputs[0]->addr);
  auto output_old_emb_idx = reinterpret_cast<T *>(outputs[1]->addr);
  auto output_miss_emb_idx = reinterpret_cast<T *>(outputs[2]->addr);
  auto output_swap_cache_idx = reinterpret_cast<T *>(outputs[3]->addr);

  std::vector<T> output_miss_idx(batch_size_, -1);

  float total_count = 0;
  int count_size = 0;
  float hit_count = 0;

  // search_cache_idx
  for (size_t i = 0; i < batch_size_; ++i) {
    if (input_indices[i] == emb_max_num) {
      output_miss_idx[i] = -1;
      output_cache_idx[i] = cache_max_num;
      output_miss_emb_idx[i] = -1;
      continue;
    }

    T key = input_indices[i];
    T tmp_entry = HashFunc(key, hashmap_length_);

    int count = 1;
    count_size += 1;
    while ((!hashmap[tmp_entry].IsEmpty() && !hashmap[tmp_entry].IsKey(key))) {
      tmp_entry = (tmp_entry + 1) % hashmap_length_;
      count += 1;
    }

    total_count += count;
    if (hashmap[tmp_entry].IsEmpty()) {
      output_miss_idx[i] = i;
      output_miss_emb_idx[i] = key;
      output_cache_idx[i] = -1;
    } else {
      hit_count += 1;
      output_miss_idx[i] = -1;
      output_cache_idx[i] = hashmap[tmp_entry].value;
      hashmap[tmp_entry].step = step_[0];
      output_miss_emb_idx[i] = -1;
    }
  }
  MS_LOG(INFO) << "avg search count: " << total_count / count_size;
  MS_LOG(INFO) << "cache hit rate: " << hit_count / count_size;

  // swap hash map
  for (size_t i = 0; i < batch_size_; ++i) {
    if (output_miss_emb_idx[i] < 0) {
      output_swap_cache_idx[i] = -1;
      output_old_emb_idx[i] = -1;
    } else {
      T emb_idx = output_miss_emb_idx[i];
      T entry = HashFunc(emb_idx, hashmap_length_);
      T tag_count = 1;
      while (!hashmap[entry].IsEmpty()) {
        entry = (entry + 1) % hashmap_length_;
        tag_count++;
      }

      hashmap[entry].key = emb_idx;
      hashmap[entry].step = step_[0];
      hashmap[entry].tag = tag_count;

      T tmp_entry = (entry + 1) % hashmap_length_;

      while (hashmap[tmp_entry].IsEmpty() || hashmap[tmp_entry].IsUsing(step_[0])) {
        tmp_entry = (tmp_entry + 1) % hashmap_length_;
      }

      output_swap_cache_idx[i] = hashmap[tmp_entry].value;
      output_old_emb_idx[i] = hashmap[tmp_entry].key;
      hashmap[entry].value = output_swap_cache_idx[i];
      hashmap[tmp_entry].SetEmpty();
      Compress(hashmap, hashmap_length_, tmp_entry);
    }
  }

  // update step
  step_[0] += 1;

  // update cache idx
  for (size_t i = 0; i < batch_size_; ++i) {
    if (output_miss_idx[i] < 0 || output_miss_idx[i] >= cache_max_num) {
      continue;
    }
    output_cache_idx[i] = output_swap_cache_idx[i];
  }
}

}  // namespace kernel
}  // namespace mindspore
