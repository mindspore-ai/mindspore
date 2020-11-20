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

#include "backend/kernel_compiler/cpu/search_cache_idx_cpu_kernel.h"
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void SearchCacheIdxCPUKernel::InitKernel(const CNodePtr &kernel_node) {
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

bool SearchCacheIdxCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
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
void SearchCacheIdxCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  HashmapEntry<T> *hashmap = reinterpret_cast<HashmapEntry<T> *>(inputs[0]->addr);
  auto input_indices = reinterpret_cast<T *>(inputs[1]->addr);
  step_ = *reinterpret_cast<T *>(inputs[2]->addr);
  emb_max_num = *reinterpret_cast<T *>(inputs[3]->addr);
  cache_max_num = *reinterpret_cast<T *>(inputs[4]->addr);
  auto output_cache_idx = reinterpret_cast<T *>(outputs[0]->addr);
  auto output_miss_idx = reinterpret_cast<T *>(outputs[1]->addr);
  auto output_miss_emb_idx = reinterpret_cast<T *>(outputs[2]->addr);

  float total_count = 0;
  int count_size = 0;
  float hit_count = 0;
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
      hashmap[tmp_entry].step = step_;
      output_miss_emb_idx[i] = -1;
    }
  }
  if (count_size != 0) {
    MS_LOG(INFO) << "avg search count: " << total_count / count_size;
    MS_LOG(INFO) << "cache hit rate: " << hit_count / count_size;
  }
}
}  // namespace kernel
}  // namespace mindspore
