/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "utils/cache_embedding_hashmap_struct.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMapCacheIdxInputsNum = 5;
constexpr size_t kMapCacheIdxOutputsNum = 4;
}  // namespace

template <typename T>
int Compress(HashmapEntry<T> *entry_p, const size_t &length, T entry) {
  T i = (entry + 1) % static_cast<T>(length);
  T off = 1;
  int compress_count = 0;
  for (; !entry_p[i].IsEmpty(); i = (i + 1) % static_cast<T>(length), off++) {
    if (entry_p[i].tag_ > off) {
      entry_p[entry].key_ = entry_p[i].key_;
      entry_p[entry].value_ = entry_p[i].value_;
      entry_p[entry].step_ = entry_p[i].step_;
      entry_p[entry].tag_ = entry_p[i].tag_ - off;
      entry_p[i].SetEmpty();
      off = 0;
      entry = i;
    }
    compress_count++;
  }
  return compress_count;
}

void UpdateShape(size_t miss_count, const CNodePtr &node) {
  std::vector<size_t> out_shape;
  (void)out_shape.emplace_back(miss_count);
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  std::vector<TypeId> dtypes(output_num);
  for (size_t i = 0; i < output_num; i++) {
    dtypes[i] = AnfAlgo::GetOutputDeviceDataType(node, i);
  }
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, {AnfAlgo::GetOutputInferShape(node, 0), out_shape, out_shape, out_shape},
                                      node.get());
}

void MapCacheIdxCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  node_wpt_ = kernel_node;
  auto hashmap_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (hashmap_shape.size() != 2) {
    MS_LOG(EXCEPTION) << "Dimension of HashMap must be 2, (n, 4)";
  }
  hashmap_length_ = hashmap_shape[0];
  if (hashmap_length_ == 0) {
    MS_LOG(EXCEPTION) << "Value of hashmap_length_ must > 0!";
  }
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

bool MapCacheIdxCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapCacheIdxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapCacheIdxOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Only support int32, int64";
  }
  return true;
}

template <typename T>
void MapCacheIdxCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  auto node = node_wpt_.lock();
  auto emb_idx_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
  batch_size_ = 1;
  for (size_t i = 0; i < emb_idx_shape.size(); ++i) {
    batch_size_ *= emb_idx_shape[i];
  }
  HashmapEntry<T> *hashmap = reinterpret_cast<HashmapEntry<T> *>(inputs[0]->addr);
  auto input_indices = reinterpret_cast<T *>(inputs[1]->addr);
  T *step_ = reinterpret_cast<T *>(inputs[2]->addr);
  T emb_max_num = *reinterpret_cast<T *>(inputs[3]->addr);
  T offset = *reinterpret_cast<T *>(inputs[4]->addr);
  auto output_cache_idx = reinterpret_cast<T *>(outputs[0]->addr);
  auto output_old_emb_idx = reinterpret_cast<T *>(outputs[1]->addr);
  auto output_miss_emb_idx = reinterpret_cast<T *>(outputs[2]->addr);
  auto output_swap_cache_idx = reinterpret_cast<T *>(outputs[3]->addr);
  std::vector<T> miss_idx;
  size_t miss_count = 0;
  float total_count = 0;
  int count_size = 0;
  float hit_count = 0;
  // search_cache_idx
  for (size_t i = 0; i < batch_size_; ++i) {
    T key = input_indices[i] - offset;
    if (key >= emb_max_num || key < 0) {
      output_cache_idx[i] = -1;
      continue;
    }
    T tmp_entry = HashFunc(key, hashmap_length_);
    size_t count = 1;
    count_size += 1;
    while ((!hashmap[tmp_entry].IsEmpty() && !hashmap[tmp_entry].IsKey(key))) {
      tmp_entry = (tmp_entry + 1) % static_cast<T>(hashmap_length_);
      if (count > hashmap_length_) {
        MS_LOG(EXCEPTION) << "Hashmap is full, search cache idx failed, please set a larger vocab_cache_size!";
      }
      count += 1;
    }
    total_count += SizeToFloat(count);
    if (hashmap[tmp_entry].IsEmpty()) {
      (void)miss_idx.emplace_back(i);
      output_miss_emb_idx[miss_count] = key;
      output_cache_idx[i] = -1;
      miss_count++;
    } else {
      hit_count += 1;
      output_cache_idx[i] = hashmap[tmp_entry].value_;
      hashmap[tmp_entry].step_ = step_[0];
    }
  }
  if (miss_count != 0) {
    MS_LOG(INFO) << "Miss count: " << miss_count;
  }
  if (count_size != 0) {
    MS_LOG(INFO) << "Avg search count: " << total_count / count_size;
    MS_LOG(INFO) << "Cache hit rate: " << hit_count / count_size;
  }
  float total_insert_count = 0;
  float total_delete_count = 0;
  // swap hash map
  for (size_t i = 0; i < miss_count; ++i) {
    T emb_idx = output_miss_emb_idx[i];
    T entry = HashFunc(emb_idx, hashmap_length_);
    size_t tag_count = 1;
    while (!hashmap[entry].IsEmpty()) {
      entry = (entry + 1) % static_cast<T>(hashmap_length_);
      if (tag_count > hashmap_length_) {
        MS_LOG(EXCEPTION) << "Hashmap is full, insert new key failed, please set a larger vocab_cache_size!";
      }
      tag_count++;
    }
    hashmap[entry].key_ = emb_idx;
    hashmap[entry].step_ = step_[0];
    hashmap[entry].tag_ = static_cast<T>(tag_count);
    T tmp_entry = (entry + 1) % static_cast<T>(hashmap_length_);
    size_t delete_count = 1;
    while (hashmap[tmp_entry].IsEmpty() || hashmap[tmp_entry].IsUsing(step_[0])) {
      tmp_entry = (tmp_entry + 1) % static_cast<T>(hashmap_length_);
      if (delete_count > hashmap_length_) {
        MS_LOG(EXCEPTION) << "Hashmap is full, delete old key failed, please set a larger vocab_cache_size!";
      }
      delete_count++;
    }
    output_swap_cache_idx[i] = hashmap[tmp_entry].value_;
    output_old_emb_idx[i] = hashmap[tmp_entry].key_;
    hashmap[entry].value_ = output_swap_cache_idx[i];
    hashmap[tmp_entry].SetEmpty();
    int compress_count = Compress(hashmap, hashmap_length_, tmp_entry);
    total_delete_count += IntToFloat(compress_count + SizeToInt(delete_count));
    total_insert_count += SizeToFloat(tag_count);
  }
  if (miss_count != 0) {
    MS_LOG(INFO) << "Insert count: " << total_insert_count / miss_count;
    MS_LOG(INFO) << "Delete count: " << total_delete_count / miss_count;
  }
  step_[0] += 1;
  for (size_t i = 0; i < miss_count; ++i) {
    output_cache_idx[miss_idx[i]] = output_swap_cache_idx[i];
  }
  UpdateShape(miss_count, node);
}
}  // namespace kernel
}  // namespace mindspore
