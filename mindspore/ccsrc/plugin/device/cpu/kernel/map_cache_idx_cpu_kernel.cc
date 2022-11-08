/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/map_cache_idx_cpu_kernel.h"
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
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

void CheckMissCount(size_t miss_count, int count_size, float total_count, float hit_count) {
  if (miss_count != 0) {
    MS_LOG(INFO) << "Miss count: " << miss_count;
  }
  if (count_size != 0) {
    MS_LOG(INFO) << "Avg search count: " << total_count / count_size;
    MS_LOG(INFO) << "Cache hit rate: " << hit_count / count_size;
  }
}

bool MapCacheIdxCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapCacheIdxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapCacheIdxOutputsNum, kernel_name_);

  outputs_ = outputs;
  is_need_retrieve_output_shape_ = true;
  outputs_size_ = outputs.size();
  for (size_t i = 0; i < outputs_size_; i++) {
    dtypes_.push_back(outputs[i]->GetDtype());
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MapCacheIdxCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_UNKNOWN_OUT_SHAPE && ret != KRET_OK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', resize failed, ret: " << ret;
    return ret;
  }

  auto hashmap_shape = inputs[kIndex0]->GetShapeVector();
  auto emb_idx_shape = inputs[kIndex1]->GetShapeVector();
  batch_size_ = SizeOf(emb_idx_shape);

  hashmap_length_ = LongToSize(hashmap_shape[0]);
  if (hashmap_length_ == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the first dimension of 'HashMap' must be greater than 0, but got "
                  << hashmap_length_;
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}

template <typename T>
bool MapCacheIdxCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &workspace,
                                           const std::vector<kernel::AddressPtr> &outputs) {
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
  miss_count_ = 0;
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
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', hashmap is full, search cache idx failed, please set a larger vocab_cache_size!";
      }
      count += 1;
    }
    total_count += SizeToFloat(count);
    if (hashmap[tmp_entry].IsEmpty()) {
      (void)miss_idx.emplace_back(i);
      output_miss_emb_idx[miss_count_] = key;
      output_cache_idx[i] = -1;
      miss_count_++;
    } else {
      hit_count += 1;
      output_cache_idx[i] = hashmap[tmp_entry].value_;
      hashmap[tmp_entry].step_ = step_[0];
    }
  }
  CheckMissCount(miss_count_, count_size, total_count, hit_count);
  float total_insert_count = 0;
  float total_delete_count = 0;
  // swap hash map
  for (size_t i = 0; i < miss_count_; ++i) {
    T emb_idx = output_miss_emb_idx[i];
    T entry = HashFunc(emb_idx, hashmap_length_);
    size_t tag_count = 1;
    while (!hashmap[entry].IsEmpty()) {
      entry = (entry + 1) % static_cast<T>(hashmap_length_);
      if (tag_count > hashmap_length_) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', hashmap is full, insert new key failed, please set a larger vocab_cache_size!";
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
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', hashmap is full, delete old key failed, please set a larger vocab_cache_size!";
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
  if (miss_count_ != 0) {
    MS_LOG(INFO) << "Insert count: " << total_insert_count / miss_count_;
    MS_LOG(INFO) << "Delete count: " << total_delete_count / miss_count_;
  }
  step_[0] += 1;
  for (size_t i = 0; i < miss_count_; ++i) {
    output_cache_idx[miss_idx[i]] = output_swap_cache_idx[i];
  }

  return true;
}

void MapCacheIdxCpuKernelMod::SyncData() {
  ShapeVector out_shape = {SizeToLong(miss_count_)};
  for (size_t i = 1; i < outputs_size_; i++) {
    outputs_[i]->SetShapeVector(out_shape);
  }
}

std::vector<std::pair<KernelAttr, MapCacheIdxCpuKernelMod::MapCacheIdxFunc>> MapCacheIdxCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &MapCacheIdxCpuKernelMod::LaunchKernel<int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &MapCacheIdxCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &MapCacheIdxCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &MapCacheIdxCpuKernelMod::LaunchKernel<int>}};

std::vector<KernelAttr> MapCacheIdxCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MapCacheIdxFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MapCacheIdx, MapCacheIdxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
