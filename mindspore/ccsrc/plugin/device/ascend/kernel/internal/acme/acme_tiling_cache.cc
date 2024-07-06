/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/acme/acme_tiling_cache.h"

#include "transform/acl_ir/op_api_cache.h"

namespace mindspore {
namespace kernel {
TilingCacheItemPtr AcmeTilingCache::Bind(uint64_t key) {
  auto iter = cache_.find(key);
  if (iter != cache_.end()) {
    iter->second->ref_count_++;
    return iter->second;
  }
  return nullptr;
}

void AcmeTilingCache::Unbind(const TilingCacheItemPtr &item) {
  if (item != nullptr) {
    item->ref_count_--;
    MS_LOG(DEBUG) << "unbind, addr: " << item->tiling_info_->tiling_addr_ << ", host_addr: " << item->host_addr_
                  << ", ref: " << item->ref_count_;
  }
}

std::vector<TilingCacheItemPtr> AcmeTilingCache::CombOutSuspectedUselessItems() {
  std::vector<TilingCacheItemPtr> erased_items;
  std::vector<uint64_t> keys;
  for (auto &iter : cache_) {
    if (iter.second->ref_count_ <= 0) {
      (void)keys.emplace_back(iter.first);
      (void)erased_items.emplace_back(iter.second);
      MS_LOG(DEBUG) << "Comb out key: " << iter.first << ", addr: " << iter.second->tiling_info_->tiling_addr_
                    << ", host_addr: " << iter.second->host_addr_ << ", ref: " << iter.second->ref_count_;
    }
  }

  for (auto key : keys) {
    cache_.erase(key);
  }

  return erased_items;
}

bool AcmeTilingCache::Insert(uint64_t key, const TilingCacheItemPtr &ti_ptr) {
  if (cache_.size() == kMaxKernelCount) {
    MS_LOG(INFO) << "The kernel is not cached because of the capacity limit. The key is " << key
                 << ", and the tiling_info is " << ti_ptr->tiling_info_;
    return false;
  }

  if (cache_.find(key) != cache_.end()) {
    MS_LOG(EXCEPTION) << "kernel is already in cache, where the key is " << key
                      << ", device_addr: " << ti_ptr->tiling_info_->tiling_addr_
                      << ", host_addr: " << ti_ptr->host_addr_ << ", size: " << ti_ptr->size_;
  }

  cache_[key] = ti_ptr;
  return true;
}

uint64_t AcmeTilingCache::GenerateKey(const std::string &name, const std::vector<KernelTensor *> &inputs) {
  transform::g_hash_offset = 0;
  transform::GatherInfo(name);
  for (auto &input : inputs) {
    auto type = input->type_id();
    if (type == kObjectTypeTensorType) {
      transform::GatherInfo(input);
    } else if (type == kObjectTypeNumber) {
      auto data_type = input->dtype_id();
      switch (data_type) {
        case kNumberTypeBool: {
          auto value = input->GetValueWithCheck<bool>();
          transform::GatherInfo(value);
          break;
        }
        case kNumberTypeInt32: {
          auto value = input->GetValueWithCheck<int32_t>();
          transform::GatherInfo(value);
          break;
        }
        case kNumberTypeInt64: {
          auto value = input->GetValueWithCheck<int64_t>();
          transform::GatherInfo(value);
          break;
        }
        case kNumberTypeFloat32: {
          auto value = input->GetValueWithCheck<float>();
          transform::GatherInfo(value);
          break;
        }
        case kNumberTypeFloat64: {
          auto value = input->GetValueWithCheck<double>();
          transform::GatherInfo(value);
          break;
        }
        default:
          MS_LOG(INTERNAL_EXCEPTION) << "Unsupported dtype " << data_type << ", kernel: " << name;
      }
    } else if (type == kObjectTypeTuple || type == kObjectTypeList) {
      auto data_type = input->dtype_id();
      switch (data_type) {
        case kNumberTypeInt32: {
          auto value = input->GetValueWithCheck<std::vector<int32_t>>();
          transform::GatherInfo(value);
          break;
        }
        case kNumberTypeInt64: {
          auto value = input->GetValueWithCheck<std::vector<int64_t>>();
          transform::GatherInfo(value);
          break;
        }
        default:
          MS_LOG(INTERNAL_EXCEPTION) << "Unsupported dtype " << data_type << ", kernel: " << name;
      }
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Unsupported input type " << type << ", kernel: " << name;
    }
  }

  auto hash_id = transform::calc_hash_id();
  return hash_id;
}
}  // namespace kernel
}  // namespace mindspore
