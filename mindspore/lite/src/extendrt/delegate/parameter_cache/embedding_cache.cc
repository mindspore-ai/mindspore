/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "src/extendrt/delegate/parameter_cache/embedding_cache.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "src/extendrt/delegate/parameter_cache/gpu/gpu_cache_mem.h"
#include "src/extendrt/delegate/parameter_cache/lfu_cache.h"
#include "src/extendrt/delegate/parameter_cache/factory_mgr_base.h"
#include "core/utils/convert_utils_base.h"

namespace {
constexpr size_t kEmbeddingTensorShapeSize = 2;
}
namespace mindspore {
namespace cache {
void LookUpTableTask(size_t indices_lens, size_t first_dim_size, const char *input_addr, const int *indices_addr,
                     char *output_addr, size_t embedding_len, int min_host_index) {
  for (size_t i = 0; i < indices_lens; ++i) {
    int index = indices_addr[i] - min_host_index;
    if (index >= 0 && index < static_cast<int>(first_dim_size)) {
      size_t pos = index * embedding_len;
      std::memcpy(output_addr, input_addr + pos, embedding_len);
    } else {
      memset(output_addr, 0, embedding_len);
    }
    output_addr += embedding_len;
  }
}

EmbeddingCache::~EmbeddingCache() {
  if (hash_swap_value_device_addr_ != nullptr) {
    device_cache_->FreeMemory(hash_swap_value_device_addr_);
    hash_swap_value_device_addr_ = nullptr;
  }
  if (hash_swap_value_addr_ != nullptr) {
    free(hash_swap_value_addr_);
    hash_swap_value_addr_ = nullptr;
  }
  if (hash_swap_index_addr_ != nullptr) {
    device_cache_->FreeMemory(hash_swap_index_addr_);
    hash_swap_index_addr_ = nullptr;
  }
}

Status EmbeddingCache::Init(mindspore::MSTensor host_cache_tensor, mindspore::MSTensor device_tensor) {
  MS_ASSERT(device_tensor.Shape().size() == kEmbeddingTensorShapeSize);
  MS_ASSERT(host_cache_tensor.Shape().size() == kEmbeddingTensorShapeSize);
  MS_ASSERT(device_tensor.DataType() == host_cache_tensor.DataType());
  MS_ASSERT(host_cache_tensor.Data() != nullptr);

  if (device_tensor.Shape()[1] != host_cache_tensor.Shape()[1]) {
    MS_LOG(ERROR) << device_tensor.Name() << " embedding_size is invalid, device size is " << device_tensor.Shape()[1]
                  << ", host size is " << host_cache_tensor.Shape()[1];
    return kLiteError;
  }
  if (SizeToInt(host_cache_size_) != host_cache_tensor.Shape()[0]) {
    MS_LOG(ERROR) << device_tensor.Name() << " host_cache_size is invalid, host_cache_size"
                  << host_cache_tensor.Shape()[0] << ", index begin:" << min_host_index_
                  << ", index end:" << max_host_index_ << "rank_group_size_ num:" << rank_group_size_
                  << ", rank id:" << rank_id_ << ", vocab_size_:" << vocab_size_;
    return kLiteError;
  }

  data_type_ = device_tensor.DataType();
  switch (data_type_) {
    case DataType::kNumberTypeFloat32:
      sizeof_data_type_ = sizeof(float);
      break;
    default:
      MS_LOG(ERROR) << device_tensor.Name() << " unsupported data type " << static_cast<int>(data_type_);
      return kLiteError;
  }
  host_addr_ = host_cache_tensor.MutableData();
  embedding_size_ = device_tensor.Shape()[1];
  device_start_index_ = device_cache_size_ * rank_id_;
  // host cache tensor is device tensor
  if (device_tensor.Shape()[0] == host_cache_tensor.Shape()[0]) {
    device_start_index_ = min_host_index_;
  }
  return kSuccess;
}

Status EmbeddingCache::MallocCacheMemory() {
  auto hash_swap_value_size = embedding_size_ * batch_elements_ * sizeof_data_type_;
  hash_swap_value_device_addr_ = device_cache_->MallocMemory(hash_swap_value_size);
  if (hash_swap_value_device_addr_ == nullptr) {
    MS_LOG(ERROR) << "malloc hash_swap_value_device failed, malloc size " << hash_swap_value_size;
    return kLiteMemoryFailed;
  }

  hash_swap_value_addr_ = malloc(hash_swap_value_size);
  if (hash_swap_value_addr_ == nullptr) {
    MS_LOG(ERROR) << "malloc hash_swap_value failed, malloc size " << hash_swap_value_size;
    return kLiteMemoryFailed;
  }

  // data type of index
  hash_swap_index_addr_ = static_cast<int *>(device_cache_->MallocMemory(batch_elements_ * sizeof(int)));
  if (hash_swap_index_addr_ == nullptr) {
    MS_LOG(ERROR) << "malloc hash_swap_index failed, malloc size " << batch_elements_ * sizeof(int);
    return kLiteMemoryFailed;
  }
  return kSuccess;
}

Status EmbeddingCache::Init(uint32_t device_id, const void *context, mindspore::MSTensor host_cache_tensor,
                            mindspore::MSTensor device_tensor) {
  auto ret = Init(host_cache_tensor, device_tensor);
  if (ret != kSuccess) {
    return ret;
  }
  cache_ = lite::FactoryManagerBase<std::string, cache::CacheAlgorithm>::Instance().GetProduct("lfu");
  if (cache_ == nullptr) {
    MS_LOG(ERROR) << "malloc LFUCacheAlgorithm failed";
    return kLiteMemoryFailed;
  }
  ret = cache_->Init(device_cache_size_, min_host_index_, max_host_index_);
  if (ret != kSuccess) {
    return kLiteError;
  }

  device_cache_ = lite::FactoryManagerBase<std::string, cache::CacheMemBase>::Instance().GetProduct("gpu");
  if (device_cache_ == nullptr) {
    MS_LOG(ERROR) << "get cache failed";
    return kLiteMemoryFailed;
  }
  if (!device_cache_->InitDevice(device_id, context)) {
    MS_LOG(ERROR) << "init device failed";
    return kLiteError;
  }
  ret = MallocCacheMemory();
  if (ret != kSuccess) {
    return ret;
  }

  MS_LOG(INFO) << "init succ,  rank_group_size_ num:" << rank_group_size_ << ", rank id:" << rank_id_
               << ", vocab_size_:" << vocab_size_ << ", host_cache_size_:" << host_cache_size_
               << ", device_cache_size_:" << device_cache_size_ << ", embedding_size_:" << embedding_size_
               << ", batch_elements_:" << batch_elements_ << ", index begin:" << min_host_index_
               << ", index end:" << max_host_index_;
  return kSuccess;
}

Status EmbeddingCache::SetHostCacheAddr(void *addr, size_t size) {
  if (sizeof_data_type_ * host_cache_size_ * embedding_size_ != size) {
    return kLiteParamInvalid;
  }
  host_addr_ = addr;

  // copy part of host mem to device
  auto ret =
    device_cache_->CopyHostMemToDevice(device_addr_, addr, sizeof_data_type_ * device_cache_size_ * embedding_size_);
  if (!ret) {
    MS_LOG(ERROR) << "CopyHostMemToDevice failed, copy size "
                  << sizeof_data_type_ * device_cache_size_ * embedding_size_;
    return kLiteMemoryFailed;
  }

  // init cache
  auto index_num = device_cache_size_;
  for (size_t i = 0; i < index_num; i++) {
    cache_->Put(min_host_index_ + i, i);
  }

  return kSuccess;
}

Status EmbeddingCache::SetDeviceCacheAddr(void *device_mem_addr, size_t size) {
  if (sizeof_data_type_ * device_cache_size_ * embedding_size_ != size) {
    return kLiteParamInvalid;
  }

  device_addr_ = device_mem_addr;
  SetHostCacheAddr(host_addr_, sizeof_data_type_ * host_cache_size_ * embedding_size_);

  return kSuccess;
}

Status EmbeddingCache::CheckCacheHit(const int *batch_ids, const size_t batch_ids_len, int *cache_index) {
  std::vector<int> need_swap_indies;
  std::vector<int> need_swap_indies_cache_index;
  auto ret =
    cache_->CheckCacheHit(batch_ids, batch_ids_len, cache_index, &need_swap_indies, &need_swap_indies_cache_index);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "CheckCacheHit failed";
    return ret;
  }
  auto swap_indices_size = need_swap_indies.size();
  if (swap_indices_size > 0) {
    LookUpTableTask(swap_indices_size, host_cache_size_, static_cast<char *>(host_addr_), need_swap_indies.data(),
                    static_cast<char *>(hash_swap_value_addr_), embedding_size_ * sizeof_data_type_, min_host_index_);

    auto device_cache_ret = device_cache_->CopyHostMemToDevice(hash_swap_value_device_addr_, hash_swap_value_addr_,
                                                               swap_indices_size * embedding_size_ * sizeof_data_type_);
    if (!device_cache_ret) {
      MS_LOG(ERROR) << "copy swap value to device failed";
      return kLiteMemoryFailed;
    }

    device_cache_ret = device_cache_->CopyHostMemToDevice(hash_swap_index_addr_, need_swap_indies_cache_index.data(),
                                                          swap_indices_size * sizeof(int));
    if (!device_cache_ret) {
      MS_LOG(ERROR) << "copy swap indies to device failed";
      return kLiteMemoryFailed;
    }

    device_cache_ret = device_cache_->HashSwapIn(device_addr_, hash_swap_value_device_addr_, hash_swap_index_addr_,
                                                 device_cache_size_, embedding_size_, swap_indices_size);
    if (!device_cache_ret) {
      MS_LOG(ERROR) << "HashSwapIn failed";
      return kLiteMemoryFailed;
    }
  }

  return kSuccess;
}
}  // namespace cache
}  // namespace mindspore
