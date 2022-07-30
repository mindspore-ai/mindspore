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
#include "src/litert/delegate/parameter_cache/embedding_cache_manager.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace {
constexpr size_t kGatherInputsSize = 3;
}
namespace mindspore {
namespace cache {
Status EmbeddingCacheManager::Init(const std::string &cache_model_path, size_t vocab_size, size_t device_cache_size) {
  if (cache_model_path.empty() || vocab_size == 0 || device_cache_size >= vocab_size) {
    MS_LOG(INFO) << "no cache model ,  vocab_size " << vocab_size << ",  device_cache_size " << device_cache_size;
    return kSuccess;
  }

  host_cache_model_ = std::make_shared<HostCacheModel>();
  if (host_cache_model_ == nullptr) {
    MS_LOG(ERROR) << "HostCacheModel malloc failed";
    return kLiteMemoryFailed;
  }
  auto ret = host_cache_model_->LoadCache(cache_model_path);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "load cache failed";
    return ret;
  }
  vocab_size_ = vocab_size;
  device_cache_size_ = device_cache_size;

  MS_LOG(INFO) << "cache manager init succ, cache model" << cache_model_path << " ,  vocab_size " << vocab_size
               << ",  device_cache_size " << device_cache_size;
  return ret;
}

Status EmbeddingCacheManager::Init(DelegateModel<schema::Primitive> *model, size_t vocab_size,
                                   size_t device_cache_size) {
  if (model == nullptr || vocab_size == 0 || device_cache_size >= vocab_size) {
    MS_LOG(INFO) << "no cache model ,  vocab_size " << vocab_size << ",  device_cache_size " << device_cache_size;
    return kSuccess;
  }

  host_cache_model_ = std::make_shared<HostCacheModel>();
  if (host_cache_model_ == nullptr) {
    MS_LOG(ERROR) << "HostCacheModel malloc failed";
    return kLiteMemoryFailed;
  }
  auto ret = host_cache_model_->LoadCache(model);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "load cache failed";
    return ret;
  }
  vocab_size_ = vocab_size;
  device_cache_size_ = device_cache_size;

  MS_LOG(INFO) << "cache manager init succ,  vocab_size " << vocab_size << ",  device_cache_size " << device_cache_size;
  return ret;
}

bool EmbeddingCacheManager::CheckIsCacheKernel(kernel::Kernel *kernel) {
  if (host_cache_model_ == nullptr) {
    return false;
  }
  return host_cache_model_->CheckIsCacheKernel(kernel);
}

Status EmbeddingCacheManager::InitCacheKernel(kernel::Kernel *kernel, uint32_t device_id, const void *context) {
  if (host_cache_model_ == nullptr) {
    MS_LOG(ERROR) << "cache model is nullptr, kernel " << kernel->name() << " init cache failed";
    return kLiteError;
  }
  auto host_cache_tensor = host_cache_model_->GetHostCacheTensor(kernel);
  if (host_cache_tensor == nullptr) {
    MS_LOG(ERROR) << kernel->name() << ": invalid cache kernel";
    return kLiteError;
  }

  // only support embedding cache
  if (kernel->type() != schema::PrimitiveType_Gather) {
    MS_LOG(ERROR) << kernel->name() << " is not embedding kernel";
    return kLiteError;
  }
  MS_ASSERT(kernel->inputs().size() == kGatherInputsSize);
  auto device_tensor = kernel->inputs()[0];
  size_t batch_elements = kernel->inputs()[1].ElementNum();
  auto cache =
    std::make_shared<EmbeddingCache>(vocab_size_, device_cache_size_, batch_elements, rank_id_, rank_group_size_);
  if (cache == nullptr) {
    MS_LOG(ERROR) << kernel->name() << ": malloc EmbeddingCache failed";
    return kLiteError;
  }

  auto ret = cache->Init(device_id, context, host_cache_tensor, device_tensor);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << kernel->name() << ": EmbeddingCache init failed";
    return kLiteError;
  }

  caches_[device_tensor.Name()] = cache;
  MS_LOG(INFO) << kernel->name() << " is cache kernel, input tensor " << kernel->inputs()[1].Name() << ", cache tensor "
               << device_tensor.Name();

  return kSuccess;
}

bool EmbeddingCacheManager::IsCacheTensor(mindspore::MSTensor tensor) {
  if (host_cache_model_ == nullptr) {
    return false;
  }
  auto cache = caches_.find(tensor.Name());
  if (cache != caches_.end()) {
    return true;
  }
  return false;
}

std::vector<int64_t> EmbeddingCacheManager::GetCacheShape(mindspore::MSTensor tensor) {
  std::vector<int64_t> shape = tensor.Shape();
  if (shape.size() > 0 && IsCacheTensor(tensor)) {
    shape[0] = device_cache_size_;
  }
  return shape;
}

size_t EmbeddingCacheManager::GetCacheDataSize(mindspore::MSTensor tensor) {
  auto data_size = tensor.DataSize();
  auto &shape = tensor.Shape();
  if (shape.size() > 0 && IsCacheTensor(tensor) && shape[0] > 0) {
    data_size = data_size * device_cache_size_ / shape[0];
  }
  return data_size;
}

Status EmbeddingCacheManager::SetDeviceCacheAddr(const std::string &tensor_name, void *device_mem_addr, size_t size) {
  auto cache_iter = caches_.find(tensor_name);
  if (cache_iter == caches_.end() || cache_iter->second == nullptr) {
    MS_LOG(ERROR) << "not find cache, " << tensor_name;
    return kLiteError;
  }
  auto cache = cache_iter->second;
  return cache->SetDeviceCacheAddr(device_mem_addr, size);
}

// device_addr is model input device addr
int EmbeddingCacheManager::CacheHandle(const std::string &tensor_name, mindspore::MSTensor model_input_tensor,
                                       void *model_input_device_addr) {
  auto cache_iter = caches_.find(tensor_name);
  if (cache_iter == caches_.end()) {
    MS_LOG(ERROR) << "not find cache, " << tensor_name;
    return lite::RET_ERROR;
  }
  auto cache = cache_iter->second;
  hash_indices_.resize(model_input_tensor.ElementNum());
  auto ret = cache->CheckCacheHit(static_cast<int *>(model_input_tensor.MutableData()), hash_indices_.size(),
                                  hash_indices_.data());
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "CheckCacheHit failed, " << model_input_tensor.Name();
    return lite::RET_ERROR;
  }

  for (size_t i = 0; i < hash_indices_.size(); i++) {
    if (hash_indices_[i] != -1) {
      hash_indices_[i] += cache->GetDeviceStartIndex();
    }
  }

  auto cuda_ret = cudaMemcpy(model_input_device_addr, hash_indices_.data(), hash_indices_.size() * sizeof(int),
                             cudaMemcpyHostToDevice);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "copy mem failed, " << model_input_tensor.Name();
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "cache handle succ, " << model_input_tensor.Name() << "," << tensor_name;

  return lite::RET_OK;
}
}  // namespace cache
}  // namespace mindspore
