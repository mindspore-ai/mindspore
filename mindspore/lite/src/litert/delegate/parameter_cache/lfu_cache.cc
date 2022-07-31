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
#include <vector>
#include <string>
#include "src/common/log_adapter.h"
#include "src/litert/delegate/parameter_cache/lfu_cache.h"
#include "src/litert/delegate/parameter_cache/factory_mgr_base.h"
namespace mindspore {
namespace cache {
RET_COMMON_PRODUCT_REGISTRAR(std::string, cache::CacheAlgorithm, cache::LFUCacheAlgorithm, "lfu", LFUCacheAlgorithm);

LFUCacheAlgorithm::~LFUCacheAlgorithm() {
  for (auto iter : key_table_) {
    delete *(iter.second);
  }
  key_table_.clear();
  frequency_table_.clear();
}

Status LFUCacheAlgorithm::Init(size_t cache_size, int min_host_index, int max_host_index) {
  if (cache_size <= 0 || min_host_index < 0 || max_host_index <= 0) {
    return kLiteParamInvalid;
  }
  cache_size_ = cache_size;
  min_host_index_ = min_host_index;
  max_host_index_ = max_host_index;
  return kSuccess;
}

CacheNoe *LFUCacheAlgorithm::GetNode(int key) {
  auto key_table_iter = key_table_.find(key);
  if (key_table_iter == key_table_.end()) {
    return nullptr;
  }
  auto node_iter = key_table_iter->second;
  auto node = *node_iter;

  auto node_list_iter = frequency_table_.find(key);
  if (node_list_iter == frequency_table_.end()) {
    return nullptr;
  }
  auto &node_list = node_list_iter->second;
  node_list.erase(node_iter);

  if (node_list.empty()) {
    frequency_table_.erase(node_list_iter);
  }

  node->frequency += 1;
  frequency_table_[node->frequency].emplace_front(node);
  key_table_[key] = frequency_table_[node->frequency].begin();
  return node;
}

int LFUCacheAlgorithm::Get(int key) {
  auto node = GetNode(key);
  if (node != nullptr) {
    return node->value;
  }
  return -1;
}

void LFUCacheAlgorithm::Put(int key, int value) {
  auto node = GetNode(key);
  if (node != nullptr) {
    node->value = value;
    return;
  }

  if (cache_size_ == 0) {
    return;
  }

  CacheNoe *add_node = nullptr;
  if (key_table_.size() == cache_size_) {
    add_node = frequency_table_.begin()->second.back();
    key_table_.erase(add_node->key);
    frequency_table_.begin()->second.pop_back();
    if (frequency_table_.begin()->second.size() == 0) {
      frequency_table_.erase(frequency_table_.begin()->first);
    }
    add_node->value = value;
    add_node->key = key;
    add_node->frequency = 1;
  } else {
    add_node = new CacheNoe(key, 1, value);
    if (add_node == nullptr) {
      return;
    }
  }

  frequency_table_[1].emplace_front(add_node);
  key_table_[key] = frequency_table_[1].begin();
}

void LFUCacheAlgorithm::GetHitNodesAndSwapIndex(const int *batch_ids, const size_t batch_ids_len, int *cache_index,
                                                std::unordered_map<int, CacheNoe *> *hit_index_nodes,
                                                std::unordered_map<int, std::vector<int>> *need_swap_map) {
  // 找到没有命中和命中的index
  for (size_t i = 0; i < batch_ids_len; i++) {
    auto key = batch_ids[i];
    if (key < min_host_index_ || key >= max_host_index_) {
      cache_index[i] = -1;
      // out range
      continue;
    }

    auto hit_iter = hit_index_nodes->find(key);
    if (hit_iter != hit_index_nodes->end()) {
      auto node = hit_iter->second;
      node->frequency += 1;
      cache_index[i] = node->value;
      continue;
    }

    auto swap_iter = need_swap_map->find(key);
    if (swap_iter != need_swap_map->end()) {
      swap_iter->second.push_back(i);
      continue;
    }

    auto node_iter_iter = key_table_.find(key);
    if (node_iter_iter == key_table_.end()) {
      (*need_swap_map)[key].push_back(i);
      continue;
    }
    auto node_iter = node_iter_iter->second;
    auto node = *node_iter;

    auto node_list_iter = frequency_table_.find(node->frequency);
    if (node_list_iter == frequency_table_.end()) {
      continue;
    }
    auto &node_list = node_list_iter->second;
    node_list.erase(node_iter);

    if (node_list.empty()) {
      frequency_table_.erase(node_list_iter);
    }
    // hit
    node->frequency += 1;
    cache_index[i] = node->value;
    (*hit_index_nodes)[key] = node;
  }
  return;
}

std::list<CacheNoe *> LFUCacheAlgorithm::GetSwapNodes(const std::unordered_map<int, std::vector<int>> &need_swap_map) {
  std::list<CacheNoe *> need_swap_nodes;
  auto swap_size = need_swap_map.size();

  while (swap_size > 0 && !frequency_table_.empty()) {
    auto node_list_iter = frequency_table_.begin();
    if (node_list_iter->second.size() > swap_size) {
      auto iter = node_list_iter->second.begin();
      std::advance(iter, swap_size);
      need_swap_nodes.splice(need_swap_nodes.end(), node_list_iter->second, node_list_iter->second.begin(), iter);
      swap_size = 0;
    } else {
      swap_size -= node_list_iter->second.size();
      need_swap_nodes.splice(need_swap_nodes.end(), node_list_iter->second);
      frequency_table_.erase(node_list_iter);
    }
  }
  return need_swap_nodes;
}

Status LFUCacheAlgorithm::CheckCacheHit(const int *batch_ids, const size_t batch_ids_len, int *cache_index,
                                        std::vector<int> *need_swap_indies,
                                        std::vector<int> *need_swap_indies_cache_index) {
  if (batch_ids == nullptr) {
    MS_LOG(ERROR) << "batch_ids is nullptr";
    return kLiteNullptr;
  }
  if (cache_index == nullptr) {
    MS_LOG(ERROR) << "cache_index is nullptr";
    return kLiteNullptr;
  }
  std::unordered_map<int, std::vector<int>> need_swap_map;
  std::unordered_map<int, CacheNoe *> hit_index_nodes;
  GetHitNodesAndSwapIndex(batch_ids, batch_ids_len, cache_index, &hit_index_nodes, &need_swap_map);

  // get need_swap_indies.size() least recently used node
  std::list<CacheNoe *> need_swap_nodes = GetSwapNodes(need_swap_map);

  // 更新老节点的值
  {
    if (need_swap_map.size() != need_swap_nodes.size()) {
      MS_LOG(ERROR) << " need_swap_map.size() " << need_swap_map.size() << " != need_swap_nodes.size() "
                    << need_swap_nodes.size();
      return kLiteError;
    }
    need_swap_indies_cache_index->reserve(need_swap_map.size());
    auto need_swap_map_iter = need_swap_map.begin();
    for (auto iter = need_swap_nodes.begin();
         iter != need_swap_nodes.end() && need_swap_map_iter != need_swap_map.end(); iter++, need_swap_map_iter++) {
      auto node = *iter;
      key_table_.erase(node->key);
      node->key = need_swap_map_iter->first;
      node->frequency = 1;
      for (auto index : need_swap_map_iter->second) {
        cache_index[index] = node->value;
      }
      need_swap_indies->push_back(need_swap_map_iter->first);
      need_swap_indies_cache_index->push_back(node->value);
      MS_LOG(INFO) << "device index " << node->value << ",for host index " << need_swap_map_iter->first;
      key_table_[(*iter)->key] = iter;
    }

    auto node_list_iter = frequency_table_.begin();
    if (node_list_iter->second.size() > 0) {
      auto iter = node_list_iter->second.begin();
      if ((*iter)->frequency == 1) {
        node_list_iter->second.splice(node_list_iter->second.begin(), need_swap_nodes);
      } else {
        frequency_table_[1] = need_swap_nodes;
      }
    } else {
      frequency_table_[1] = need_swap_nodes;
    }
  }
  for (auto node_iter : hit_index_nodes) {
    auto node = node_iter.second;
    frequency_table_[node->frequency].emplace_front(node);
    key_table_[node->key] = frequency_table_[node->frequency].begin();
  }
  return kSuccess;
}
}  // namespace cache
}  // namespace mindspore
