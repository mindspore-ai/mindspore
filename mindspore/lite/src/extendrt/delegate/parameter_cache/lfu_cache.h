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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_PARAMETER_CACHE_LFU_CACHE_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_PARAMETER_CACHE_LFU_CACHE_H_

#include <map>
#include <unordered_map>
#include <list>
#include <vector>
#include "include/api/status.h"
#include "src/extendrt/delegate/parameter_cache/cache_algorithm.h"
namespace mindspore {
namespace cache {
class LFUCacheAlgorithm : public CacheAlgorithm {
 public:
  LFUCacheAlgorithm() {}
  ~LFUCacheAlgorithm() override;

  int Get(int key) override;
  void Put(int key, int value) override;
  Status Init(size_t cache_size, int min_host_index, int max_host_index) override;
  Status CheckCacheHit(const int *batch_ids, const size_t batch_ids_len, int *cache_index,
                       std::vector<int> *need_swap_indies, std::vector<int> *need_swap_indies_cache_index) override;

 private:
  CacheNoe *GetNode(int key);
  void GetHitNodesAndSwapIndex(const int *batch_ids, const size_t batch_ids_len, int *cache_index,
                               std::unordered_map<int, CacheNoe *> *hit_index_nodes,
                               std::unordered_map<int, std::vector<int>> *need_swap_map);
  std::list<CacheNoe *> GetSwapNodes(const std::unordered_map<int, std::vector<int>> &need_swap_map);

  std::unordered_map<int, std::list<CacheNoe *>::iterator> key_table_;
  std::map<int, std::list<CacheNoe *>> frequency_table_;
  size_t cache_size_{0};

  int min_host_index_{0};
  int max_host_index_{1};
};
}  // namespace cache
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_PARAMETER_CACHE_LFU_CACHE_H_
