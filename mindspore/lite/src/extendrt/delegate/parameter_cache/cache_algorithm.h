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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_PARAMETER_CACHE_CACHE_ALGORITHM_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_PARAMETER_CACHE_CACHE_ALGORITHM_H_

#include <vector>
#include "include/api/status.h"

namespace mindspore {
namespace cache {
struct CacheNoe {
  CacheNoe(int _index, int _frequency, int _value) : key(_index), frequency(_frequency), value(_value) {}
  int key;  // host input index
  int frequency;
  int value;  // cache index
};

class CacheAlgorithm {
 public:
  virtual ~CacheAlgorithm() {}
  virtual int Get(int key) = 0;
  virtual void Put(int key, int value) = 0;
  virtual Status Init(size_t cache_size, int min_host_index, int max_host_index) = 0;
  virtual Status CheckCacheHit(const int *batch_ids, const size_t batch_ids_len, int *cache_index,
                               std::vector<int> *need_swap_indies, std::vector<int> *need_swap_indies_cache_index) = 0;
};
}  // namespace cache
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_PARAMETER_CACHE_CACHE_ALGORITHM_H_
