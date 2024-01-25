/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PI_JIT_MEMPOOL_H
#define MINDSPORE_PI_JIT_MEMPOOL_H

#include <string>
#include <utility>
#include <vector>

namespace mindspore {
namespace pijit {
template <typename T>
class MemPool {
 public:
  MemPool(const char *file, int line, const char *type_name)
      : decs_(std::string(file) + ":" + std::to_string(line) + " mempool of " + type_name) {}
  ~MemPool() { Clear(__FILE__, __LINE__); }

  template <typename U, typename... Args>
  U *New(Args &&... args) {
    U *r = new U(std::forward<Args>(args)...);
    pool_.emplace_back(r);
    return r;
  }

  void Clear(const char *f, int l) {
    for (auto &i : pool_) {
      delete i;
    }
    pool_.clear();
  }

 private:
  std::vector<T *> pool_;
  std::string decs_;
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_MEMPOOL_H
