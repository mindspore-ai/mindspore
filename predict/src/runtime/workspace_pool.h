/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PREDICT_SRC_RUNTIME_WORKSPACE_POOL_H_
#define PREDICT_SRC_RUNTIME_WORKSPACE_POOL_H_
#include <memory>
#include <vector>
#include <set>
#include <utility>
#include <functional>
#include <mutex>

namespace mindspore {
namespace predict {
class WorkspacePool {
 public:
  WorkspacePool() = default;
  ~WorkspacePool();
  WorkspacePool(const WorkspacePool &) = delete;
  WorkspacePool &operator=(const WorkspacePool &) = delete;
  static WorkspacePool *GetInstance();
  void *AllocWorkSpaceMem(size_t size);
  void FreeWorkSpaceMem(void *ptr);

 private:
  std::vector<std::pair<size_t, void *>> allocList{};
  std::set<std::pair<size_t, void *>, std::greater<std::pair<size_t, void *>>> freeList{};
};
}  // namespace predict
}  // namespace mindspore
#endif  // PREDICT_SRC_RUNTIME_WORKSPACE_POOL_H_
