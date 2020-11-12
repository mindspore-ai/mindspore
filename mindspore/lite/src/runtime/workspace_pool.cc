/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/workspace_pool.h"
#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif
#include <algorithm>
#include "src/common/log_adapter.h"

namespace mindspore {
namespace predict {
static constexpr size_t kWorkspacePageSize = 4096;
static constexpr int kTempAllocaAlignment = 64;
WorkspacePool *WorkspacePool::GetInstance() {
  static WorkspacePool instance;
  return &instance;
}

void *WorkspacePool::AllocWorkSpaceMem(size_t size) {
  size_t nbytes = (size + (kWorkspacePageSize - 1)) / kWorkspacePageSize * kWorkspacePageSize;
  if (nbytes == 0) {
    nbytes = kWorkspacePageSize;
  }
  std::pair<size_t, void *> alloc;
  // fist alloc
  if (freeList.empty()) {
    alloc.first = nbytes;
#ifdef __APPLE__
    int err = posix_memalign(&alloc.second, kTempAllocaAlignment, nbytes);
    if (err != 0) {
      MS_LOGE("posix_memalign failed, error code:%d", err);
      return alloc.second;
    }
#else
#ifdef _WIN32
    alloc.second = _aligned_malloc(nbytes, kTempAllocaAlignment);
#else
    alloc.second = memalign(kTempAllocaAlignment, nbytes);
#endif
#endif
  } else if (freeList.size() == 1) {  // one element
    alloc = *(freeList.begin());
    freeList.erase(freeList.begin());
    if (alloc.first < nbytes) {
      free(alloc.second);
      alloc.first = nbytes;
#ifdef __APPLE__
      int err = posix_memalign(&alloc.second, kTempAllocaAlignment, nbytes);
      if (err != 0) {
        MS_LOGE("posix_memalign failed, error code:%d", err);
        return alloc.second;
      }
#else
#ifdef _WIN32
      alloc.second = _aligned_malloc(nbytes, kTempAllocaAlignment);
#else
      alloc.second = memalign(kTempAllocaAlignment, nbytes);
#endif
#endif
    }
  } else {
    if ((*(freeList.begin())).first >= nbytes) {
      auto iter = freeList.begin();
      for (; iter != freeList.end(); ++iter) {
        if ((*iter).first < size) {
          alloc = *(--iter);
          freeList.erase(iter);
          break;
        }
      }
      if (iter == freeList.end()) {
        alloc = *(freeList.rbegin());
        freeList.erase(--freeList.end());
      }
    } else {
      alloc = *(freeList.begin());
      freeList.erase(freeList.begin());
      free(alloc.second);
      alloc.first = nbytes;
#ifdef __APPLE__
      int err = posix_memalign(&alloc.second, kTempAllocaAlignment, nbytes);
      if (err != 0) {
        MS_LOGE("posix_memalign failed, error code:%d", err);
        return alloc.second;
      }
#else
#ifdef _WIN32
      alloc.second = _aligned_malloc(nbytes, kTempAllocaAlignment);
#else
      alloc.second = memalign(kTempAllocaAlignment, nbytes);
#endif
#endif
    }
  }
  allocList.emplace_back(alloc);
  return alloc.second != nullptr ? alloc.second : nullptr;
}

void WorkspacePool::FreeWorkSpaceMem(const void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  std::pair<size_t, void *> alloc;
  if (allocList.empty()) {
    MS_LOG(ERROR) << "no mem have been alloc";
    return;
  } else if (allocList.back().second == ptr) {
    alloc = allocList.back();
    allocList.pop_back();
  } else {
    auto iter = allocList.begin();
    for (; iter != allocList.end(); ++iter) {
      if ((*iter).second == ptr) {
        alloc = *iter;
        allocList.erase(iter);
        break;
      }
    }
    if (iter == allocList.end()) {
      MS_LOG(ERROR) << "no value ptr have been alloc";
      return;
    }
  }
  freeList.insert(alloc);
}

WorkspacePool::~WorkspacePool() {
  for (auto &a : allocList) {
    free(a.second);
  }
  allocList.clear();
  for (auto &f : freeList) {
    free(f.second);
  }
  freeList.clear();
}
}  // namespace predict
}  // namespace mindspore
