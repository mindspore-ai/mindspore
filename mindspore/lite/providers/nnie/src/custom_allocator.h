/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_PROVIDERS_NNIE_SRC_CUSTOM_ALLOCATOR_H_
#define MINDSPORE_LITE_PROVIDERS_NNIE_SRC_CUSTOM_ALLOCATOR_H_

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include "include/api/allocator.h"
#include "include/hi_type.h"

namespace mindspore {
namespace nnie {
class CustomAllocator : public Allocator {
 public:
  CustomAllocator() {}
  ~CustomAllocator() override{};
  void *Malloc(size_t size) override { return nullptr; }
  void Free(void *ptr) override {}
  int RefCount(void *ptr) override { return 1; }
  int SetRefCount(void *ptr, int ref_count) override { return ref_count; }
  int DecRefCount(void *ptr, int ref_count) override { return 1; }
  int IncRefCount(void *ptr, int ref_count) override { return 1; }
};
}  // namespace nnie
}  // namespace mindspore

#endif  // MINDSPORE_LITE_PROVIDERS_NNIE_SRC_CUSTOM_ALLOCATOR_H_
