/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/mem_manager/ascend_pluggable_mem_allocator.h"
#include <functional>
#include "include/runtime/pipeline/pipeline.h"

namespace mindspore {
namespace device {
namespace ascend {
void EnablePluggableAllocator(std::function<MallocFuncType> alloc_fn, std::function<FreeFuncType> free_fn) {
  runtime::Pipeline::Get().WaitForward();
  AscendMemoryPool::GetInstance().EnablePluggableAllocator(alloc_fn, free_fn);
}

void DisablePluggableAllocator() {
  runtime::Pipeline::Get().WaitForward();
  AscendMemoryPool::GetInstance().DisablePluggableAllocator();
}

void RegPluggableAllocator(py::module *m) {
  (void)m->def(
    "_enable_pluggable_allocator",
    [](uint64_t malloc_ptr, uint64_t free_ptr) {
      std::function<MallocFuncType> malloc_fn = reinterpret_cast<MallocFuncType *>(malloc_ptr);
      std::function<FreeFuncType> free_fn = reinterpret_cast<FreeFuncType *>(free_ptr);
      EnablePluggableAllocator(malloc_fn, free_fn);
    },
    "Enable pluggable allocator.");
  (void)m->def("_disable_pluggable_allocator", &DisablePluggableAllocator, "Disable pluggable allocator.");
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
