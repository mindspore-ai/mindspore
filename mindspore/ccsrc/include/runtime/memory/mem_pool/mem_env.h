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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_MEMORY_MEM_POOL_MEM_ENV_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_MEMORY_MEM_POOL_MEM_ENV_H_

#include <memory>
#include <utility>
#include <string>
#include "include/backend/visible.h"

namespace mindspore {
namespace memory {
namespace mem_pool {
// Memory dev config.
const char kAllocConf[] = "MS_ALLOC_CONF";
const char kAllocAclAllocator[] = "acl_allocator";
const char kAllocSomasWholeBlock[] = "somas_whole_block";
const char kAllocEnableVmm[] = "enable_vmm";
const char kAllocVmmAlignSize[] = "vmm_align_size";
const char kAllocMemoryRecycle[] = "memory_recycle";
const char kAllocMemoryTracker[] = "memory_tracker";
const char kAllocSimpleTracker[] = "simple_tracker";
const char kAllocMemoryTrackerPath[] = "memory_tracker_path";
const char kAllocDefragMemoryStepFreq[] = "defrag_memory_step_freq";
const char kAllocMemoryPool[] = "older_pool";
const char kAllocEnableSmallPool[] = "enable_small_pool";
const char kAllocEnableMemHuge1G[] = "enable_mem_huge_1g";

BACKEND_EXPORT std::string GetAllocConfigValue(const std::string &alloc_config);
BACKEND_EXPORT bool IsEnableAllocConfig(const std::string &alloc_config);
BACKEND_EXPORT bool IsDisableAllocConfig(const std::string &alloc_config);
}  // namespace mem_pool
}  // namespace memory
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIME_MEMORY_MEM_POOL_MEM_ENV_H_
