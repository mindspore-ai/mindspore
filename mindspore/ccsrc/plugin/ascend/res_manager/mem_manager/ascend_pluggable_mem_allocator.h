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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_MEM_MANAGER_ASCEND_PLUGGABLE_MEM_ALLOCATOR_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_MEM_MANAGER_ASCEND_PLUGGABLE_MEM_ALLOCATOR_H_

#include "plugin/ascend/res_manager/visible.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_memory_pool.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;
namespace mindspore {
namespace device {
namespace ascend {
ASCEND_RES_MANAGER_EXPORT void EnablePluggableAllocator(std::function<MallocFuncType> alloc_fn,
                                                        std::function<FreeFuncType> free_fn);
ASCEND_RES_MANAGER_EXPORT void DisablePluggableAllocator();

ASCEND_RES_MANAGER_EXPORT void RegPluggableAllocator(py::module *m);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif
