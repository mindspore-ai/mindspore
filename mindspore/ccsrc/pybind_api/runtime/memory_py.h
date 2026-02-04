/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PYBIND_API_HAL_MEMORY_PY_H
#define MINDSPORE_CCSRC_PYBIND_API_HAL_MEMORY_PY_H

#include <string>
#include <memory>
#include <unordered_map>

#include "include/runtime/memory/mem_pool/mem_dynamic_allocator.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;
namespace mindspore {
namespace hal {
py::dict MemoryStats(const std::string &device_target);
void ResetMaxMemoryReserved(const std::string &device_target);
void ResetMaxMemoryAllocated(const std::string &device_target);
size_t EmptyCache(const std::string &device_target);
void MemoryReplay(const std::string &file_path);
}  // namespace hal
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_HAL_MEMORY_PY_H
