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
#ifndef MINDSPORE_CORE_UTILS_NUMA_INTERFACE_H_
#define MINDSPORE_CORE_UTILS_NUMA_INTERFACE_H_

#include <memory>
#include <vector>

#include "include/api/status.h"
#include "mindapi/base/macros.h"

namespace mindspore {
// Now we separate the link from mindspore binary with numa,
// and we use dlopen("libnuma") instead. This function will
// return a handle which you can do NumaBind and ReleaseLibrary.
MS_CORE_API std::shared_ptr<void> GetNumaAdapterHandle();

// Totally this function will do:
// 1. Get function pointer of numa api
// 2. Do numa_bind
MS_CORE_API Status NumaBind(void *handle, const int32_t &rank_id);

MS_CORE_API Status LoadNumaCpuInfo(void *handle, const int32_t rank_id, std::vector<int> *numa_cpus);
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_NUMA_INTERFACE_H_
