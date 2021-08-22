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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_NUMA_INTERFACE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_NUMA_INTERFACE_H_

#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
struct bitmask {
  uint64_t size;
  uint64_t *maskp;
};

// Now we separate the link from _c_dataengine with numa,
// and we use dlopen("libnuma") instead. This function will
// return a handle which you can do NumaBind and ReleaseLibrary.
void *GetNumaAdapterHandle();

// Totally this function will do:
// 1. Get function pointer of numa api
// 2. Do numa_bind
Status NumaBind(void *handle, const int32_t &rank_id);

// Release the numa handle for avoid memory leak, we should
// not allow handle is nullptr before we use it.
void ReleaseLibrary(void *handle);
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_NUMA_INTERFACE_H_
