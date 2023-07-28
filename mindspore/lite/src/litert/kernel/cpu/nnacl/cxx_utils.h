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

#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_NNACL_CXX_UTILS_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_NNACL_CXX_UTILS_H_

#include <stddef.h>
#include <stdint.h>

namespace mindspore::nnacl {
void *DefaultAllocatorMalloc(void *allocator, size_t sz);
void DefaultAllocatorFree(void *allocator, void *ptr);
int DefaultThreadPoolParallelLunch(void *threadPool, void *task, void *param, int taskNr);
void *DefaultGetSharingPackData(void *manager, const void *tensor_data, const size_t size, bool *is_packed);
void DefaultFreeSharingPackData(void *manager, void *tensor_data);
int DefaultUpdateThreadNumPass(int32_t kernel_type, int64_t per_unit_load_num, int64_t per_unit_store_num,
                               int64_t unit_num, int thread_num);
}  // namespace mindspore::nnacl
#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_NNACL_CXX_UTILS_H_
