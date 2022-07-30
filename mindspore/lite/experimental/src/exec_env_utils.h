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

#ifndef MINDSPORE_LITE_EXPERIMENTAL_SRC_EXEC_ENV_UTILS_H_
#define MINDSPORE_LITE_EXPERIMENTAL_SRC_EXEC_ENV_UTILS_H_

#include "thread/threadpool.h"
#include "src/litert/inner_allocator.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"

namespace mindspore::lite::experimental {
#ifdef __cplusplus
extern "C" {
#endif
void *DefaultAllocatorMalloc(void *allocator, size_t sz);
void DefaultAllocatorFree(void *allocator, void *ptr);
int DefaultThreadPoolParallelLunch(void *threadPool, void *task, void *param, int taskNr);
#ifdef __cplusplus
}
#endif
}  // namespace mindspore::lite::experimental

#endif  // MINDSPORE_LITE_EXPERIMENTAL_SRC_EXEC_ENV_UTILS_H_
