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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_H_

#include <stdbool.h>
#include "include/thread_pool_config.h"

/**
 * create thread pool and init
 * @param thread_num
 * @param mode
 */
int ConfigThreadPool(int context_id, int thread_num, CpuBindMode mode);

/**
 *
 * @param session_index, support multi session
 * @param job
 * @param content
 * @param task_num
 */
int ParallelLaunch(int context_id, int (*job)(void *, int), void *content, int task_num);

/**
 * bind each thread to specified cpu core
 * @param is_bind
 * @param mode
 */
int BindThreads(int context_id, bool is_bind, CpuBindMode mode);

/**
 * activate the thread pool
 * @param context_id
 */
void ActivateThreadPool(int context_id);

/**
 * deactivate the thread pool
 * @param context_id
 */
void DeactivateThreadPool(int context_id);

/**
 *
 * @return current thread num
 */
int GetCurrentThreadNum(int context_id);

/**
 * destroy thread pool, and release resource
 */
void DestroyThreadPool(int context_id);

#endif  // MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_H_
