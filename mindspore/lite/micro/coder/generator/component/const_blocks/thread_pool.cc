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

#include "coder/generator/component/const_blocks/thread_pool.h"

namespace mindspore::lite::micro {

const char *thread_header = R"RAW(
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
#ifdef __cplusplus
extern "C" {
#endif

#define MAX_TASK_NUM (2)

/// \brief BindMode defined for holding bind cpu strategy argument.
typedef enum {
  NO_BIND_MODE = 0, /**< no bind */
  HIGHER_MODE = 1,  /**< bind higher cpu first */
  MID_MODE = 2      /**< bind middle cpu first */
} BindMode;

struct ThreadPool;

struct ThreadPool *CreateThreadPool(int thread_num, int mode);

/**
 *
 * @param session_index, support multi session
 * @param job
 * @param content
 * @param task_num
 */
int ParallelLaunch(struct ThreadPool *thread_pool, int (*job)(void *, int), void *content, int task_num);

/**
 * bind each thread to specified cpu core
 * @param is_bind
 * @param mode
 */
int BindThreads(struct ThreadPool *thread_pool, bool is_bind, int mode);

/**
 * activate the thread pool
 * @param thread_pool_id
 */
void ActivateThreadPool(struct ThreadPool *thread_pool);

/**
 * deactivate the thread pool
 * @param thread_pool_id
 */
void DeactivateThreadPool(struct ThreadPool *thread_pool);

/**
 *
 * @return current thread num
 */
int GetCurrentThreadNum(struct ThreadPool *thread_pool);

/**
 * destroy thread pool, and release resource
 */
void DestroyThreadPool(struct ThreadPool *thread_pool);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_SRC_RUNTIME_THREAD_POOL_H_
)RAW";

}  // namespace mindspore::lite::micro
