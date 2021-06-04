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
const char thread_header[] = R"RAW(
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

#ifndef MINDSPORE_LITE_MICRO_CODER_THREAD_WRAPPER_H
#define MINDSPORE_LITE_MICRO_CODER_THREAD_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

int CreateThreadPool(int thread_num);

int SetCoreAffinity(int bind_mode);

int GetCurrentThreadNum();

int ParallelLaunch(int (*func)(void *, int, float, float), void *content, int task_num);

void ClearThreadPool();

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_MICRO_CODER_THREAD_WRAPPER_H

)RAW";
}  // namespace mindspore::lite::micro
