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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_THREAD_MICRO_CORE_AFFINITY_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_THREAD_MICRO_CORE_AFFINITY_H_

#ifdef __ANDROID__
#define BIND_CORE
#include <sched.h>
#include <pthread.h>
#endif
#include "wrapper/thread/micro_thread_pool.h"

#define MAX_PATH_SIZE (256)
#define MAX_CPU_ID (9)

#define RET_TP_OK (0)
#define RET_TP_ERROR (1)
#define RET_TP_SYSTEM_ERROR (-1)

#define LOG_INFO(content, args...) \
  { printf("[INFO] %s|%d|%s: " #content "\r\n", __FILE__, __LINE__, __func__, ##args); }
#define LOG_ERROR(content, args...) \
  { printf("[ERROR] %s|%d|%s: " #content "\r\n", __FILE__, __LINE__, __func__, ##args); }

enum BindMode {
  Power_NoBind = 0,  // free schedule
  Power_Higher = 1,
  Power_Middle = 2,
};

typedef struct {
  int core_id;
  int max_freq;
} CpuInfo;

int BindThreads(enum BindMode bind_mode, ThreadPool *g_pool);
int BindThreadToCore(int task_id, ThreadPool *g_pool);
int GetCpuCoreNum(void);

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_THREAD_MICRO_CORE_AFFINITY_H_
