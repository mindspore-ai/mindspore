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

#ifndef MINDSPORE_LITE_MICRO_THREAD_POOL_H
#define MINDSPORE_LITE_MICRO_THREAD_POOL_H

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>

typedef struct Task {
  int (*func)(void *, int, float, float);
  void *content;
  atomic_int finished;
  atomic_int status;  // return status, RET_OK
} Task;

typedef struct TaskHandle {
  struct TaskHandle *next;
  int task_id;
  Task *task;
} TaskHandle;

typedef struct ThreadPool {
  int max_thread_num;
  pthread_t *thread_id;
  pthread_cond_t queue_ready;
  pthread_mutex_t queue_lock;
  TaskHandle *tasks;
  int shutdown;
} ThreadPool;

static ThreadPool *g_pool;

int CreateThreadPool(int thread_num);

int SetCoreAffinity(int bind_mode);

int GetCurrentThreadNum();

int ParallelLaunch(int (*func)(void *, int, float, float), void *content, int task_num);

void ClearThreadPool();

#endif  // MINDSPORE_LITE_MICRO_THREAD_POOL_H
