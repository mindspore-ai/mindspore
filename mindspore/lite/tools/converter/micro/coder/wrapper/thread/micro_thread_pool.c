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

#include "wrapper/thread/micro_thread_pool.h"
#include <stdlib.h>
#include <time.h>
#include "wrapper/thread/micro_core_affinity.h"

ThreadPool *g_pool;
const int kSpinCountMaxValue = 300000;

void *work_routine(void *args) {
  int spin_count = 0;
  while (1) {
    if (g_pool->task.valid) {
      volatile Task *p_task = &g_pool->task;
      int expected_index = p_task->started;
      int finish = 0;
      while (expected_index < p_task->task_num) {
        if (atomic_compare_exchange_strong(&p_task->started, &expected_index, expected_index + 1)) {
          p_task->status |= p_task->func(p_task->content, expected_index, 0, 0);
          expected_index = p_task->started;
          finish++;
        }
      }
      p_task->valid = 0;
      p_task->finished += finish;
      spin_count = 0;
    } else if (spin_count++ > g_pool->max_spin_count) {
      pthread_mutex_lock(&g_pool->queue_lock);
      if (!g_pool->shutdown) {
        pthread_cond_wait(&g_pool->queue_ready, &g_pool->queue_lock);
      }
      pthread_mutex_unlock(&g_pool->queue_lock);
    } else {
      sched_yield();
    }
    if (g_pool->shutdown) {
      pthread_exit(NULL);
    }
  }
}

void SetSpinCountMinValue(void) {
  if (g_pool != NULL) {
    g_pool->max_spin_count = 1;
  }
}

void SetSpinCountMaxValue(void) {
  if (g_pool != NULL) {
    g_pool->max_spin_count = kSpinCountMaxValue;
  }
}

int CreateThreadPool(int thread_num) {
  ClearThreadPool();
  if (thread_num <= 0) {
    return RET_TP_SYSTEM_ERROR;
  }
  int core_num = GetCpuCoreNum();
  if (core_num != 0) {
    thread_num = thread_num > core_num ? core_num : thread_num;
  }

  g_pool = (ThreadPool *)malloc(sizeof(ThreadPool));
  if (g_pool == NULL) {
    return RET_TP_SYSTEM_ERROR;
  }
  if (thread_num > 1) {
    g_pool->thread_id = (pthread_t *)malloc(sizeof(pthread_t) * (thread_num - 1));
    if (g_pool->thread_id == NULL) {
      return RET_TP_SYSTEM_ERROR;
    }
  }
  g_pool->max_thread_num = thread_num;
  g_pool->shutdown = 0;
  g_pool->task.status = g_pool->task.finished = g_pool->task.task_num = 0;
  g_pool->task.valid = 0;
  SetSpinCountMinValue();
  if (pthread_mutex_init(&(g_pool->queue_lock), NULL) != 0) {
    return RET_TP_SYSTEM_ERROR;
  }
  if (pthread_cond_init(&(g_pool->queue_ready), NULL) != 0) {
    return RET_TP_SYSTEM_ERROR;
  }
  if (thread_num > 1) {
    for (int i = 0; i < (thread_num - 1); i++) {
      if (pthread_create(&g_pool->thread_id[i], NULL, work_routine, NULL)) {
        return RET_TP_SYSTEM_ERROR;
      }
    }
  }
  return 0;
}

int SetCoreAffinity(int bind_mode) {
  if (g_pool == NULL) {
    return RET_TP_ERROR;
  }
  return BindThreads(bind_mode, g_pool);
}

int GetCurrentThreadNum() {
  if (g_pool == NULL) {
    LOG_ERROR("thread pool is NULL")
    return 0;
  }
  return g_pool->max_thread_num;
}

int ParallelLaunch(int (*func)(void *, int, float, float), void *content, int task_num) {
  if (g_pool == NULL) {
    LOG_ERROR("thread pool is NULL")
    return RET_TP_ERROR;
  }
  if (task_num == 0) {
    return 0;
  }

  if (task_num == 1) {
    int ret = func(content, 0, 0, 1);
    return ret;
  }
  volatile Task *p_task = &g_pool->task;
  atomic_store(&p_task->valid, 0);
  p_task->func = func;
  p_task->content = content;
  p_task->task_num = task_num;
  p_task->finished = 1;
  p_task->started = 1;
  atomic_store(&p_task->valid, 1);

  pthread_mutex_lock(&g_pool->queue_lock);
  pthread_cond_broadcast(&g_pool->queue_ready);
  pthread_mutex_unlock(&g_pool->queue_lock);

  p_task->status |= func(content, 0, 0, 0);

  int expected_index = p_task->started;
  while (expected_index < task_num) {
    if (atomic_compare_exchange_strong(&p_task->started, &expected_index, expected_index + 1)) {
      p_task->status |= func(content, expected_index, 0, 0);
      (void)++p_task->finished;
      expected_index = p_task->started;
    }
  }
  p_task->valid = 0;

  while (p_task->finished != task_num) {
    sched_yield();
  }
  if (p_task->status != 0) {
    return RET_TP_ERROR;
  }
  return 0;
}

void ClearThreadPool() {
  if (g_pool == NULL) {
    return;
  }
  if (g_pool->shutdown) {
    return;
  }
  g_pool->shutdown = 1;
  pthread_mutex_lock(&g_pool->queue_lock);
  pthread_cond_broadcast(&g_pool->queue_ready);
  pthread_mutex_unlock(&g_pool->queue_lock);
  if (g_pool->max_thread_num > 1) {
    for (int i = 0; i < (g_pool->max_thread_num - 1); i++) {
      pthread_join(g_pool->thread_id[i], NULL);
    }
    free(g_pool->thread_id);
  }
  pthread_mutex_destroy(&g_pool->queue_lock);
  pthread_cond_destroy(&g_pool->queue_ready);
  free(g_pool);
  g_pool = NULL;
}
