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
#include "wrapper/thread/micro_core_affinity.h"

void *work_routine(void *args) {
  while (1) {
    pthread_mutex_lock(&g_pool->queue_lock);
    while (!g_pool->tasks && !g_pool->shutdown) {
      pthread_cond_wait(&g_pool->queue_ready, &g_pool->queue_lock);
    }
    if (g_pool->shutdown) {
      pthread_mutex_unlock(&g_pool->queue_lock);
      pthread_exit(NULL);
    }
    TaskHandle *task_handle = g_pool->tasks;
    g_pool->tasks = task_handle->next;
    pthread_mutex_unlock(&g_pool->queue_lock);
    Task *task = task_handle->task;
    if (task == NULL) {
      continue;
    }
    task->status |= task->func(task->content, task_handle->task_id, 0, 1);
    ++task->finished;
    free(task_handle);
  }
}

int CreateThreadPool(int thread_num) {
  g_pool = (ThreadPool *)malloc(sizeof(ThreadPool));
  if (g_pool == NULL) {
    return RET_TP_SYSTEM_ERROR;
  }
  g_pool->max_thread_num = thread_num;
  g_pool->tasks = NULL;
  g_pool->shutdown = 0;
  g_pool->thread_id = (pthread_t *)malloc(sizeof(pthread_t) * thread_num);
  if (g_pool->thread_id == NULL) {
    return RET_TP_SYSTEM_ERROR;
  }
  if (pthread_mutex_init(&(g_pool->queue_lock), NULL) != 0) {
    return RET_TP_SYSTEM_ERROR;
  }
  if (pthread_cond_init(&(g_pool->queue_ready), NULL) != 0) {
    return RET_TP_SYSTEM_ERROR;
  }
  for (int i = 0; i < thread_num; i++) {
    if (pthread_create(&g_pool->thread_id[i], NULL, work_routine, g_pool) != 0) {
      return RET_TP_SYSTEM_ERROR;
    }
  }
  return 0;
}

int SetCoreAffinity(int bind_mode) {
  if (g_pool == NULL) {
    return RET_TP_ERROR;
  }
  return BindThreads(bind_mode);
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
    return -1;
  }
  if (task_num == 0) {
    return 0;
  }
  if (task_num == 1) {
    int ret = func(content, 0, 0, 1);
    return ret;
  }
  Task task = {func, content, 0, 0};
  // multi Task shared same base task
  for (int i = 0; i < task_num; i++) {
    TaskHandle *task_handle = (TaskHandle *)malloc(sizeof(TaskHandle));
    task_handle->task_id = i;
    task_handle->task = &task;
    task_handle->next = NULL;
    // add task
    pthread_mutex_lock(&g_pool->queue_lock);
    TaskHandle *task_tail = g_pool->tasks;
    if (!task_tail) {
      g_pool->tasks = task_handle;
    } else {
      while (task_tail->next) {
        task_tail = task_tail->next;
      }
      task_tail->next = task_handle;
    }
    pthread_cond_signal(&g_pool->queue_ready);
    pthread_mutex_unlock(&g_pool->queue_lock);
  }
  while (task.finished != task_num) {
    sched_yield();
  }
  if (task.status != 0) {
    return -1;
  }
  return 0;
}

void ClearThreadPool() {
  if (g_pool->shutdown) {
    return;
  }
  g_pool->shutdown = 1;
  pthread_mutex_lock(&g_pool->queue_lock);
  pthread_cond_broadcast(&g_pool->queue_ready);
  pthread_mutex_unlock(&g_pool->queue_lock);

  for (int i = 0; i < g_pool->max_thread_num; i++) {
    pthread_join(g_pool->thread_id[i], NULL);
  }
  free(g_pool->thread_id);
  while (g_pool->tasks) {
    TaskHandle *task = g_pool->tasks;
    g_pool->tasks = g_pool->tasks->next;
    free(task);
  }
  pthread_mutex_destroy(&g_pool->queue_lock);
  pthread_cond_destroy(&g_pool->queue_ready);
  free(g_pool);
}
