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

#include "src/runtime/thread_pool.h"
#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <semaphore.h>
#include <string.h>
#include <stdlib.h>

#ifdef __ANDROID__
#define BIND_CORE
#include <unistd.h>
#include <sched.h>
#endif

#ifdef THREAD_POOL_DEBUG
#include <stdio.h>
#define LOG_INFO(content, args...) \
  { printf("[INFO] %s|%d|%s: " #content "\r\n", __FILE__, __LINE__, __func__, ##args); }
#define LOG_ERROR(content, args...) \
  { printf("[ERROR] %s|%d|%s: " #content "\r\n", __FILE__, __LINE__, __func__, ##args); }
#else
#define LOG_INFO(content, args...)
#define LOG_ERROR(content, args...)
#endif

#define RET_TP_OK (0)
#define RET_TP_ERROR (1)
#define RET_TP_SYSTEM_ERROR (-1)

#define MAX_TASK_NUM (2)
#define MAX_THREAD_NUM (8)
#define MAX_THREAD_POOL_NUM (4)
#define DEFAULT_SPIN_COUNT (30000)

typedef struct {
  int (*func)(void *arg, int);
  void *content;
} Task;

typedef struct Thread {
  int thread_pool_id;
  int thread_id;
  struct Thread *next;
  pthread_t pthread;
  Task *task_list[MAX_TASK_NUM];
  atomic_int task_size;
  atomic_int head;
  atomic_int tail;
  atomic_bool activate;
  atomic_bool is_running;
  sem_t sem;
} Thread;

typedef struct {
  Thread *head;
  Thread *tail;
  pthread_mutex_t lock;
  int size;
} ThreadList;

typedef struct ThreadPool {
  ThreadList *thread_list;
  int thread_num;
  BindMode mode;
  atomic_bool is_alive;
} ThreadPool;

static ThreadPool thread_pool_list[MAX_THREAD_POOL_NUM];
static atomic_int thread_pool_refcount[MAX_THREAD_POOL_NUM] = {ATOMIC_VAR_INIT(0)};
static atomic_bool thread_pool_is_created[MAX_THREAD_POOL_NUM] = {ATOMIC_VAR_INIT(false)};

ThreadPool *GetInstance(int thread_pool_id) {
  if (thread_pool_id < 0 || thread_pool_id >= MAX_THREAD_POOL_NUM) {
    LOG_ERROR("invaid context id: %d", thread_pool_id);
    // DestroyThreadPool(thread_pool_id);
    return NULL;
  }
  return &thread_pool_list[thread_pool_id];
}

Thread *GetThread(int thread_pool_id, int thread_id) {
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed, thread_pool_id: %d, thread_id: %d", thread_pool_id, thread_id);
    return NULL;
  }
  ThreadList *thread_list = thread_pool->thread_list;
  if (thread_list == NULL) {
    LOG_ERROR("thead list is null");
    return NULL;
  }
  if (thread_id >= thread_list->size) {
    LOG_ERROR("invalid thread id: %d, thread_pool_id: %d, thread size: %d", thread_id, thread_pool_id,
             thread_list->size);
    return NULL;
  }
  if (thread_id == 0) {
    return thread_list->head;
  }
  Thread *thread = thread_list->head;
  while (thread != NULL) {
    if (thread->thread_id == thread_id) {
      break;
    }
    thread = thread->next;
  }
  return thread;
}

void FreeThread(ThreadList *thread_list, Thread *thread) {
  if (thread_list == NULL) {
    LOG_ERROR("thead list is null");
    return;
  }
  if (thread == NULL) {
    LOG_ERROR("thread is nullptr");
    return;
  }
  // only support sequential release
  thread_list->head = thread->next;
  sem_post(&thread->sem);
  while (thread != NULL && !thread->is_running) {
    sem_destroy(&thread->sem);
    free(thread);
    thread = NULL;
  }
}

#ifdef BIND_CORE
#define MAX_CORE_NUM (16)
static int gCoreNum = 8;
static int gHigNum = 0;
static int gMidNum = 0;
static int cpu_cores[MAX_CORE_NUM];
static bool run_once = true;

#define MAX_CPU_ID (9)
#define MAX_PATH_SIZE (256)
typedef struct {
  int core_id;
  int max_freq;
} CpuInfo;

int GetCpuCoreNum() { return (int)sysconf(_SC_NPROCESSORS_CONF); }

static int ConcatCPUPath(int cpuID, const char *str1, const char *str2, char *str3) {
  if (cpuID > MAX_CPU_ID || str1 == NULL || str2 == NULL) {
    return RET_TP_ERROR;
  }
  memset(str3, 0, strlen(str3));
  char *tmp = str3;
  char id = cpuID + '0';
  memcpy(tmp, str1, strlen(str1));
  tmp += strlen(str1);
  memcpy(tmp, &id, 1);
  tmp += 1;
  memcpy(tmp, str2, strlen(str2));
  return RET_TP_OK;
}

int GetMaxFrequence(int core_id) {
  char path[MAX_PATH_SIZE] = "";
  int ret = ConcatCPUPath(core_id, "/sys/devices/system/cpu/cpufreq/stats/cpu", "/time_in_state", path);
  if (ret != RET_TP_OK) {
    LOG_ERROR("parse cpuid from /sys/devices/system/cpu/cpufreq/stats/cpu/time_in_state failed!");
    return RET_TP_ERROR;
  }
  FILE *fp = fopen(path, "rb");
  if (fp == NULL) {
    ret = ConcatCPUPath(core_id, "/sys/devices/system/cpu/cpufreq/stats/cpu", "/cpufreq/stats/time_in_state", path);
    if (ret != RET_TP_OK) {
      LOG_ERROR("parse cpuid from /sys/devices/system/cpu/cpufreq/stats/cpu/cpufreq/stats/time_instate failed!");
      return RET_TP_ERROR;
    }
    fp = fopen(path, "rb");
    if (fp == NULL) {
      ret = ConcatCPUPath(core_id, "/sys/devices/system/cpu/cpu", "/cpufreq/cpuinfo_max_freq", path);
      if (ret != RET_TP_OK) {
        LOG_ERROR("parse cpuid from /sys/devices/system/cpu/cpufreq/cpuinfo_max_freq failed!");
        return RET_TP_ERROR;
      }
      fp = fopen(path, "rb");
      if (fp == NULL) {
        LOG_ERROR("GetCPUMaxFreq failed, cannot find cpuinfo_max_freq.");
        return RET_TP_ERROR;
      }
      int maxFreq = -1;
      int result __attribute__((unused));
      result = fscanf(fp, "%d", &maxFreq);
      fclose(fp);
      return maxFreq;
    }
  }
  int maxFreq = -1;
  while (feof(fp) == 0) {
    int freq = 0;
    int tmp = fscanf(fp, "%d", &freq);
    if (tmp != 1) {
      break;
    }
    if (freq > maxFreq) {
      maxFreq = freq;
    }
  }
  fclose(fp);
  return maxFreq;
}

int SortCpuProcessor() {
  gCoreNum = GetCpuCoreNum();
  if (gCoreNum <= 0) {
    LOG_ERROR("invalid cpu count");
    return RET_TP_ERROR;
  }
  CpuInfo freq_set[gCoreNum];
  for (int i = 0; i < gCoreNum; ++i) {
    int max_freq = GetMaxFrequence(i);
    freq_set[i].core_id = i;
    freq_set[i].max_freq = max_freq;
  }
  // sort core id by frequency
  for (int i = 0; i < gCoreNum; ++i) {
    for (int j = i + 1; j < gCoreNum; ++j) {
      if (freq_set[i].max_freq <= freq_set[j].max_freq) {
        CpuInfo temp = freq_set[i];
        freq_set[i] = freq_set[j];
        freq_set[j] = temp;
      }
    }
  }
  for (int i = 0; i < gCoreNum; ++i) {
    cpu_cores[i] = freq_set[i].core_id;
    LOG_INFO("sorted_order: %d, frequency: %d", freq_set[i].core_id, freq_set[i].max_freq);
  }
  gHigNum = 0;
  gMidNum = 0;
  int max_freq = freq_set[0].max_freq;
  int min_freq = freq_set[gCoreNum - 1].max_freq;
  int little = 0;
  for (int i = 0; i < gCoreNum; ++i) {
    if (freq_set[i].max_freq == max_freq) {
      gHigNum++;
    }
    if (freq_set[i].max_freq == min_freq) {
      little++;
    }
  }
  gMidNum = gCoreNum - gHigNum - little;
  if (gHigNum == gCoreNum || max_freq == min_freq) {
    // fix MTK800
    gHigNum = 2;
    gMidNum = 2;
    LOG_INFO("core frequency may be wrong.");
  }
  LOG_INFO("gCoreNum: %d, gHigNum: %d, gMidNum: %d, gLitNum: %d", gCoreNum, gHigNum, gMidNum, little);
  return RET_TP_OK;
}

#ifndef CPU_SET
#define CPU_SETSIZE 1024
#define __NCPUBITS (8 * sizeof(unsigned long))
typedef struct {
  unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;
#define CPU_SET(cpu, cpusetp) ((cpusetp)->__bits[(cpu) / __NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))
#define CPU_ZERO(cpusetp) memset((cpusetp), 0, sizeof(cpu_set_t))
#endif  // CPU_SET

int SetAffinity(pthread_t thread_id, cpu_set_t *cpuSet) {
#ifdef __ANDROID__
#if __ANDROID_API__ >= 21
  LOG_INFO("thread: %d, mask: %lu", pthread_gettid_np(thread_id), cpuSet->__bits[0]);
  int ret = sched_setaffinity(pthread_gettid_np(thread_id), sizeof(cpu_set_t), cpuSet);
  if (ret != RET_TP_OK) {
    LOG_ERROR("bind thread %d to cpu failed. ERROR %d", pthread_gettid_np(thread_id), ret);
    return RET_TP_OK;
  }
#endif
#else
#ifdef __APPLE__
  LOG_ERROR("not bind thread to apple's cpu.");
  return RET_TP_ERROR;
#else
  int ret = pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), cpuSet);
  if (ret != RET_TP_OK) {
    LOG_ERROR("set thread: %lu to cpu failed", thread_id);
    return RET_TP_SYSTEM_ERROR;
  }
#endif  // __APPLE__
#endif
  return RET_TP_OK;
}

int BindMasterThread(int thread_pool_id, bool is_bind) {
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return RET_TP_ERROR;
  }
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (is_bind) {
    unsigned int attach_id;
    if (thread_pool->mode == MID_MODE) {
      attach_id = cpu_cores[gHigNum + gMidNum - 1];
    } else {
      attach_id = cpu_cores[0];
    }
    LOG_INFO("mode: %d, attach id: %u", thread_pool->mode, attach_id);
    CPU_SET(attach_id, &mask);
  } else {
    for (int i = 0; i < gHigNum + gMidNum; ++i) {
      CPU_SET(cpu_cores[i], &mask);
    }
  }
  int ret = SetAffinity(pthread_self(), &mask);
  if (ret != RET_TP_OK) {
    LOG_ERROR("set master thread affinity failed");
    return RET_TP_ERROR;
  }
  LOG_INFO("BindMasterThread success.");
  return RET_TP_OK;
}

int BindSalverThreads(int thread_pool_id, bool is_bind) {
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return RET_TP_ERROR;
  }
  cpu_set_t mask;
  if (is_bind && thread_pool->mode != NO_BIND_MODE) {
    unsigned int attach_id;
    for (int i = 0; i < thread_pool->thread_num - 1; ++i) {
      if (thread_pool->mode == MID_MODE) {
        int core_id = gHigNum + gMidNum - i - 2;
        if (core_id >= 0) {
          attach_id = cpu_cores[core_id];
        } else {
          attach_id = cpu_cores[0];
        }
      } else {
        attach_id = cpu_cores[i + 1];
      }
      LOG_INFO("mode: %d, attach id: %u", thread_pool->mode, attach_id);
      CPU_ZERO(&mask);
      CPU_SET(attach_id, &mask);
      Thread *thread = GetThread(thread_pool_id, i);
      if (thread == NULL) {
        LOG_ERROR("get thread failed, thread_pool_id: %d, thread_id: %d", thread_pool_id, i);
        return false;
      }
      int ret = SetAffinity(thread->pthread, &mask);
      if (ret != RET_TP_OK) {
        LOG_ERROR("set thread affinity failed");
        return RET_TP_ERROR;
      }
    }
  } else {
    CPU_ZERO(&mask);
    for (int i = 0; i < gHigNum + gMidNum; ++i) {
      CPU_SET(cpu_cores[i], &mask);
    }
    for (int i = 0; i < thread_pool->thread_num - 1; ++i) {
      Thread *thread = GetThread(thread_pool_id, i);
      if (thread == NULL) {
        LOG_ERROR("get thread failed, thread_pool_id: %d, thread_id: %d", thread_pool_id, i);
        return false;
      }
      int ret = SetAffinity(thread->pthread, &mask);
      if (ret != RET_TP_OK) {
        LOG_ERROR("set thread affinity failed");
        return RET_TP_ERROR;
      }
    }
  }
  LOG_INFO("BindSalverThreads success");
  return RET_TP_OK;
}
#endif

int BindThreads(int thread_pool_id, bool is_bind, int mode) {
#ifdef BIND_CORE
  if (mode == NO_BIND_MODE) {
    return RET_TP_OK;
  }
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return RET_TP_ERROR;
  }
  thread_pool->mode = mode;
  int ret = BindMasterThread(thread_pool_id, is_bind);
  if (ret != RET_TP_OK) {
    LOG_ERROR("bind master thread failed.");
  }
  ret = BindSalverThreads(thread_pool_id, is_bind);
  if (ret != RET_TP_OK) {
    LOG_ERROR("bind salver thread failed.");
  }
  return ret;
#else
  return RET_TP_OK;
#endif
}

bool PushTaskToQueue(int thread_pool_id, int thread_id, Task *task) {
  Thread *thread = GetThread(thread_pool_id, thread_id);
  if (thread == NULL) {
    LOG_ERROR("get thread failed, thread_pool_id: %d, thread_id: %d", thread_pool_id, thread_id);
    return false;
  }
  const int tail_index = atomic_load_explicit(&thread->tail, memory_order_relaxed);
  int next = (tail_index + 1) % MAX_TASK_NUM;
  if (next == atomic_load_explicit(&thread->head, memory_order_acquire)) {
    return false;
  }
  thread->task_list[tail_index] = task;
  atomic_store_explicit(&thread->tail, next, memory_order_release);
  atomic_fetch_add_explicit(&thread->task_size, 1, memory_order_relaxed);
  // atomic_store_explicit(&thread->task_size, thread->task_size + 1, memory_order_relaxed);
  sem_post(&thread->sem);
  return true;
}

bool PopTaskFromQueue(Thread *thread, Task **task) {
  if (thread == NULL) {
    LOG_ERROR("thread is nullptr");
    return false;
  }
  if (thread->task_size == 0) {
    return false;
  }
  const int head_index = atomic_load_explicit(&thread->head, memory_order_relaxed);
  if (head_index == atomic_load_explicit(&thread->tail, memory_order_acquire)) {
    return false;
  }
  *task = thread->task_list[head_index];
  atomic_store_explicit(&thread->head, (head_index + 1) % MAX_TASK_NUM, memory_order_release);
  return true;
}

void WaitAllThread(int thread_pool_id) {
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return;
  }
  bool k_success_flag = false;
  while (!k_success_flag) {
    k_success_flag = true;
    for (int i = 0; i < thread_pool->thread_num - 1; ++i) {
      Thread *thread = GetThread(thread_pool_id, i);
      if (thread == NULL) {
        LOG_ERROR("get thread failed, thread_pool_id: %d, thread_id: %d", thread_pool_id, i);
        return;
      }
      if (thread->task_size != 0) {
        k_success_flag = false;
        break;
      }
    }
  }
}

int DistributeTask(int thread_pool_id, Task *task, int task_num) {
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return RET_TP_ERROR;
  }
  if (task_num > thread_pool->thread_num || task_num <= 1) {
    LOG_ERROR("invalid task num: %d, thread num: %d", task_num, thread_pool->thread_num);
    return RET_TP_ERROR;
  }
  bool k_success_flag = false;
  int size = thread_pool->thread_num < task_num ? thread_pool->thread_num : task_num;
  for (int i = 0; i < size - 1; ++i) {
    do {
      k_success_flag = true;
      if (!PushTaskToQueue(thread_pool_id, i, task)) {
        k_success_flag = false;
      }
    } while (!k_success_flag);
  }
  // master thread
  task->func(task->content, size - 1);
  if (task->func == NULL) {
    LOG_ERROR("task->func is nullptr");
    return RET_TP_ERROR;
  }

  // wait
  WaitAllThread(thread_pool_id);
  return RET_TP_OK;
}

int AddTask(int thread_pool_id, int func(void *, int), void *content, int task_num) {
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return RET_TP_ERROR;
  }
  // if single thread, run master thread
  if (thread_pool->thread_num <= 1 || task_num <= 1) {
    for (int i = 0; i < task_num; ++i) {
      func(content, i);
    }
    return RET_TP_OK;
  }
  Task task;
  task.func = func;
  task.content = content;
  return DistributeTask(thread_pool_id, &task, task_num);
}

int ParallelLaunch(int thread_pool_id, int (*func)(void *, int), void *content, int task_num) {
  return AddTask(thread_pool_id, func, content, task_num);
}

void ThreadRun(Thread *thread) {
  ThreadPool *thread_pool = GetInstance(thread->thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return;
  }
  Task *task = NULL;
  int thread_id = thread->thread_id;
  int spin_count = 0;
  thread->is_running = true;
  while (thread_pool->is_alive) {
    while (thread->activate) {
      if (PopTaskFromQueue(thread, &task)) {
        task->func(task->content, thread_id);
        if (task->func == NULL) {
          LOG_ERROR("task->func is nullptr");
          return;
        }
        atomic_fetch_sub_explicit(&thread->task_size, 1, memory_order_relaxed);
        // atomic_store_explicit(&thread->task_size, thread->task_size - 1, memory_order_relaxed);
        spin_count = 0;
        sem_trywait(&thread->sem);
      } else {
        sched_yield();
        spin_count++;
      }
      if (spin_count == DEFAULT_SPIN_COUNT) {
        break;
      }
    }
    sem_wait(&thread->sem);
  }
  thread->is_running = false;
}

void PushThreadToList(int thread_pool_id, Thread *thread) {
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return;
  }
  ThreadList *thread_list = thread_pool->thread_list;
  if (thread_list == NULL) {
    LOG_ERROR("thread list is null");
    DestroyThreadPool(thread_pool_id);
    return;
  }
  pthread_mutex_lock(&thread_list->lock);
  if (thread_list->size == 0) {
    thread_list->head = thread;
    thread_list->tail = thread;
  } else {
    thread_list->tail->next = thread;
    thread_list->tail = thread;
  }
  thread_list->size++;
  pthread_mutex_unlock(&thread_list->lock);
}

int CreateNewThread(int thread_pool_id, int thread_id) {
  LOG_INFO("thread_pool_id: %d, create thread: %d", thread_pool_id, thread_id);
  Thread *thread = (Thread *)malloc(sizeof(Thread));
  if (thread == NULL) {
    LOG_ERROR("create thread failed");
    DestroyThreadPool(thread_pool_id);
    return RET_TP_ERROR;
  }
  thread->thread_pool_id = thread_pool_id;
  thread->thread_id = thread_id;
  thread->head = ATOMIC_VAR_INIT(0);
  thread->tail = ATOMIC_VAR_INIT(0);
  thread->task_size = ATOMIC_VAR_INIT(0);
  thread->activate = ATOMIC_VAR_INIT(true);
  thread->is_running = ATOMIC_VAR_INIT(false);
  thread->next = NULL;
  sem_init(&thread->sem, 0, 0);
  PushThreadToList(thread_pool_id, thread);
  pthread_create(&thread->pthread, NULL, (void *)ThreadRun, thread);
  pthread_detach(thread->pthread);
  return RET_TP_OK;
}

int ReConfigThreadPool(int thread_pool_id, int thread_num, int mode) {
  LOG_INFO("reconfig thread pool, thread_pool_id: %d, thread_num: %d, mode: %d", thread_pool_id, thread_num, mode);
  if (thread_num <= 0 || thread_num > MAX_THREAD_NUM) {
    LOG_ERROR("invalid thread num: %d", thread_num);
    return RET_TP_ERROR;
  }
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return RET_TP_ERROR;
  }
  if (thread_num <= thread_pool->thread_num) {
    LOG_INFO("no need to add thread");
    return RET_TP_OK;
  }
  int curr_thread_num = thread_pool->thread_num;
  thread_pool->thread_num = thread_num > MAX_THREAD_NUM ? MAX_THREAD_NUM : thread_num;
  thread_pool->mode = mode;
  if (thread_pool->thread_list == NULL) {
    thread_pool->thread_list = (ThreadList *)malloc(sizeof(ThreadList));
    if (thread_pool->thread_list == NULL) {
      LOG_ERROR("create thread list failed");
      DestroyThreadPool(thread_pool_id);
      return RET_TP_ERROR;
    }
    thread_pool->thread_list->head = NULL;
    thread_pool->thread_list->tail = NULL;
    thread_pool->thread_list->size = 0;
    pthread_mutex_init(&thread_pool->thread_list->lock, NULL);
  }
  int add_thread_num = thread_pool->thread_num - curr_thread_num;
  for (int i = curr_thread_num - 1, j = 0; j < add_thread_num; ++i, ++j) {
    int ret = CreateNewThread(thread_pool_id, i);
    if (ret != RET_TP_OK) {
      LOG_ERROR("create new thread failed");
      return RET_TP_ERROR;
    }
  }
  return BindThreads(thread_pool_id, true, mode);
}

int CreateThreadPool(int thread_pool_id, int thread_num, int mode) {
  LOG_INFO("create thread pool, thread_pool_id: %d, thread_num: %d, mode: %d", thread_pool_id, thread_num, mode);
  if (thread_num <= 0 || thread_num > MAX_THREAD_NUM) {
    LOG_ERROR("invalid thread num: %d", thread_num);
    return RET_TP_ERROR;
  }
#ifdef BIND_CORE
  if (run_once) {
    SortCpuProcessor();
    run_once = false;
  }
#endif
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return RET_TP_ERROR;
  }
  thread_pool->thread_num = thread_num > MAX_THREAD_NUM ? MAX_THREAD_NUM : thread_num;
  thread_pool->is_alive = ATOMIC_VAR_INIT(true);
  thread_pool->mode = mode;
  thread_pool->thread_list = NULL;
  if (thread_num > 1) {
    thread_pool->thread_list = (ThreadList *)malloc(sizeof(ThreadList));
    if (thread_pool->thread_list == NULL) {
      LOG_ERROR("create thread list failed");
      DestroyThreadPool(thread_pool_id);
      return RET_TP_ERROR;
    }
    thread_pool->thread_list->head = NULL;
    thread_pool->thread_list->tail = NULL;
    thread_pool->thread_list->size = 0;
    pthread_mutex_init(&thread_pool->thread_list->lock, NULL);
  }
  for (int i = 0; i < thread_pool->thread_num - 1; ++i) {
    int ret = CreateNewThread(thread_pool_id, i);
    if (ret != RET_TP_OK) {
      LOG_ERROR("create thread %d failed", i);
      DestroyThreadPool(thread_pool_id);
      return RET_TP_ERROR;
    }
  }
  return RET_TP_OK;
}

int ConfigThreadPool(int thread_pool_id, int thread_num, int mode) {
  LOG_INFO("config: thread_pool_id: %d, thread_num: %d, mode: %d, is_created: %d, refcount: %d", thread_pool_id,
           thread_num, mode, thread_pool_is_created[thread_pool_id], thread_pool_refcount[thread_pool_id]);
  if (thread_pool_id >= MAX_THREAD_POOL_NUM) {
    LOG_ERROR("invalid context id: %d", thread_pool_id);
    return RET_TP_ERROR;
  }
  if (thread_num <= 0 || thread_num > MAX_THREAD_NUM) {
    LOG_ERROR("invalid thread num: %d", thread_num);
    return RET_TP_ERROR;
  }
  thread_pool_refcount[thread_pool_id] += 1;
  int ret;
  if (thread_pool_is_created[thread_pool_id]) {
    ret = ReConfigThreadPool(thread_pool_id, thread_num, mode);
    if (ret != RET_TP_OK) {
      LOG_ERROR("reconfig thread pool failed, thread_pool_id: %d, thread_num: %d, mode: %d", thread_pool_id, thread_num,
               mode);
    }
  } else {
    thread_pool_is_created[thread_pool_id] = true;
    ret = CreateThreadPool(thread_pool_id, thread_num, mode);
    if (ret != RET_TP_OK) {
      LOG_ERROR("create thread pool failed, thread_pool_id: %d, thread_num: %d, mode: %d", thread_pool_id, thread_num,
               mode);
    }
  }
  return ret;
}

void ActivateThreadPool(int thread_pool_id) {
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return;
  }
  ThreadList *thread_list = thread_pool->thread_list;
  if (thread_list == NULL) {
    LOG_ERROR("thread pool: %d list is null", thread_pool_id);
    return;
  }
  Thread *thread = thread_list->head;
  while (thread != NULL) {
    sem_post(&thread->sem);
    thread->activate = true;
    thread = thread->next;
  }
}

void DeactivateThreadPool(int thread_pool_id) {
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return;
  }
  ThreadList *thread_list = thread_pool->thread_list;
  if (thread_list == NULL) {
    LOG_ERROR("thread pool: %d list is null", thread_pool_id);
    return;
  }
  Thread *thread = thread_list->head;
  while (thread != NULL) {
    thread->activate = false;
    thread = thread->next;
  }
}

void DestroyThreadPool(int thread_pool_id) {
  thread_pool_refcount[thread_pool_id]--;
  if (thread_pool_refcount[thread_pool_id] > 0) {
    LOG_ERROR("no need to free, thread_pool_id: %d, refcount: %d",
              thread_pool_id, thread_pool_refcount[thread_pool_id]);
    return;
  }
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return;
  }
  if (thread_pool->thread_list == NULL) {
    LOG_ERROR("thread pool: %d list is null", thread_pool_id);
    return;
  }
  DeactivateThreadPool(thread_pool_id);
  thread_pool_is_created[thread_pool_id] = false;
  thread_pool->is_alive = false;
  for (int i = 0; i < thread_pool->thread_num - 1; ++i) {
    Thread *thread = GetThread(thread_pool_id, i);
    if (thread != NULL) {
      FreeThread(thread_pool->thread_list, thread);
    }
  }
  free(thread_pool->thread_list);
  thread_pool->thread_list = NULL;
  LOG_INFO("destroy thread pool success, thread_pool_id: %d, refcount: %d", thread_pool_id,
           thread_pool_refcount[thread_pool_id]);
}

int GetCurrentThreadNum(int thread_pool_id) {
  ThreadPool *thread_pool = GetInstance(thread_pool_id);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instane failed");
    return 0;
  }
  return thread_pool->thread_num;
}
