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
#include <unistd.h>

#ifdef __WIN32__
#include <windows.h>
#endif

#ifdef __ANDROID__
#define BIND_CORE
#include <sched.h>
#endif
#ifdef MS_COMPILE_IOS
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/machine.h>
#endif  // MS_COMPILE_IOS

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
#define RET_TP_ERROR (-8)
#define RET_TP_SYSTEM_ERROR (-1)

#define DEFAULT_SPIN_COUNT (30000)

typedef struct {
  int (*func)(void *arg, int);
  void *content;
  int *return_code;
  int task_num;
} Task;

typedef struct Thread {
  void *thread_pool;
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
  sem_t sem_inited;
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

Thread *GetThread(struct ThreadPool *thread_pool, int thread_id) {
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed, thread_id: %d", thread_id);
    return NULL;
  }
  ThreadList *thread_list = thread_pool->thread_list;
  if (thread_list == NULL) {
    LOG_ERROR("thead list is null");
    return NULL;
  }
  if (thread_id >= thread_list->size) {
    LOG_ERROR("invalid thread id: %d, thread size: %d", thread_id, thread_list->size);
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
  while (true) {
    if (thread != NULL && !thread->is_running) {
      (void)sem_destroy(&thread->sem);
      free(thread);
      thread = NULL;
      break;
    }
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

enum Arch {
  UnKnown_Arch = 0,
  Cortex_A5,
  Cortex_A7,
  Cortex_A8,
  Cortex_A9,
  Cortex_A12,
  Cortex_A15,
  Cortex_A17,
  Cortex_A32,
  Cortex_A34,
  Cortex_A35,
  Cortex_A53,
  Cortex_A55,
  Cortex_A57,
  Cortex_A65,
  Cortex_A72,
  Cortex_A73,
  Cortex_A75,
  Cortex_A76,
  Cortex_A77,
  Cortex_A78,
  Cortex_X1
};

typedef struct {
  int core_id;
  int max_freq;
  enum Arch arch;
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

int ParseCpuPart(const char *line, int start, int size) {
  int cpu_part = 0;
  for (int i = start; i < size && i < start + 3; i++) {
    char c = line[i];
    int d;
    if (c >= '0' && c <= '9') {
      d = c - '0';
    } else if ((c - 'A') < 6) {
      d = 10 + (c - 'A');
    } else if ((c - 'a') < 6) {
      d = 10 + (c - 'a');
    } else {
      LOG_ERROR("CPU part in /proc/cpuinfo is ignored due to unexpected non-hex character");
      break;
    }
    cpu_part = cpu_part * 16 + d;
  }
  return cpu_part;
}

enum Arch GetArch(int cpu_part) {
  // https://en.wikipedia.org/wiki/Comparison_of_ARMv7-A_cores
  // https://en.wikipedia.org/wiki/Comparison_of_ARMv8-A_cores
  switch (cpu_part) {
    case 0x800:  // High-performance Kryo 260 (r10p2) / Kryo 280 (r10p1) "Gold" -> Cortex-A73
      return Cortex_A73;
    case 0x801:  // Low-power Kryo 260 / 280 "Silver" -> Cortex-A53
      return Cortex_A53;
    case 0x802:  // High-performance Kryo 385 "Gold" -> Cortex-A75
      return Cortex_A75;
    case 0x803:  // Low-power Kryo 385 "Silver" -> Cortex-A55r0
      return Cortex_A55;
    case 0x804:  // High-performance Kryo 485 "Gold" / "Gold Prime" -> Cortex-A76
      return Cortex_A76;
    case 0x805:  // Low-performance Kryo 485 "Silver" -> Cortex-A55
      return Cortex_A55;
    case 0xC05:
      return Cortex_A5;
    case 0xC07:
      return Cortex_A7;
    case 0xC08:
      return Cortex_A8;
    case 0xC09:
      return Cortex_A9;
    case 0xC0C:
      return Cortex_A12;
    case 0xC0D:
      return Cortex_A12;
    case 0xC0E:
      return Cortex_A17;
    case 0xC0F:
      return Cortex_A15;
    case 0xD01:  // also Huawei Kunpeng 920 series taishan_v110 when not on android
      return Cortex_A32;
    case 0xD02:
      return Cortex_A34;
    case 0xD03:
      return Cortex_A53;
    case 0xD04:
      return Cortex_A35;
    case 0xD05:
      return Cortex_A55;
    case 0xD06:
      return Cortex_A65;
    case 0xD07:
      return Cortex_A57;
    case 0xD08:
      return Cortex_A72;
    case 0xD09:
      return Cortex_A73;
    case 0xD0A:
      return Cortex_A75;
    case 0xD0B:
      return Cortex_A76;
    case 0xD0D:
      return Cortex_A77;
    case 0xD0E:  // Cortex-A76AE
      return Cortex_A76;
    case 0xD40:  // Kirin 980 Big/Medium cores -> Cortex-A76
      return Cortex_A76;
    case 0xD41:
      return Cortex_A78;
    case 0xD43:  // Cortex-A65AE
      return Cortex_A65;
    case 0xD44:
      return Cortex_X1;
    default:
      return UnKnown_Arch;
  }
}

int SetArch(CpuInfo *freq_set, int core_num) {
  if (core_num <= 0) {
    LOG_ERROR("core_num must be greater than 0.");
    return RET_TP_ERROR;
  }
  FILE *fp = fopen("/proc/cpuinfo", "r");
  if (fp == NULL) {
    LOG_ERROR("read /proc/cpuinfo error.");
    return RET_TP_ERROR;
  }
  enum Arch *archs = malloc(core_num * sizeof(enum Arch));
  if (archs == NULL) {
    fclose(fp);
    LOG_ERROR("malloc memory for archs error.");
    return RET_TP_ERROR;
  }
  const int max_line_size = 1024;
  char line[max_line_size] = {0};
  int count = 0;
  while (!feof(fp)) {
    fgets(line, max_line_size, fp);
    // line start with "CPU part"
    if (0 == memcmp(line, "CPU part", 8)) {
      // get number like 0xD03
      for (int i = 0; i < max_line_size - 4; ++i) {
        if (line[i] == '0' && line[i + 1] == 'x') {
          int cpu_part = ParseCpuPart(line, i + 2, max_line_size);
          enum Arch arch = GetArch(cpu_part);
          if (arch == UnKnown_Arch) {
            LOG_ERROR("cpu's architecture is unknown.");
            free(archs);
            fclose(fp);
            return RET_TP_ERROR;
          }
          count++;
          if (count > core_num) {
            LOG_ERROR("number of cpu_part in /proc/cpuinfo is more than core_num.");
            free(archs);
            fclose(fp);
            return RET_TP_ERROR;
          }
          archs[count - 1] = arch;
        }
      }
    }
  }
  if (count < core_num) {
    LOG_ERROR("number of cpu_part in /proc/cpuinfo is less than core_num.");
    free(archs);
    fclose(fp);
    return RET_TP_ERROR;
  }
  for (int i = 0; i < core_num; ++i) {
    freq_set[i].arch = archs[i];
  }
  free(archs);
  fclose(fp);
  return RET_TP_OK;
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
    freq_set[i].arch = UnKnown_Arch;
  }
  int err_code = SetArch(freq_set, gCoreNum);
  if (err_code != RET_TP_OK) {
    LOG_INFO("set arch failed, ignoring arch.");
  }
  // sort core id by frequency into descending order
  for (int i = 0; i < gCoreNum; ++i) {
    for (int j = i + 1; j < gCoreNum; ++j) {
      if (freq_set[i].max_freq < freq_set[j].max_freq ||
          (freq_set[i].max_freq == freq_set[j].max_freq && freq_set[i].arch <= freq_set[j].arch)) {
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
#if defined(__APPLE__)
  LOG_ERROR("not bind thread to apple's cpu.");
  return RET_TP_ERROR;
#else
  int ret = pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), cpuSet);
  if (ret != RET_TP_OK) {
    LOG_ERROR("set thread: %d to cpu failed", thread_id);
    return RET_TP_SYSTEM_ERROR;
  }
#endif  // __APPLE__
#endif
  return RET_TP_OK;
}

int BindMasterThread(struct ThreadPool *thread_pool, bool is_bind) {
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed");
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

int FreeBindSalverThreads(struct ThreadPool *thread_pool) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int i = 0; i < gHigNum + gMidNum; ++i) {
    CPU_SET(cpu_cores[i], &mask);
  }
  for (int i = 0; i < thread_pool->thread_num - 1; ++i) {
    Thread *thread = GetThread(thread_pool, i);
    if (thread == NULL) {
      LOG_ERROR("get thread failed, thread_id: %d", i);
      return false;
    }
    int ret = SetAffinity(thread->pthread, &mask);
    if (ret != RET_TP_OK) {
      LOG_ERROR("set thread affinity failed");
      return RET_TP_ERROR;
    }
  }
  return RET_TP_OK;
}

int DoBindSalverThreads(struct ThreadPool *thread_pool) {
  cpu_set_t mask;
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
    Thread *thread = GetThread(thread_pool, i);
    if (thread == NULL) {
      LOG_ERROR("get thread failed, thread_id: %d", i);
      return false;
    }
    int ret = SetAffinity(thread->pthread, &mask);
    if (ret != RET_TP_OK) {
      LOG_ERROR("set thread affinity failed");
      return RET_TP_ERROR;
    }
  }
  return RET_TP_OK;
}

int BindSalverThreads(struct ThreadPool *thread_pool, bool is_bind) {
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed");
    return RET_TP_ERROR;
  }
  int ret;
  if (is_bind && thread_pool->mode != NO_BIND_MODE) {
    ret = DoBindSalverThreads(thread_pool);
  } else {
    ret = FreeBindSalverThreads(thread_pool);
  }
  if (ret == RET_TP_OK) {
    LOG_INFO("BindSalverThreads success");
  }
  return ret;
}
#endif

int BindThreads(struct ThreadPool *thread_pool, bool is_bind, int mode) {
#ifdef BIND_CORE
  if (mode == NO_BIND_MODE) {
    return RET_TP_OK;
  }
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed");
    return RET_TP_ERROR;
  }
  thread_pool->mode = mode;
  int ret = BindMasterThread(thread_pool, is_bind);
  if (ret != RET_TP_OK) {
    LOG_ERROR("bind master thread failed.");
  }
  ret = BindSalverThreads(thread_pool, is_bind);
  if (ret != RET_TP_OK) {
    LOG_ERROR("bind salver thread failed.");
  }
  return ret;
#else
  return RET_TP_OK;
#endif
}

bool PushTaskToQueue(struct ThreadPool *thread_pool, int thread_id, Task *task) {
  Thread *thread = GetThread(thread_pool, thread_id);
  if (thread == NULL) {
    LOG_ERROR("get thread failed, thread_id: %d", thread_id);
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
  sem_post(&thread->sem);
  return true;
}

bool PopTaskFromQueue(Thread *thread, Task **task) {
  if (thread == NULL) {
    LOG_ERROR("thread is nullptr");
    return false;
  }
  if (atomic_load_explicit(&thread->task_size, memory_order_relaxed) == 0) {
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

void WaitAllThread(struct ThreadPool *thread_pool) {
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed");
    return;
  }
  bool k_success_flag = false;
  while (!k_success_flag) {
    k_success_flag = true;
    for (int i = 0; i < thread_pool->thread_num - 1; ++i) {
      Thread *thread = GetThread(thread_pool, i);
      if (thread == NULL) {
        LOG_ERROR("get thread failed, thread_id: %d", i);
        return;
      }
      if (atomic_load_explicit(&thread->task_size, memory_order_acquire) != 0) {
        k_success_flag = false;
        break;
      }
    }
  }
}

int DistributeTask(struct ThreadPool *thread_pool, Task *task, int task_num) {
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed");
    return RET_TP_ERROR;
  }
  if (task_num > thread_pool->thread_num || task_num <= 1) {
    LOG_ERROR("invalid task num: %d, thread num: %d", task_num, thread_pool->thread_num);
    return RET_TP_ERROR;
  }
  bool k_success_flag = false;
  if (thread_pool->thread_num < task_num) {
    LOG_ERROR("task_num: %d should not be larger than thread num: %d", task_num, thread_pool->thread_num);
    return RET_TP_ERROR;
  }
  for (int i = 0; i < task_num - 1; ++i) {
    do {
      k_success_flag = true;
      if (!PushTaskToQueue(thread_pool, i, task)) {
        k_success_flag = false;
      }
    } while (!k_success_flag);
  }
  // master thread
  if (task->func == NULL) {
    LOG_ERROR("task->func is nullptr");
    return RET_TP_ERROR;
  }
  if (task->task_num <= task_num - 1) {
    LOG_ERROR("task_num out of range in master thread");
    return RET_TP_ERROR;
  }
  task->return_code[task_num - 1] = task->func(task->content, task_num - 1);
  // wait
  WaitAllThread(thread_pool);
  for (size_t i = 0; i < task->task_num; i++) {
    if (task->return_code[i] != 0) {
      return task->return_code[i];
    }
  }
  return RET_TP_OK;
}

int AddTask(struct ThreadPool *thread_pool, int func(void *, int), void *content, int task_num) {
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed");
    return RET_TP_ERROR;
  }
  // if single thread, run master thread
  if (thread_pool->thread_num <= 1 || task_num <= 1) {
    for (int i = 0; i < task_num; ++i) {
      int ret = func(content, i);
      if (ret != 0) {
        return ret;
      }
    }
    return RET_TP_OK;
  }
  Task task;
  task.func = func;
  task.content = content;
  task.return_code = (int *)malloc(sizeof(int) * task_num);
  task.task_num = task_num;
  if (task.return_code == NULL) {
    LOG_ERROR("malloc return code return nullptr");
    return RET_TP_ERROR;
  }
  memset(task.return_code, 0, sizeof(int) * task_num);
  int ret = DistributeTask(thread_pool, &task, task_num);
  free(task.return_code);
  return ret;
}

int ParallelLaunch(struct ThreadPool *thread_pool, int (*func)(void *, int), void *content, int task_num) {
  return AddTask(thread_pool, func, content, task_num);
}

void ThreadRun(Thread *thread) {
  thread->is_running = true;
  ThreadPool *thread_pool = (ThreadPool *)(thread->thread_pool);
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed");
    thread->is_running = false;
    return;
  }
  Task *task = NULL;
  int thread_id = thread->thread_id;
  int spin_count = 0;
  sem_post(&thread->sem_inited);
  while (thread_pool->is_alive) {
    while (thread->activate) {
      if (PopTaskFromQueue(thread, &task)) {
        if (task->func == NULL) {
          LOG_ERROR("task->func is nullptr");
          return;
        }
        if (task->task_num <= thread_id) {
          LOG_ERROR("task_num out of range in worker thread");
          return;
        }
        task->return_code[thread_id] = task->func(task->content, thread_id);
        atomic_fetch_sub_explicit(&thread->task_size, 1, memory_order_release);
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

void PushThreadToList(struct ThreadPool *thread_pool, Thread *thread) {
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed");
    return;
  }
  ThreadList *thread_list = thread_pool->thread_list;
  if (thread_list == NULL) {
    LOG_ERROR("thread list is null");
    DestroyThreadPool(thread_pool);
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

int CreateNewThread(struct ThreadPool *thread_pool, int thread_id) {
  LOG_INFO("create thread: %d", thread_id);
  Thread *thread = (Thread *)malloc(sizeof(Thread));
  if (thread == NULL) {
    LOG_ERROR("create thread failed");
    DestroyThreadPool(thread_pool);
    return RET_TP_ERROR;
  }
  thread->thread_pool = thread_pool;
  thread->thread_id = thread_id;
  thread->head = ATOMIC_VAR_INIT(0);
  thread->tail = ATOMIC_VAR_INIT(0);
  thread->task_size = ATOMIC_VAR_INIT(0);
  thread->activate = ATOMIC_VAR_INIT(true);
  thread->is_running = ATOMIC_VAR_INIT(true);
  thread->next = NULL;
  sem_init(&thread->sem, 0, 0);
  sem_init(&thread->sem_inited, 0, 0);
  PushThreadToList(thread_pool, thread);
  pthread_create(&thread->pthread, NULL, (void *)ThreadRun, thread);
  sem_wait(&thread->sem_inited);
  pthread_detach(thread->pthread);
  return RET_TP_OK;
}

ThreadPool *CreateThreadPool(int thread_num, int mode) {
#ifdef __WIN32__
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  long max_thread_num = sys_info.dwNumberOfProcessors;
#else
  long max_thread_num = sysconf(_SC_NPROCESSORS_ONLN);
#endif
  LOG_INFO("create thread pool, thread_num: %d, mode: %d", thread_num, mode);
  if (thread_num <= 0 || thread_num > max_thread_num) {
    LOG_ERROR("invalid thread num: %d", thread_num);
    return NULL;
  }
#ifdef BIND_CORE
  if (run_once) {
    int ret = SortCpuProcessor();
    run_once = false;
    if (ret != RET_TP_OK) {
      LOG_ERROR("SortCpuProcessor failed");
      return NULL;
    }
  }
#endif
  ThreadPool *thread_pool = (struct ThreadPool *)(malloc(sizeof(ThreadPool)));
  if (thread_pool == NULL) {
    LOG_ERROR("Malloc ThreadPool failed");
    return NULL;
  }
  thread_pool->thread_num = thread_num > max_thread_num ? max_thread_num : thread_num;
  thread_pool->is_alive = ATOMIC_VAR_INIT(true);
  thread_pool->mode = mode;
  thread_pool->thread_list = NULL;
  if (thread_num > 1) {
    thread_pool->thread_list = (ThreadList *)malloc(sizeof(ThreadList));
    if (thread_pool->thread_list == NULL) {
      LOG_ERROR("create thread list failed");
      DestroyThreadPool(thread_pool);
      thread_pool = NULL;
      return NULL;
    }
    thread_pool->thread_list->head = NULL;
    thread_pool->thread_list->tail = NULL;
    thread_pool->thread_list->size = 0;
    pthread_mutex_init(&thread_pool->thread_list->lock, NULL);
  }
  for (int i = 0; i < thread_pool->thread_num - 1; ++i) {
    int ret = CreateNewThread(thread_pool, i);
    if (ret != RET_TP_OK) {
      LOG_ERROR("create thread %d failed", i);
      DestroyThreadPool(thread_pool);
      thread_pool = NULL;
      return NULL;
    }
  }
  if (thread_pool == NULL) {
    LOG_ERROR("create thread failed");
    DestroyThreadPool(thread_pool);
    return NULL;
  }
  return thread_pool;
}

void ActivateThreadPool(struct ThreadPool *thread_pool) {
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed");
    return;
  }
  ThreadList *thread_list = thread_pool->thread_list;
  if (thread_list == NULL) {
    LOG_ERROR("thread pool's list is null");
    return;
  }
  Thread *thread = thread_list->head;
  while (thread != NULL) {
    sem_post(&thread->sem);
    thread->activate = true;
    thread = thread->next;
  }
}

void DeactivateThreadPool(struct ThreadPool *thread_pool) {
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed");
    return;
  }
  ThreadList *thread_list = thread_pool->thread_list;
  if (thread_list == NULL) {
    LOG_ERROR("thread pool's list is null");
    return;
  }
  Thread *thread = thread_list->head;
  while (thread != NULL) {
    thread->activate = false;
    thread = thread->next;
  }
}

void DestroyThreadPool(struct ThreadPool *thread_pool) {
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed");
    return;
  }
  if (thread_pool->thread_list == NULL) {
    LOG_ERROR("thread pool's list is null");
    return;
  }
  DeactivateThreadPool(thread_pool);
  thread_pool->is_alive = false;
  LOG_ERROR("DestroyThreadPool thread num : %d", thread_pool->thread_num);
  for (int i = 0; i < thread_pool->thread_num - 1; ++i) {
    Thread *thread = GetThread(thread_pool, i);
    if (thread != NULL) {
      FreeThread(thread_pool->thread_list, thread);
    }
  }
  free(thread_pool->thread_list);
  thread_pool->thread_list = NULL;
  LOG_INFO("destroy thread pool success");
}

int GetCurrentThreadNum(struct ThreadPool *thread_pool) {
  if (thread_pool == NULL) {
    LOG_ERROR("get thread pool instance failed");
    return 0;
  }
  return thread_pool->thread_num;
}
