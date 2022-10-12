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

#include "wrapper/thread/micro_core_affinity.h"
#if defined(_MSC_VER) || defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include "nnacl/op_base.h"

int GetCpuCoreNum() {
  int core_num = 1;
#if defined(_MSC_VER) || defined(_WIN32)
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  core_num = sysinfo.dwNumberOfProcessors;
#else
  core_num = sysconf(_SC_NPROCESSORS_CONF);
#endif
  return core_num;
}

int ConcatCPUPath(int cpuID, const char *str1, const char *str2, char *str3) {
  if (cpuID > MAX_CPU_ID || str1 == NULL || str2 == NULL) {
    return RET_TP_SYSTEM_ERROR;
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

int SortCpuCores(int *cpu_cores, int *cpu_high_num, int *cpu_mid_num, int *cpu_little_num) {
  int cpu_core_num = GetCpuCoreNum();
  if (cpu_core_num <= 0 || cpu_core_num > MAX_CPU_ID) {
    LOG_ERROR("invalid cpu count");
    return RET_TP_SYSTEM_ERROR;
  }
  CpuInfo freq_set[MAX_CPU_ID] = {0};
  for (int i = 0; i < cpu_core_num; ++i) {
    int max_freq = GetMaxFrequence(i);
    freq_set[i].core_id = i;
    freq_set[i].max_freq = max_freq;
  }
  // sort core id by frequency
  for (int i = 0; i < cpu_core_num; ++i) {
    for (int j = i + 1; j < cpu_core_num; ++j) {
      if (freq_set[i].max_freq <= freq_set[j].max_freq) {
        CpuInfo temp = freq_set[i];
        freq_set[i] = freq_set[j];
        freq_set[j] = temp;
      }
    }
  }
  for (int i = 0; i < cpu_core_num; ++i) {
    cpu_cores[i] = freq_set[i].core_id;
    LOG_INFO("sorted_order: %d, frequency: %d", freq_set[i].core_id, freq_set[i].max_freq);
  }
  int max_freq = freq_set[0].max_freq;
  int min_freq = freq_set[cpu_core_num - 1].max_freq;
  for (int i = 0; i < cpu_core_num; ++i) {
    if (freq_set[i].max_freq == max_freq) {
      (*cpu_high_num)++;
    }
    if (freq_set[i].max_freq == min_freq) {
      (*cpu_little_num)++;
    }
  }
  *cpu_mid_num = cpu_core_num - *cpu_high_num - *cpu_little_num;
  if (*cpu_high_num == cpu_core_num || max_freq == min_freq) {
    // fix MTK800
    *cpu_high_num = 2;
    *cpu_mid_num = 2;
    LOG_INFO("core frequency may be wrong.");
  }
  LOG_INFO("gCoreNum: %d, gHigNum: %d, gMidNum: %d, gLitNum: %d", cpu_core_num, *cpu_high_num, *cpu_mid_num,
           *cpu_little_num);
  return RET_TP_OK;
}

int InitBindCoreId(size_t thread_num, enum BindMode bind_mode, int *bind_id) {
  if (thread_num > MAX_CPU_ID) {
    LOG_ERROR("invalid thread num,exceed max cpu id");
    return RET_TP_ERROR;
  }
  int sort_cpu_id[MAX_CPU_ID] = {0};
  int cpu_high_num = 0;
  int cpu_mid_num = 0;
  int cpu_little_num = 0;
  int ret = SortCpuCores(sort_cpu_id, &cpu_high_num, &cpu_mid_num, &cpu_little_num);
  if (ret != RET_TP_OK) {
    return ret;
  }
  int cpu_nums = cpu_high_num + cpu_mid_num + cpu_little_num;
  if (bind_mode == Power_Higher || bind_mode == Power_NoBind) {
    for (size_t i = 0; i < thread_num; ++i) {
      bind_id[i] = sort_cpu_id[i % cpu_nums];
    }
  } else if (bind_mode == Power_Middle) {
    for (size_t i = 0; i < thread_num; ++i) {
      bind_id[i] = sort_cpu_id[(i + cpu_high_num) % cpu_nums];
    }
  } else {
    LOG_ERROR("invalid bind mode");
    return RET_TP_ERROR;
  }
  return RET_TP_OK;
}

#if defined(BIND_CORE)
int SetAffinity(const pthread_t thread_id, cpu_set_t *cpu_set) {
#ifdef __ANDROID__
#if __ANDROID_API__ >= 21
  int ret = sched_setaffinity(pthread_gettid_np(thread_id), sizeof(cpu_set_t), cpu_set);
  if (ret != RET_TP_OK) {
    LOG_ERROR("invalid bind mode");
    return RET_TP_SYSTEM_ERROR;
  }
#endif
#else
  int ret = pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), cpu_set);
  if (ret != RET_TP_OK) {
    LOG_ERROR("invalid bind mode");
    return RET_TP_SYSTEM_ERROR;
  }
#endif
  return RET_TP_OK;
}
#endif

int FreeScheduleThreads(const int *bind_id, ThreadPool *g_pool) {
#if defined(BIND_CORE)
  cpu_set_t mask;
  CPU_ZERO(&mask);
  int thread_num = g_pool->max_thread_num;
  for (int i = 0; i < thread_num; i++) {
    CPU_SET(bind_id[i], &mask);
  }
  for (int i = 0; i < thread_num; i++) {
    int ret;
    if (i == 0) {
      ret = SetAffinity(pthread_self(), &mask);
    } else {
      ret = SetAffinity(g_pool->thread_id[i - 1], &mask);
    }
    if (ret != RET_TP_OK) {
      return ret;
    }
  }
#endif  // BIND_CORE
  return RET_TP_OK;
}

int BindThreadsToCore(const int *bind_id, ThreadPool *g_pool) {
#if defined(BIND_CORE)
  int thread_num = g_pool->max_thread_num;
  for (size_t i = 0; i < thread_num; i++) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(bind_id[i], &mask);
    int ret;
    if (i == 0) {
      ret = SetAffinity(pthread_self(), &mask);
    } else {
      ret = SetAffinity(g_pool->thread_id[i - 1], &mask);
    }
    if (ret != RET_TP_OK) {
      LOG_ERROR("error binding task %zu to core %d\n", i, bind_id[i]);
      return ret;
    }
  }
#endif  // BIND_CORE
  return RET_TP_OK;
}

int BindThreads(enum BindMode bind_mode, ThreadPool *g_pool) {
#if defined(BIND_CORE)
  int bind_id[MAX_CPU_ID];
  int ret = InitBindCoreId(g_pool->max_thread_num, bind_mode, bind_id);
  if (ret != RET_TP_OK) {
    return ret;
  }
  if (bind_mode == Power_NoBind) {
    return FreeScheduleThreads(bind_id, g_pool);
  } else {
    return BindThreadsToCore(bind_id, g_pool);
  }
#endif  // BIND_CORE
  return RET_TP_OK;
}
