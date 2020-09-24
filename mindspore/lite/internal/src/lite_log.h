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

#ifndef MINDSPORE_LITE_INTERNAL_SRC_LITE_LOG_H_
#define MINDSPORE_LITE_INTERNAL_SRC_LITE_LOG_H_

#include <stdlib.h>
#include <stdio.h>
#ifndef Release
#include <assert.h>
#endif

#ifdef Debug
#define LITE_DEBUG_LOG(format, ...) \
  printf("[DEBUG] [%s %s] [%s] [%d] " format "\n", __DATE__, __TIME__, __FILE__, __LINE__, __VA_ARGS__)
#define LITE_INFO_LOG(format, ...) \
  printf("[INFO] [%s %s] [%s] [%d] " format "\n", __DATE__, __TIME__, __FILE__, __LINE__, __VA_ARGS__)
#define LITE_LOG_INFO(...) printf("[INFO] [%s %s] [%s] [%d] %s\n", __DATE__, __TIME__, __FILE__, __LINE__, __VA_ARGS__)
#define LITE_WARNING_LOG(format, ...) \
  printf("[WARNING] [%s %s] [%s] [%d] " format "\n", __DATE__, __TIME__, __FILE__, __LINE__, __VA_ARGS__)
#define LITE_ERROR_LOG(format, ...) \
  printf("[ERROR] [%s %s] [%s] [%d] " format "\n", __DATE__, __TIME__, __FILE__, __LINE__, __VA_ARGS__)
#define LITE_LOG_ERROR(...) \
  printf("[ERROR] [%s %s] [%s] [%d] %s\n", __DATE__, __TIME__, __FILE__, __LINE__, __VA_ARGS__)
#define MS_ASSERT(f) assert(f)
#define MS_C_EXCEPTION(...)                                                                          \
  printf("[EXCEPTION] [%s %s] [%s] [%d] %s\n", __DATE__, __TIME__, __FILE__, __LINE__, __VA_ARGS__); \
  exit(1)
#else
#define LITE_DEBUG_LOG(...)
#define LITE_INFO_LOG(...)
#define LITE_LOG_INFO(...)
#define LITE_WARNING_LOG(...)
#define LITE_ERROR_LOG(...)
#define LITE_LOG_ERROR(...)
#define MS_ASSERT(f) ((void)0)
#define MS_C_EXCEPTION(...) exit(1)
#endif

#endif  // MINDSPORE_LITE_INTERNAL_SRC_LITE_LOG_H_
