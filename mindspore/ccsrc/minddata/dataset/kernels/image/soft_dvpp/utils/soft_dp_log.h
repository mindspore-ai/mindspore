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

#ifndef SOFT_DP_LOG_H
#define SOFT_DP_LOG_H

#define VERSION_INFO 0x0
#define DP_DEBUG 0x1
#define DP_INFO 0x10
#define DP_WARNING 0x100
#define DP_ERR 0x1000
#define DP_EVENT 0x10000
#define DP_DEBUG_LEVEL (DP_EVENT | DP_ERR | DP_WARNING | DP_INFO | DP_DEBUG)

#if defined(DVPP_UTST) || defined(DEBUG)
#include <stdio.h>
#include <string>
#include <vector>

#define DP_LOG(model, level, format, ...)                              \
  do {                                                                 \
    if (DP_DEBUG_LEVEL & level) {                                      \
      if (DP_DEBUG & level) {                                          \
        printf(                                                        \
          "[SOFT_DP-%s] [%s %d] [DEBUG:] "                             \
          "[T%d] " format "\n",                                        \
          model, __FUNCTION__, __LINE__, VERSION_INFO, ##__VA_ARGS__); \
      } else if (DP_INFO & level) {                                    \
        printf(                                                        \
          "[SOFT_DP-%s] [%s %d] [INFO:] "                              \
          "[T%d] " format "\n",                                        \
          model, __FUNCTION__, __LINE__, VERSION_INFO, ##__VA_ARGS__); \
      } else if (DP_WARNING & level) {                                 \
        printf(                                                        \
          "[SOFT_DP-%s] [%s %d] [WARNING] "                            \
          "[T%d] " format "\n",                                        \
          model, __FUNCTION__, __LINE__, VERSION_INFO, ##__VA_ARGS__); \
      } else if (DP_ERR & level) {                                     \
        printf(                                                        \
          "[SOFT_DP-%s] [%s %d] [ERROR:] "                             \
          "[T%d] " format "\n",                                        \
          model, __FUNCTION__, __LINE__, VERSION_INFO, ##__VA_ARGS__); \
      } else {                                                         \
        printf(                                                        \
          "[SOFT_DP-%s] [%s %d] [EVENT:] "                             \
          "[T%d] " format "\n",                                        \
          model, __FUNCTION__, __LINE__, VERSION_INFO, ##__VA_ARGS__); \
      }                                                                \
    }                                                                  \
  } while (0)

#elif defined(USE_GLOG)

#include <securec.h>
#include <cstdio>
#include <vector>
#include <string>
#include "glog/logging.h"

template <typename... Args>
inline std::string GetFormatString(const char *format, Args... args) {
  char buf[BUFSIZ];
#ifdef _WIN32
  _snprintf_s(&buf[0], BUFSIZ, BUFSIZ - 1, format, args...);
#else
  snprintf_s(&buf[0], BUFSIZ, BUFSIZ - 1, format, args...);
#endif
  return buf;
}

#define DP_LOG(model, level, format, ...)                          \
  do {                                                             \
    std::string info = GetFormatString(                            \
      "[%s] [%s:%d] "                                              \
      "[T%d] " format "",                                          \
      model, __FUNCTION__, __LINE__, VERSION_INFO, ##__VA_ARGS__); \
    if (DP_WARNING & level) {                                      \
      LOG(WARNING) << info;                                        \
    } else if (DP_ERR & level) {                                   \
      LOG(ERROR) << info;                                          \
    } else {                                                       \
      LOG(INFO) << info;                                           \
    }                                                              \
  } while (0)

#else  // #if defined(DVPP_UTST) || defined(DEBUG)

#include "./slog.h"

#define DP_LOG(model, level, format, ...)                                       \
  do {                                                                          \
    if (DP_DEBUG_LEVEL & level) {                                               \
      if (DP_DEBUG & level) {                                                   \
        dlog_debug(SOFT_DP,                                                     \
                   "[%s] [%s:%d] "                                              \
                   "[T%d] " format "",                                          \
                   model, __FUNCTION__, __LINE__, VERSION_INFO, ##__VA_ARGS__); \
      } else if (DP_INFO & level) {                                             \
        dlog_info(SOFT_DP,                                                      \
                  "[%s] [%s:%d] "                                               \
                  "[T%d] " format "",                                           \
                  model, __FUNCTION__, __LINE__, VERSION_INFO, ##__VA_ARGS__);  \
      } else if (DP_WARNING & level) {                                          \
        dlog_warn(SOFT_DP,                                                      \
                  "[%s] [%s:%d] "                                               \
                  "[T%d] " format "",                                           \
                  model, __FUNCTION__, __LINE__, VERSION_INFO, ##__VA_ARGS__);  \
      } else if (DP_ERR & level) {                                              \
        dlog_error(SOFT_DP,                                                     \
                   "[%s] [%s:%d] "                                              \
                   "[T%d] " format "",                                          \
                   model, __FUNCTION__, __LINE__, VERSION_INFO, ##__VA_ARGS__); \
      } else {                                                                  \
        dlog_event(SOFT_DP,                                                     \
                   "[%s] [%s:%d] "                                              \
                   "[T%d] " format "",                                          \
                   model, __FUNCTION__, __LINE__, VERSION_INFO, ##__VA_ARGS__); \
      }                                                                         \
    }                                                                           \
  } while (0)

#endif  // #if defined(DVPP_UTST) || defined(DEBUG)

#define VPC_LOG(level, format, argv...) DP_LOG("VPC", level, format, ##argv)
#define VPC_LOGD(format, argv...) DP_LOG("VPC", DP_DEBUG, format, ##argv)
#define VPC_LOGW(format, argv...) DP_LOG("VPC", DP_WARNING, format, ##argv)
#define VPC_LOGE(format, argv...) DP_LOG("VPC", DP_ERR, format, ##argv)

#define JPEGD_LOG(level, format, argv...) DP_LOG("JPEGD", level, format, ##argv)
#define JPEGD_LOGD(format, argv...) DP_LOG("JPEGD", DP_DEBUG, format, ##argv)
#define JPEGD_LOGW(format, argv...) DP_LOG("JPEGD", DP_WARNING, format, ##argv)
#define JPEGD_LOGE(format, argv...) DP_LOG("JPEGD", DP_ERR, format, ##argv)

#define API_LOG(level, format, argv...) DP_LOG("API", level, format, ##argv)
#define API_LOGD(format, argv...) DP_LOG("API", DP_DEBUG, format, ##argv)
#define API_LOGW(format, argv...) DP_LOG("API", DP_WARNING, format, ##argv)
#define API_LOGE(format, argv...) DP_LOG("API", DP_ERR, format, ##argv)

#endif  // SOFT_DP_LOG_H
