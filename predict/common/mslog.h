/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PREDICT_COMMON_MSLOG_H_
#define PREDICT_COMMON_MSLOG_H_

#include <syslog.h>
#include <unistd.h>
#include <csignal>
#include <iostream>
#include <sstream>
#include <string>

#if defined(__ANDROID__) || defined(ANDROID)
#include <android/log.h>
#endif
namespace mindspore {
namespace predict {
constexpr const char *TAG = "MS_PREDICT";

constexpr int DEBUG = 1;
constexpr int INFO = 2;
constexpr int WARN = 3;
constexpr int ERROR = 4;

#define MSPREDICT_API __attribute__((visibility("default")))

bool MSPREDICT_API IsPrint(int level);

#if !defined(__ANDROID__) && !defined(ANDROID)

#if LOG_TO_FILE
#define MS_LOGD(fmt, args...)                                                                                    \
  {                                                                                                              \
    if (mindspore::predict::IsPrint(mindspore::predict::DEBUG)) {                                                \
      syslog(LOG_DEBUG, "%s|%d|%s[%d]|: " #fmt, mindspore::predict::TAG, \getpid(), __func__, __LINE__, ##args); \
    }                                                                                                            \
  }
#define MS_LOGI(fmt, args...)                                                                                   \
  {                                                                                                             \
    if (mindspore::predict::IsPrint(mindspore::predict::INFO)) {                                                \
      syslog(LOG_INFO, "%s|%d|%s[%d]|: " #fmt, mindspore::predict::TAG, \getpid(), __func__, __LINE__, ##args); \
    }                                                                                                           \
  }
#define MS_LOGW(fmt, args...)                                                                                      \
  {                                                                                                                \
    if (mindspore::predict::IsPrint(mindspore::predict::WARN)) {                                                   \
      syslog(LOG_WARNING, "%s|%d|%s[%d]|: " #fmt, mindspore::predict::TAG, \getpid(), __func__, __LINE__, ##args); \
    }                                                                                                              \
  }
#define MS_LOGE(fmt, args...)                                                                                 \
  {                                                                                                           \
    if (mindspore::predict::IsPrint(mindspore::predict::ERROR)) {                                             \
      syslog(LOG_ERR, "%s|%d|%s[%d]|: " #fmt, mindspore::predict::TAG, getpid(), __func__, __LINE__, ##args); \
    }                                                                                                         \
  }
#else

#define MS_LOGD(fmt, args...)                                                                                 \
  {                                                                                                           \
    if (mindspore::predict::IsPrint(mindspore::predict::DEBUG)) {                                             \
      printf("[DEBUG] %s|%d|%s|%s[%d]|: " #fmt "\r\n", mindspore::predict::TAG, getpid(), __FILE__, __func__, \
             __LINE__, ##args);                                                                               \
    }                                                                                                         \
  }
#define MS_LOGI(fmt, args...)                                                                                 \
  {                                                                                                           \
    if (mindspore::predict::IsPrint(mindspore::predict::INFO)) {                                              \
      printf("[INFO]  %s|%d|%s|%s[%d]|: " #fmt "\r\n", mindspore::predict::TAG, getpid(), __FILE__, __func__, \
             __LINE__, ##args);                                                                               \
    }                                                                                                         \
  }
#define MS_LOGW(fmt, args...)                                                                                 \
  {                                                                                                           \
    if (mindspore::predict::IsPrint(mindspore::predict::WARN)) {                                              \
      printf("[WARN]  %s|%d|%s|%s[%d]|: " #fmt "\r\n", mindspore::predict::TAG, getpid(), __FILE__, __func__, \
             __LINE__, ##args);                                                                               \
    }                                                                                                         \
  }
#define MS_LOGE(fmt, args...)                                                                                  \
  {                                                                                                            \
    if (mindspore::predict::IsPrint(mindspore::predict::ERROR)) {                                              \
      printf("[ERROR]  %s|%d|%s|%s[%d]|: " #fmt "\r\n", mindspore::predict::TAG, getpid(), __FILE__, __func__, \
             __LINE__, ##args);                                                                                \
    }                                                                                                          \
  }
#endif

#else

#define MS_LOGD(fmt, args...)                                                                                  \
  {                                                                                                            \
    if (mindspore::predict::IsPrint(mindspore::predict::DEBUG))                                                \
      __android_log_print(ANDROID_LOG_DEBUG, mindspore::predict::TAG, "|%d|%s[%d]|: " fmt, getpid(), __func__, \
                          __LINE__, ##args);                                                                   \
  }

#define MS_LOGI(fmt, args...)                                                                                 \
  {                                                                                                           \
    if (mindspore::predict::IsPrint(mindspore::predict::INFO))                                                \
      __android_log_print(ANDROID_LOG_INFO, mindspore::predict::TAG, "|%d|%s[%d]|: " fmt, getpid(), __func__, \
                          __LINE__, ##args);                                                                  \
  }

#define MS_LOGW(fmt, args...)                                                                                 \
  {                                                                                                           \
    if (mindspore::predict::IsPrint(mindspore::predict::WARN))                                                \
      __android_log_print(ANDROID_LOG_WARN, mindspore::predict::TAG, "|%d|%s[%d]|: " fmt, getpid(), __func__, \
                          __LINE__, ##args);                                                                  \
  }

#define MS_LOGE(fmt, args...)                                                                                  \
  {                                                                                                            \
    if (mindspore::predict::IsPrint(mindspore::predict::ERROR))                                                \
      __android_log_print(ANDROID_LOG_ERROR, mindspore::predict::TAG, "|%d|%s[%d]|: " fmt, getpid(), __func__, \
                          __LINE__, ##args);                                                                   \
  }

#endif

#define MS_LOG(severity) std::cout << std::endl
#define MS_DLOG(verboselevel) std::cout << std::endl
// Kill the process for safe exiting.
inline void KillProcess(const std::string &ret) {
  MS_LOG(ERROR) << "mindspore Exit Tip:" << ret;
  if (raise(SIGKILL) != 0) {
    MS_LOGE("Send SIGKILL to kill process failed");
  }
}
}  // namespace predict
}  // namespace mindspore

#define MS_ASSERT(expression)                                                                        \
  do {                                                                                               \
    if (!(expression)) {                                                                             \
      std::stringstream ss;                                                                          \
      ss << "Assertion failed: " << #expression << ", file: " << __FILE__ << ", line: " << __LINE__; \
      mindspore::predict::KillProcess(ss.str());                                                     \
    }                                                                                                \
  } while (0)

#define MS_EXIT(ret)                                                            \
  do {                                                                          \
    std::stringstream ss;                                                       \
    ss << (ret) << "  ( file: " << __FILE__ << ", line: " << __LINE__ << " )."; \
    mindspore::predict::KillProcess(ss.str());                                  \
  } while (0)

#define MS_PRINT_ERROR(fmt, args...) \
  printf(#fmt "\n", ##args);         \
  MS_LOGE(fmt, ##args);

#define MS_PRINT_INFO(fmt, args...) \
  printf(fmt "\n", ##args);         \
  MS_LOGI(fmt, ##args);

constexpr int LOG_CHECK_EVERY_FIRSTNUM = 10;
constexpr int LOG_CHECK_EVERY_NUM1 = 10;
constexpr int LOG_CHECK_EVERY_NUM2 = 100;
constexpr int LOG_CHECK_EVERY_NUM3 = 1000;
constexpr int LOG_CHECK_EVERY_NUM4 = 10000;

#define LOG_CHECK_ID_CONCAT(word1, word2) word1##word2

#define LOG_CHECK_ID LOG_CHECK_ID_CONCAT(__FUNCTION__, __LINE__)

#define LOG_CHECK_FIRST_N              \
  [](uint32_t firstNum) {              \
    static uint32_t LOG_CHECK_ID = 0;  \
    ++LOG_CHECK_ID;                    \
    return (LOG_CHECK_ID <= firstNum); \
  }

#define LOG_CHECK_EVERY_N1                                            \
  [](uint32_t firstNum, uint32_t num) {                               \
    static uint32_t LOG_CHECK_ID = 0;                                 \
    ++LOG_CHECK_ID;                                                   \
    return ((LOG_CHECK_ID <= firstNum) || (LOG_CHECK_ID % num == 0)); \
  }

#define LOG_CHECK_EVERY_N2                                                                     \
  [](uint32_t firstNum, uint32_t num1, uint32_t num2) {                                        \
    static uint32_t LOG_CHECK_ID = 0;                                                          \
    ++LOG_CHECK_ID;                                                                            \
    return ((LOG_CHECK_ID <= firstNum) || (LOG_CHECK_ID < num2 && LOG_CHECK_ID % num1 == 0) || \
            (LOG_CHECK_ID % num2 == 0));                                                       \
  }

#define LOG_CHECK_EVERY_N3                                                                     \
  [](uint32_t firstNum, uint32_t num1, uint32_t num2, uint32_t num3) {                         \
    static uint32_t LOG_CHECK_ID = 0;                                                          \
    ++LOG_CHECK_ID;                                                                            \
    return ((LOG_CHECK_ID <= firstNum) || (LOG_CHECK_ID < num2 && LOG_CHECK_ID % num1 == 0) || \
            (LOG_CHECK_ID < num3 && LOG_CHECK_ID % num2 == 0) || (LOG_CHECK_ID % num3 == 0));  \
  }

#define LOG_CHECK_EVERY_N4                                                                                            \
  [](uint32_t firstNum, uint32_t num1, uint32_t num2, uint32_t num3, uint32_t num4) {                                 \
    static uint32_t LOG_CHECK_ID = 0;                                                                                 \
    ++LOG_CHECK_ID;                                                                                                   \
    return ((LOG_CHECK_ID <= firstNum) || (LOG_CHECK_ID < num2 && LOG_CHECK_ID % num1 == 0) ||                        \
            (LOG_CHECK_ID < num3 && LOG_CHECK_ID % num2 == 0) || (LOG_CHECK_ID < num4 && LOG_CHECK_ID % num3 == 0) || \
            (LOG_CHECK_ID % num4 == 0));                                                                              \
  }

#define LOG_CHECK_EVERY_N                                                                        \
  []() {                                                                                         \
    static uint32_t LOG_CHECK_ID = 0;                                                            \
    ++LOG_CHECK_ID;                                                                              \
    return ((LOG_CHECK_ID <= LOG_CHECK_EVERY_FIRSTNUM) ||                                        \
            (LOG_CHECK_ID < LOG_CHECK_EVERY_NUM2 && LOG_CHECK_ID % LOG_CHECK_EVERY_NUM1 == 0) || \
            (LOG_CHECK_ID < LOG_CHECK_EVERY_NUM3 && LOG_CHECK_ID % LOG_CHECK_EVERY_NUM2 == 0) || \
            (LOG_CHECK_ID < LOG_CHECK_EVERY_NUM4 && LOG_CHECK_ID % LOG_CHECK_EVERY_NUM3 == 0) || \
            (LOG_CHECK_ID % LOG_CHECK_EVERY_NUM4 == 0));                                         \
  }

#endif  // PREDICT_COMMON_MSLOG_H_
