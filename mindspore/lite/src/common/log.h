/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_COMMON_LOG_H_
#define MINDSPORE_LITE_SRC_COMMON_LOG_H_

#include <memory>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include "utils/overload.h"

#ifdef MS_COMPILE_OHOS
#define LOG_DOMAIN 0x2102
#define LOG_TAG "MS_Lite"
#define FORMAT "[%{public}s:%{public}d] %{public}s# %{public}s"
#include "hilog/log.h"
#endif
// NOTICE: when relative path of 'log.h' changed, macro 'LITE_LOG_HEAR_FILE_REL_PATH' must be changed
#ifndef LITE_LOG_HEAR_FILE_REL_PATH
#define LITE_LOG_HEAR_FILE_REL_PATH "mindspore/lite/src/common/log.h"
#endif

// Get start index of file relative path in __FILE__
static constexpr size_t GetRealPathPos() noexcept {
  return sizeof(__FILE__) > sizeof(LITE_LOG_HEAR_FILE_REL_PATH) ? sizeof(__FILE__) - sizeof(LITE_LOG_HEAR_FILE_REL_PATH)
                                                                : 0;
}

namespace mindspore {
#define LITE_FILE_NAME                                                                          \
  (sizeof(__FILE__) > GetRealPathPos() ? static_cast<const char *>(__FILE__) + GetRealPathPos() \
                                       : static_cast<const char *>(__FILE__))

struct LiteLocationInfo {
  LiteLocationInfo(const char *file, int line, const char *func) : file_(file), line_(line), func_(func) {}

  ~LiteLocationInfo() = default;

  const char *file_;
  int line_;
  const char *func_;
};

class LiteLogStream {
 public:
  LiteLogStream() { sstream_ = std::make_shared<std::stringstream>(); }

  ~LiteLogStream() = default;

  template <typename T>
  LiteLogStream &operator<<(const T &val) noexcept {
    (*sstream_) << val;
    return *this;
  }

  template <typename T>
  LiteLogStream &operator<<(const std::vector<T> &val) noexcept {
    (*sstream_) << "[";
    for (size_t i = 0; i < val.size(); i++) {
      (*this) << val[i];
      if (i + 1 < val.size()) {
        (*sstream_) << ", ";
      }
    }
    (*sstream_) << "]";
    return *this;
  }

  template <typename K, typename V>
  LiteLogStream &operator<<(const std::unordered_map<K, V> &val) noexcept {
    (*sstream_) << "{";
    size_t index = 0;
    for (auto &item : val) {
      (*this) << item.first << ": " << item.second;
      if (index + 1 < val.size()) {
        (*sstream_) << ", ";
      }
    }
    (*sstream_) << "}";
    return *this;
  }

  template <typename K, typename V>
  LiteLogStream &operator<<(const std::map<K, V> &val) noexcept {
    (*sstream_) << "{";
    size_t index = 0;
    for (auto &item : val) {
      (*this) << item.first << ": " << item.second;
      if (index + 1 < val.size()) {
        (*sstream_) << ", ";
      }
    }
    (*sstream_) << "}";
    return *this;
  }

  LiteLogStream &operator<<(std::ostream &func(std::ostream &os)) noexcept {
    (*sstream_) << func;
    return *this;
  }
  friend class LiteLogWriter;

 private:
  std::shared_ptr<std::stringstream> sstream_;
};

enum class LiteLogLevel : int { DEBUG = 0, INFO, WARNING, ERROR };

class LiteLogWriter {
 public:
  LiteLogWriter(const LiteLocationInfo &location, mindspore::LiteLogLevel log_level)
      : location_(location), log_level_(log_level) {}

  ~LiteLogWriter() = default;

#ifdef _WIN32
  __declspec(dllexport) void operator<(const LiteLogStream &stream) const noexcept;
#else
  __attribute__((visibility("default"))) void operator<(const LiteLogStream &stream) const noexcept;
#endif

 private:
  void OutputLog(const std::ostringstream &msg) const;

  LiteLocationInfo location_;
  LiteLogLevel log_level_;
};

#define MSLOG_IF(level)                                                                                  \
  mindspore::LiteLogWriter(mindspore::LiteLocationInfo(LITE_FILE_NAME, __LINE__, __FUNCTION__), level) < \
    mindspore::LiteLogStream()

#define MS_LOG(level) MS_LOG_##level

#define MS_LOG_DEBUG MSLOG_IF(mindspore::LiteLogLevel::DEBUG)
#define MS_LOG_INFO MSLOG_IF(mindspore::LiteLogLevel::INFO)
#define MS_LOG_WARNING MSLOG_IF(mindspore::LiteLogLevel::WARNING)
#define MS_LOG_ERROR MSLOG_IF(mindspore::LiteLogLevel::ERROR)
}  // namespace mindspore

#ifdef Debug
#include <cassert>
#define MS_ASSERT(f) assert(f)
#else
#define MS_ASSERT(f) ((void)0)
#endif
#endif  // MINDSPORE_LITE_SRC_COMMON_LOG_H_
