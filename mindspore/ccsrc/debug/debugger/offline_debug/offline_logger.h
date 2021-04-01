/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef OFFLINE_LOGGER_H_
#define OFFLINE_LOGGER_H_

#include <iostream>

#define MS_LOG(level) MS_LOG_##level

#define MS_LOG_INFO static_cast<void>(0), !(DbgLogger::verbose) ? void(0) : DbgLogger(DbgLoggerLvl::INFO) < std::cout

#define MS_LOG_ERROR MS_LOG_INFO

#define MS_LOG_DEBUG MS_LOG_INFO

#define MS_LOG_WARNING MS_LOG_INFO

#define MS_LOG_EXCEPTION \
  static_cast<void>(0), !(DbgLogger::verbose) ? void(0) : DbgLogger(DbgLoggerLvl::EXCEPTION) < std::cout

enum DbgLoggerLvl : int { DEBUG = 0, INFO, WARNING, ERROR, EXCEPTION };

class DbgLogger {
 public:
  explicit DbgLogger(DbgLoggerLvl lvl) : lvl_(lvl) {}
  ~DbgLogger() = default;
  void operator<(std::ostream &os) const {
    char *dbg_log_path = getenv("OFFLINE_DBG_LOG");
    if (dbg_log_path != NULL) {
      FILE *fp;
      fp = freopen(dbg_log_path, "a", stdout);
      if (fp == nullptr) {
        std::cout << "ERROR: DbgLogger could not redirect all stdout to a file";
      }
    }
    os << std::endl;
    if (lvl_ == DbgLoggerLvl::EXCEPTION) {
      throw;
    }
  }
  static bool verbose;

 private:
  DbgLoggerLvl lvl_;
};
#endif  // OFFLINE_LOGGER_H_
