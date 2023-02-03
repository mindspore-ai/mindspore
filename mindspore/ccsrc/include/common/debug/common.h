/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_COMMON_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_COMMON_H_

#include <string>
#include <optional>
#include "include/common/visible.h"
#include "include/common/utils/contract.h"
#include "utils/ms_context.h"
#include "include/common/utils/comm_manager.h"
#include "utils/system/base.h"

namespace mindspore {
static const int MAX_DIRECTORY_LENGTH = 1024;
static const int MAX_FILENAME_LENGTH = 128;
static const int MAX_OS_FILENAME_LENGTH = 255;
static const char kCOMPILER_CACHE_PATH[] = "MS_COMPILER_CACHE_PATH";

class COMMON_EXPORT Common {
 public:
  Common() = default;
  ~Common() = default;
  static bool NeedMapping(const std::string &origin_name);
  static std::string GetRandomStr();
  static std::string GetRandomStr(size_t str_len);
  static bool MappingName(const std::string &input_path, std::optional<std::string> *prefix_path,
                          std::optional<std::string> *origin_name, std::optional<std::string> *mapped_name);
  static std::optional<std::string> CreatePrefixPath(const std::string &input_path,
                                                     const bool support_relative_path = false);
  static std::optional<std::string> GetConfigFile(const std::string &env);
  static bool IsStrLengthValid(const std::string &str, size_t length_limit, const std::string &error_message = "");
  static bool IsPathValid(const std::string &path, size_t length_limit, const std::string &error_message = "");
  static bool IsFilenameValid(const std::string &filename, size_t length_limit, const std::string &error_message = "");

  static std::string AddId(const std::string &filename, const std::string &suffix);
  static bool SaveStringToFile(const std::string filename, const std::string string_info);
  static bool FileExists(const std::string &filepath);
  static bool CommonFuncForConfigPath(const std::string &default_path, const std::string &env_path,
                                      std::string *const value);
  static std::string GetCompilerCachePath();
  static std::string GetKernelMetaTempDir();
  static std::string GetUserDefineCachePath();
  static bool GetDebugTerminate();
  static bool GetDebugExitSuccess();
  static void DebugTerminate(bool val, bool exit_success);

  // Get time stamp since epoch in microseconds
  static uint64_t GetTimeStamp();

 private:
  static bool IsEveryFilenameValid(const std::string &path, size_t length_limit, const std::string &error_message);

  inline static bool debugger_terminate_ = false;
  inline static bool exit_success_ = false;
};

inline std::string GetSaveGraphsPathName(const std::string &file_name, const std::string &save_path = "") {
  std::string save_graphs_path;
  if (save_path.empty()) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    save_graphs_path = ms_context->get_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH);
    if (save_graphs_path.empty()) {
      save_graphs_path = ".";
    }
  } else {
    save_graphs_path = save_path;
  }
  if (IsStandAlone()) {
    return save_graphs_path + "/" + file_name;
  }
  return save_graphs_path + "/rank_" + std::to_string(GetRank()) + "/" + file_name;
}

inline std::string ErrnoToString(const int error_number) {
  std::ostringstream ret_info;
  ret_info << " Errno: " << error_number;
#if defined(__APPLE__)
  char err_info[MAX_FILENAME_LENGTH];
  (void)strerror_r(error_number, err_info, sizeof(err_info));
  ret_info << ", ErrInfo: " << err_info;
#elif defined(SYSTEM_ENV_POSIX)
  char err_info[MAX_FILENAME_LENGTH];
  char *ret = strerror_r(error_number, err_info, sizeof(err_info));
  if (ret != nullptr) {
    ret_info << ", ErrInfo: " << ret;
  }
#elif defined(SYSTEM_ENV_WINDOWS)
  char err_info[MAX_FILENAME_LENGTH];
  (void)strerror_s(err_info, sizeof(err_info), error_number);
  ret_info << ", ErrInfo: " << err_info;
#endif
  return ret_info.str();
}
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_COMMON_H_
