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

#include "include/common/debug/common.h"

#include <memory>
#include <iomanip>
#include <optional>
#include <fstream>
#include <string>
#include <random>
#include "utils/system/env.h"
#include "utils/system/file_system.h"
#include "utils/log_adapter.h"
#include "utils/file_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore {
bool Common::NeedMapping(const std::string &origin_name) {
#if defined(SYSTEM_ENV_POSIX)
  if (origin_name.length() > NAME_MAX) {
    return true;
  } else {
    return false;
  }
#endif
  return false;
}

std::string Common::GetRandomStr() {
  std::string npy_suffix = ".npy";
#ifndef _MSC_VER
  unsigned int seed = static_cast<unsigned int>(GetTimeStamp());
  std::string random_name = std::to_string(rand_r(&seed)) + npy_suffix;
#else
  std::random_device rd;
  std::mt19937 gen(rd());
  constexpr int min_value = 0;
  std::uniform_int_distribution<> distrib(min_value);
  std::string random_name = std::to_string(distrib(gen)) + npy_suffix;
#endif
  return random_name;
}

bool Common::MappingName(const std::string &input_path, std::optional<std::string> *prefix_path,
                         std::optional<std::string> *origin_name, std::optional<std::string> *mapped_name) {
  FileUtils::SplitDirAndFileName(input_path, prefix_path, origin_name);
  if (!prefix_path->has_value() || !origin_name->has_value()) {
    MS_LOG(ERROR) << "Cannot get prefix_path or file_name from: " << input_path;
    return false;
  }
  auto origin_name_str = origin_name->value();
  if (!NeedMapping(origin_name_str)) {
    MS_EXCEPTION_IF_NULL(mapped_name);
    *mapped_name = origin_name_str;
    return false;
  }
  MS_LOG(INFO) << "The file name: " << input_path << " is too long, change it to a random string.";
  // name, max_loop is used to avoid the foreverloop.
  size_t loop_count = 0;
  size_t max_loop = 2;
  std::string file_path;
  while (loop_count < max_loop) {
    loop_count += 1;
    *mapped_name = GetRandomStr();
    file_path = prefix_path->value() + "/" + mapped_name->value();
    if (FileExists(file_path)) {
      continue;
    }
    return true;
  }
  MS_LOG(WARNING) << "Random strings are repeated, which may lead to coverring the old data! File path is: "
                  << file_path;
  return true;
}

std::optional<std::string> Common::CreatePrefixPath(const std::string &input_path, const bool support_relative_path) {
  std::optional<std::string> prefix_path;
  std::optional<std::string> file_name;
  FileUtils::SplitDirAndFileName(input_path, &prefix_path, &file_name);
  if (!file_name.has_value()) {
    MS_LOG(ERROR) << "Cannot get file_name from: " << input_path;
    return std::nullopt;
  }
  auto file_name_str = file_name.value();
#if defined(SYSTEM_ENV_POSIX)
  if (file_name_str.length() > NAME_MAX) {
    MS_LOG(ERROR) << "The length of file name: " << file_name_str.length() << " exceeds limit: " << NAME_MAX;
    return std::nullopt;
  }
#endif

  std::string prefix_path_str;
  if (prefix_path.has_value()) {
    auto create_prefix_path = FileUtils::CreateNotExistDirs(prefix_path.value(), support_relative_path);
    if (!create_prefix_path.has_value()) {
      return std::nullopt;
    }
    prefix_path_str = create_prefix_path.value();
  } else {
    auto pwd_path = FileUtils::GetRealPath("./");
    if (!pwd_path.has_value()) {
      MS_LOG(ERROR) << "Cannot get pwd path";
      return std::nullopt;
    }
    prefix_path_str = pwd_path.value();
  }
  return std::string(prefix_path_str + "/" + file_name_str);
}

bool Common::CommonFuncForConfigPath(const std::string &default_path, const std::string &env_path,
                                     std::string *const value) {
  MS_EXCEPTION_IF_NULL(value);
  value->clear();
  if (!env_path.empty()) {
    char real_path[PATH_MAX] = {0};
#if defined(SYSTEM_ENV_WINDOWS)
    if (_fullpath(real_path, common::SafeCStr(env_path), PATH_MAX) == nullptr) {
      MS_LOG(ERROR) << "The dir " << env_path << " does not exist.";
      return false;
    }
    *value = real_path;
    return true;
#else

    if (realpath(env_path.c_str(), real_path)) {
      *value = real_path;
      return true;
    }
    MS_LOG(ERROR) << "Invalid env path, path : " << env_path;
    return false;
#endif
  }
  *value = default_path;
  return true;
}

std::optional<std::string> Common::GetConfigFile(const std::string &env) {
  if (env.empty()) {
    MS_LOG(EXCEPTION) << "Invalid env";
  }
  auto config_path_str = common::GetEnv(env);
  if (config_path_str.empty()) {
    MS_LOG(ERROR) << "Please export env:" << env;
    return std::nullopt;
  }
  MS_LOG(INFO) << "Async Dump Getenv env:" << env << "=" << config_path_str;

  auto real_path = FileUtils::GetRealPath(common::SafeCStr(config_path_str));
  if (!real_path.has_value()) {
    MS_LOG(ERROR) << "Can't get real_path";
    return std::nullopt;
  }
  std::string dump_config_file = real_path.value();
  std::shared_ptr<system::FileSystem> fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  if (!fs->FileExist(dump_config_file)) {
    MS_LOG(ERROR) << dump_config_file << " not exist.";
    return std::nullopt;
  }
  auto point_pos = dump_config_file.find_last_of('.');
  if (point_pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "Invalid json file name:" << dump_config_file;
  }
  auto suffix = dump_config_file.substr(point_pos + 1);
  if (suffix != "json") {
    MS_LOG(EXCEPTION) << "[DataDump] dump config file suffix only supports json! But got:." << suffix;
  }
  return dump_config_file;
}

bool Common::IsStrLengthValid(const std::string &str, size_t length_limit, const std::string &error_message) {
  auto len_str = str.length();
  if (len_str > length_limit) {
    MS_LOG(ERROR) << error_message << "The length is " << str.length() << ", exceeding the limit of " << length_limit
                  << ".";
    return false;
  }
  return true;
}

bool Common::IsEveryFilenameValid(const std::string &path, size_t length_limit, const std::string &error_message) {
  if (path.empty()) {
    MS_LOG(WARNING) << error_message << "The path is empty.";
    return false;
  }
  size_t len_path = path.length();
  size_t left_pos = 0;
  for (size_t i = 0; i < len_path; i++) {
    if (i != 0) {
      if (path[i] == '\\' || path[i] == '/') {
        auto cur_len = i - left_pos;
        if (cur_len > length_limit) {
          MS_LOG(WARNING) << error_message << "The name length of '" << path.substr(left_pos, cur_len) << "' is "
                          << cur_len << ". It is out of the limit which is " << length_limit << ".";
          return false;
        }
        left_pos = i + 1;
      }
    }
  }
  if (!(path[len_path - 1] == '\\' || path[len_path - 1] == '/')) {
    auto cur_len = len_path - left_pos;
    if (cur_len > length_limit) {
      MS_LOG(WARNING) << error_message << "The name length of '" << path.substr(left_pos, cur_len) << "' is " << cur_len
                      << ". It is out of the limit which is " << length_limit << ".";
      return false;
    }
  }
  return true;
}

bool Common::IsPathValid(const std::string &path, size_t length_limit, const std::string &error_message) {
  std::string err_msg = "Detail: ";
  if (!error_message.empty()) {
    err_msg = error_message + " " + err_msg;
  }

  if (path.empty()) {
    MS_LOG(WARNING) << err_msg << "The path is empty.";
    return false;
  }

  if (!IsStrLengthValid(path, length_limit, err_msg)) {
    return false;
  }

  if (!std::all_of(path.begin(), path.end(), [](char c) {
        return ::isalpha(c) || ::isdigit(c) || c == '-' || c == '_' || c == '.' || c == '/';
      })) {
    MS_LOG(ERROR) << err_msg << "The path only supports alphabets, digit or {'-', '_', '.', '/'}, but got:" << path
                  << ".";
    return false;
  }

  if (path[0] != '/') {
    MS_LOG(ERROR) << err_msg << "The path only supports absolute path and should start with '/'.";
    return false;
  }

  if (!IsEveryFilenameValid(path, MAX_OS_FILENAME_LENGTH, err_msg)) {
    return false;
  }
  return true;
}

bool Common::IsFilenameValid(const std::string &filename, size_t length_limit, const std::string &error_message) {
  std::string err_msg = "Detail: ";
  if (!error_message.empty()) {
    err_msg = error_message + " " + err_msg;
  }

  if (filename.empty()) {
    MS_LOG(WARNING) << err_msg << "The filename is empty.";
    return false;
  }

  if (!IsStrLengthValid(filename, length_limit, err_msg)) {
    return false;
  }
  auto func = [](char c) { return ::isalpha(c) || ::isdigit(c) || c == '-' || c == '_' || c == '.'; };
  if (!std::all_of(filename.begin(), filename.end(), func)) {
    MS_LOG(ERROR) << err_msg << "The filename only supports alphabets, digit or {'-', '_', '.'}, but got:" << filename
                  << ".";
    return false;
  }
  return true;
}

std::string Common::AddId(const std::string &filename, const std::string &suffix) {
  static size_t g_id = 0;
  std::ostringstream s;
  auto i = filename.rfind(suffix);
  const int spaces = 4;
  if (i >= filename.size()) {
    s << filename;
    s << "_" << std::setfill('0') << std::setw(spaces) << g_id;
  } else {
    s << filename.substr(0, i);
    s << "_" << std::setfill('0') << std::setw(spaces) << g_id;
    if (i + 1 < filename.size()) {
      s << filename.substr(i);
    }
  }
  g_id++;
  return s.str();
}

bool Common::SaveStringToFile(const std::string filename, const std::string string_info) {
  if (filename.size() >= PATH_MAX) {
    MS_LOG(ERROR) << "File path " << filename << " is too long.";
    return false;
  }
  auto real_path = CreatePrefixPath(filename);
  if (!real_path.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. path=" << filename;
    return false;
  }

  ChangeFileMode(real_path.value(), S_IRWXU);
  std::ofstream ofs;
  ofs.open(real_path.value());

  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << real_path.value() << "' failed!" << ErrnoToString(errno);
    return false;
  }
  ofs << string_info << std::endl;
  ofs.close();
  // set file mode to read only by user
  ChangeFileMode(real_path.value(), S_IRUSR);
  return true;
}

bool Common::FileExists(const std::string &filepath) {
  std::ifstream f(filepath);
  bool cache_file_existed = f.good();
  f.close();
  return cache_file_existed;
}

std::string Common::GetUserDefineCachePath() {
  static std::string config_path = "";
  if (config_path != "") {
    return config_path;
  }
  const char *value = ::getenv(kCOMPILER_CACHE_PATH);
  if (value == nullptr) {
    config_path = "./";
  } else {
    config_path = std::string(value);
    (void)FileUtils::CreateNotExistDirs(config_path);
    if (config_path[config_path.length() - 1] != '/') {
      config_path += "/";
    }
  }
  return config_path;
}

std::string Common::GetCompilerCachePath() {
  static const std::string user_defined_path = GetUserDefineCachePath();
  std::string rank_id_str = common::GetEnv(kRankID);
  if (rank_id_str.empty()) {
    MS_LOG(DEBUG) << "Environment variable 'RANK_ID' is empty, using the default value: 0";
    rank_id_str = "0";
  }
  const std::string compile_cache_dir = user_defined_path + "rank_" + rank_id_str + "/";
  return compile_cache_dir;
}

std::string Common::GetKernelMetaTempDir() {
  auto cache_path = GetUserDefineCachePath();
  std::string rank_id_str = common::GetEnv(kRankID);
  if (rank_id_str.empty()) {
    MS_LOG(DEBUG) << "Environment variable 'RANK_ID' is empty, using the default value: 0";
    rank_id_str = "0";
  }
  auto kernel_meta_temp_dir = cache_path + +"rank_" + rank_id_str + "/kernel_meta_temp_dir/";
  (void)FileUtils::CreateNotExistDirs(kernel_meta_temp_dir);
  return kernel_meta_temp_dir;
}

bool Common::GetDebugTerminate() { return debugger_terminate_; }

bool Common::GetDebugExitSuccess() { return exit_success_; }

void Common::DebugTerminate(bool val, bool exit_success) {
  debugger_terminate_ = val;
  exit_success_ = exit_success;
}

uint64_t Common::GetTimeStamp() {
  auto cur_sys_time = std::chrono::system_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(cur_sys_time.time_since_epoch()).count();
  return static_cast<uint64_t>(timestamp);
}

struct GlogLogDirRegister {
  GlogLogDirRegister() {
    const char *logtostderr = std::getenv("GLOG_logtostderr");
    const char *log_dir = std::getenv("GLOG_log_dir");
    if (logtostderr != nullptr && log_dir != nullptr) {
      std::string logtostderr_str = std::string(logtostderr);
      std::string log_dir_str = std::string(log_dir);
      if (logtostderr_str != "0") {
        return;
      }
#ifndef BUILD_LITE
      const char *rank_id = std::getenv("RANK_ID");
      const char *gpu_rank_id = std::getenv("OMPI_COMM_WORLD_RANK");
      std::string rank = "0";
      if (rank_id != nullptr) {
        rank = std::string(rank_id);
      } else if (gpu_rank_id != nullptr) {
        rank = std::string(gpu_rank_id);
      }
      log_dir_str += "/rank_" + rank + "/logs/";
#endif
      auto real_log_dir_str = Common::CreatePrefixPath(log_dir_str, true);
      // While 'GLOG_logtostderr' = 0, logs output to files. 'GLOG_log_dir' must be specified as the path of log files.
      // Here can not throw exception and use python to catch, because the PYBIND11_MODULE is not yet been initialed.
      if (!real_log_dir_str.has_value()) {
        MS_LOG(ERROR) << "The path of log files, which set by 'GLOG_log_dir', is invalid.";
        exit(EXIT_FAILURE);
      }
#ifndef BUILD_LITE
      if (rank_id != nullptr && gpu_rank_id != nullptr && std::string(rank_id) != std::string(gpu_rank_id)) {
        MS_LOG(WARNING) << "Environment variables RANK_ID and OMPI_COMM_WORLD_RANK set by different values: RANK_ID="
                        << std::string(rank_id) << ", OMPI_COMM_WORLD_RANK=" << std::string(gpu_rank_id)
                        << ". We will use RANK_ID to get rank id by default.";
      }
#endif
    }
  }
} _glog_log_dir_register;
}  // namespace mindspore
