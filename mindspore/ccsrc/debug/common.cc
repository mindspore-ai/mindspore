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

#include "debug/common.h"

#include <memory>
#include <iomanip>
#include <optional>
#include "utils/ms_context.h"
#include "utils/system/env.h"
#include "utils/system/file_system.h"
#include "utils/log_adapter.h"

namespace mindspore {
std::optional<std::string> Common::GetRealPath(const std::string &input_path) {
  std::string out_path;
  auto path_split_pos = input_path.find_last_of('/');
  if (path_split_pos == std::string::npos) {
    path_split_pos = input_path.find_last_of('\\');
  }
  // get real path
  char real_path[PATH_MAX] = {0};
  if (path_split_pos != std::string::npos) {
    std::string prefix_path = input_path.substr(0, path_split_pos);
    if (prefix_path.length() >= PATH_MAX) {
      MS_LOG(ERROR) << "Prefix path is too longer!";
      return std::nullopt;
    }
    std::string last_path = input_path.substr(path_split_pos, input_path.length() - path_split_pos);
    auto ret = CreateNotExistDirs(prefix_path);
    if (!ret) {
      MS_LOG(ERROR) << "CreateNotExistDirs Failed!";
      return std::nullopt;
    }
#if defined(SYSTEM_ENV_POSIX)
    if (nullptr == realpath(prefix_path.c_str(), real_path)) {
      MS_LOG(ERROR) << "dir " << prefix_path << " does not exist.";
      return std::nullopt;
    }
#elif defined(SYSTEM_ENV_WINDOWS)
    if (nullptr == _fullpath(real_path, prefix_path.c_str(), PATH_MAX)) {
      MS_LOG(ERROR) << "dir " << prefix_path << " does not exist.";
      return std::nullopt;
    }
#else
    MS_LOG(EXCEPTION) << "Unsupported platform.";
#endif
    out_path = std::string(real_path) + last_path;
  }

  if (path_split_pos == std::string::npos) {
    if (input_path.length() >= PATH_MAX) {
      MS_LOG(ERROR) << "Prefix path is too longer!";
      return std::nullopt;
    }
#if defined(SYSTEM_ENV_POSIX)
    if (nullptr == realpath(input_path.c_str(), real_path)) {
      MS_LOG(ERROR) << "File " << input_path << " does not exist, it will be created.";
    }
#elif defined(SYSTEM_ENV_WINDOWS)
    if (nullptr == _fullpath(real_path, input_path.c_str(), PATH_MAX)) {
      MS_LOG(ERROR) << "File " << input_path << " does not exist, it will be created.";
    }
#else
    MS_LOG(EXCEPTION) << "Unsupported platform.";
#endif
    out_path = std::string(real_path);
  }
  return out_path;
}

bool Common::CreateNotExistDirs(const std::string &path) {
  std::shared_ptr<system::FileSystem> fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  char temp_path[PATH_MAX] = {0};
  if (path.length() > PATH_MAX) {
    MS_LOG(ERROR) << "Path lens is max than " << PATH_MAX;
    return false;
  }
  for (uint32_t i = 0; i < path.length(); i++) {
    temp_path[i] = path[i];
    if (temp_path[i] == '\\' || temp_path[i] == '/') {
      if (i != 0) {
        char tmp_char = temp_path[i];
        temp_path[i] = '\0';
        std::string path_handle(temp_path);
        if (!fs->FileExist(path_handle)) {
          MS_LOG(INFO) << "Dir " << path_handle << " does not exit, creating...";
          if (!fs->CreateDir(path_handle)) {
            MS_LOG(ERROR) << "Create " << path_handle << " dir error";
            return false;
          }
        }
        temp_path[i] = tmp_char;
      }
    }
  }

  if (!fs->FileExist(path)) {
    MS_LOG(INFO) << "Dir " << path << " does not exit, creating...";
    if (!fs->CreateDir(path)) {
      MS_LOG(ERROR) << "Create " << path << " dir error";
      return false;
    }
  }
  return true;
}

std::optional<std::string> Common::GetConfigFile(const std::string &env) {
  if (env.empty()) {
    MS_LOG(EXCEPTION) << "Invalid env";
  }
  auto config_path_str = std::getenv(env.c_str());
  if (config_path_str == nullptr) {
    MS_LOG(ERROR) << "Please export env:" << env;
    return {};
  }
  MS_LOG(INFO) << "Async Dump Getenv env:" << env << "=" << config_path_str;

  std::string dump_config_file(config_path_str);
  std::shared_ptr<system::FileSystem> fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  if (!fs->FileExist(dump_config_file)) {
    MS_LOG(ERROR) << dump_config_file << " not exist.";
    return {};
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

std::optional<std::string> Common::GetEnvConfigFile() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string env_config_path = context->get_param<std::string>(MS_CTX_ENV_CONFIG_PATH);
  if (env_config_path.empty()) {
    MS_LOG(INFO) << "The env_config_path is not set in context.";
    return {};
  }
  MS_LOG(INFO) << "Get env_config_path: " << env_config_path;

  std::string config_file(env_config_path);
  std::shared_ptr<system::FileSystem> fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  if (!fs->FileExist(config_file)) {
    MS_LOG(ERROR) << config_file << " not exist.";
    return {};
  }
  auto point_pos = config_file.find_last_of('.');
  if (point_pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "Invalid json file name:" << config_file;
  }
  return config_file;
}

bool Common::IsStrLengthValid(const std::string &str, const int &length_limit, const std::string &error_message) {
  const int len_str = str.length();
  if (len_str > length_limit) {
    MS_LOG(WARNING) << error_message << "The length is " << str.length() << ", exceeding the limit of " << length_limit
                    << ".";
    return false;
  }
  return true;
}

bool Common::IsEveryFilenameValid(const std::string &path, const int &length_limit, const std::string &error_message) {
  int left_pos = 0;
  if (path.empty()) {
    MS_LOG(WARNING) << error_message << "The path is empty.";
    return false;
  }
  int len_path = path.length();
  for (int i = 0; i < len_path; i++) {
    if (i != 0) {
      if (path[i] == '\\' || path[i] == '/') {
        int cur_len = i - left_pos;
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
    int cur_len = len_path - left_pos;
    if (cur_len > length_limit) {
      MS_LOG(WARNING) << error_message << "The name length of '" << path.substr(left_pos, cur_len) << "' is " << cur_len
                      << ". It is out of the limit which is " << length_limit << ".";
      return false;
    }
  }
  return true;
}

bool Common::IsPathValid(const std::string &path, const int &length_limit, const std::string &error_message) {
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

  if (!std::all_of(path.begin(), path.end(),
                   [](char c) { return ::isalpha(c) || ::isdigit(c) || c == '-' || c == '_' || c == '/'; })) {
    MS_LOG(WARNING) << err_msg << "The path only supports alphabets, digit or {'-', '_', '/'}, but got:" << path << ".";
    return false;
  }

  if (path[0] != '/') {
    MS_LOG(WARNING) << err_msg << "The path only supports absolute path and should start with '/'.";
    return false;
  }

  if (!IsEveryFilenameValid(path, MAX_OS_FILENAME_LENGTH, err_msg)) {
    return false;
  }
  return true;
}

bool Common::IsFilenameValid(const std::string &filename, const int &length_limit, const std::string &error_message) {
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

  if (!std::all_of(filename.begin(), filename.end(),
                   [](char c) { return ::isalpha(c) || ::isdigit(c) || c == '-' || c == '_' || c == '.'; })) {
    MS_LOG(WARNING) << err_msg << "The filename only supports alphabets, digit or {'-', '_', '.'}, but got:" << filename
                    << ".";
    return false;
  }
  return true;
}

std::string Common::AddId(const std::string &filename, const std::string &suffix) {
  static size_t g_id = 0;
  std::ostringstream s;
  auto i = filename.rfind(suffix);
  if (i >= filename.size()) {
    s << filename;
    s << "_" << std::setfill('0') << std::setw(4) << g_id;
  } else {
    s << filename.substr(0, i);
    s << "_" << std::setfill('0') << std::setw(4) << g_id;
    if (i + 1 < filename.size()) {
      s << filename.substr(i);
    }
  }
  g_id++;
  return s.str();
}
}  // namespace mindspore
