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

#include "debug/common.h"

#include <memory>
#include <optional>
#include "utils/system/env.h"
#include "utils/system/file_system.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

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

    if (nullptr == realpath(prefix_path.c_str(), real_path)) {
      MS_LOG(ERROR) << "dir " << prefix_path << " does not exit.";
      return std::nullopt;
    }
    out_path = std::string(real_path) + last_path;
  }

  if (path_split_pos == std::string::npos) {
    if (input_path.length() >= PATH_MAX) {
      MS_LOG(ERROR) << "Prefix path is too longer!";
      return std::nullopt;
    }
    if (nullptr == realpath(input_path.c_str(), real_path)) {
      MS_LOG(ERROR) << "File " << input_path << " does not exit, it will be created.";
    }
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
  auto suffix = dump_config_file.substr(dump_config_file.find_last_of('.') + 1);
  if (suffix != "json") {
    MS_LOG(EXCEPTION) << "[DataDump] dump config file suffix only support json! But got:." << suffix;
  }
  return dump_config_file;
}
}  // namespace mindspore
