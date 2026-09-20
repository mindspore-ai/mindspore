/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_PATH_CHECK_H_
#define MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_PATH_CHECK_H_

#include <string>
#include "utils/file_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
inline bool IsCanonicalPathUnderDir(const std::string &file_path, const std::string &dir_path) {
  const auto real_dir = FileUtils::GetRealPath(dir_path.c_str());
  if (!real_dir.has_value()) {
    return false;
  }
  std::string dir = real_dir.value();
  if (!dir.empty() && dir.back() != '/') {
    dir += '/';
  }
  return file_path.rfind(dir, 0) == 0;
}

// Framework-generated custom libraries: AKG outputs akg_kernel_meta/, Python _compile_aot
// writes to ./custom_kernel_meta/ or MS_COMPILER_CACHE_PATH (and rank_{id}/ when distributed).
inline bool IsPathUnderCustomAotGeneratedDir(const std::string &file_path) {
  auto cache = common::GetEnv("MS_COMPILER_CACHE_PATH");
  if (cache.empty()) {
    return IsCanonicalPathUnderDir(file_path, "./akg_kernel_meta/") ||
           IsCanonicalPathUnderDir(file_path, "./custom_kernel_meta/");
  }
  if (IsCanonicalPathUnderDir(file_path, cache)) {
    return true;
  }
  auto rank_id = common::GetEnv("RANK_ID");
  if (rank_id.empty()) {
    rank_id = "0";
  }
  if (cache.back() != '/') {
    cache += '/';
  }
  return IsCanonicalPathUnderDir(file_path, cache + "rank_" + rank_id + "/") ||
         IsCanonicalPathUnderDir(file_path, cache + "akg_kernel_meta/");
}

// Check MS_CUSTOM_AOT_WHITE_LIST before dlopen. The env is opt-in to keep compatibility.
// from_model_file should be true when the primitive was restored from a model file (is_load).
inline void CheckCustomAotWhiteList(const std::string &file_path, const std::string &log_prefix, bool from_model_file) {
  constexpr auto kWhiteList = "MS_CUSTOM_AOT_WHITE_LIST";
  const auto white_list_env = common::GetEnv(kWhiteList);
  if (white_list_env.empty()) {
    static bool print_warning_once = true;
    static bool print_info_once = true;
    if (from_model_file && print_warning_once) {
      MS_LOG(WARNING) << log_prefix
                      << ", no white list is set for the custom AOT operator library. Loading a library from an"
                      << " untrusted path carried by a model file may cause security problems. Set the legal"
                      << " directory of the library in the environment variable MS_CUSTOM_AOT_WHITE_LIST.";
      print_warning_once = false;
    } else if (!from_model_file && print_info_once) {
      MS_LOG(INFO) << log_prefix
                   << ", no white list is set for the custom AOT operator library. Set the legal directory of"
                   << " the library in the environment variable MS_CUSTOM_AOT_WHITE_LIST.";
      print_info_once = false;
    }
    return;
  }
  auto white_list = FileUtils::GetRealPath(white_list_env.c_str());
  if (!white_list.has_value()) {
    MS_LOG(EXCEPTION) << "Illegal white list path set in MS_CUSTOM_AOT_WHITE_LIST: " << white_list_env;
  }
  std::string wl = white_list.value();
  if (!wl.empty() && wl.back() != '/') {
    wl += '/';
  }
  if (file_path.rfind(wl, 0) != 0 && !IsPathUnderCustomAotGeneratedDir(file_path)) {
    MS_LOG(EXCEPTION) << log_prefix
                      << ", the file is not place in the legal path file defined by MS_CUSTOM_AOT_WHITE_LIST: "
                      << white_list.value() << ". The file path is: " << file_path;
  }
}
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_PATH_CHECK_H_
