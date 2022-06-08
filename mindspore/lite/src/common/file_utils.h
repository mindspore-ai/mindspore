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

#ifndef MINDSPORE_LITE_SRC_COMMON_FILE_UTILS_H_
#define MINDSPORE_LITE_SRC_COMMON_FILE_UTILS_H_

#include <sys/stat.h>
#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
#define F_OK 0
using mode_t = int;
#include <io.h>
#else
#include <unistd.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <iostream>
#include <memory>
#include <fstream>
#include "src/common/utils.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {
constexpr const char MINDIR_POSTFIX[] = ".mindir";
#ifdef _WIN32
constexpr const char FILE_SEPARATOR[] = "\\";
#else
constexpr const char FILE_SEPARATOR[] = "/";
#endif
bool IsCharEndWith(const char *src, const char *end);

std::fstream *OpenFile(const std::string &file_path, std::ios_base::openmode open_mode = std::ios::in | std::ios::out);

char *ReadFileSegment(const std::string &file, int64_t offset, int64_t len);

char *ReadFile(const char *file, size_t *size);

std::string RealPath(const char *path);

int CreateOutputDir(std::string *file_path);

template <typename T>
void WriteToTxt(const std::string &file_path, void *data, size_t element_size) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "data is nullptr.";
    return;
  }
  std::ofstream out_file;
  out_file.open(file_path, std::ios::out);
  if (!out_file.is_open()) {
    MS_LOG(ERROR) << "file open failed";
    return;
  }
  auto real_data = reinterpret_cast<T *>(data);
  for (size_t i = 0; i < element_size; i++) {
    out_file << real_data[i] << " ";
  }
  out_file.close();
}

inline int WriteToBin(const std::string &file_path, void *data, const size_t size) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "data is nullptr.";
    return RET_ERROR;
  }
  std::ofstream out_file;
  out_file.open(file_path.c_str(), std::ios::binary);
  if (!out_file.good() || !out_file.is_open()) {
    return RET_ERROR;
  }
  out_file.write(reinterpret_cast<char *>(data), size);
  out_file.close();
  return RET_OK;
}

std::string GetDirectory(const std::string &path);

bool ParserPathAndModelName(const std::string &output_path, std::string *save_path, std::string *model_name);

static inline void ChangeFileMode(const std::string &file_name, mode_t mode) {
  if (access(file_name.c_str(), F_OK) == -1) {
    return;
  }
  if (chmod(file_name.c_str(), mode) != 0) {
    MS_LOG(WARNING) << "Change file `" << file_name << "` to mode " << std::oct << mode << " fail.";
  }
}
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_FILE_UTILS_H_
