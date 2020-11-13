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

#ifndef MINDSPORE_LITE_COMMON_FILE_UTILS_H_
#define MINDSPORE_LITE_COMMON_FILE_UTILS_H_

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
char *ReadFile(const char *file, size_t *size);

std::string RealPath(const char *path);

template <typename T>
void WriteToTxt(const std::string &file_path, void *data, size_t element_size) {
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

inline int WriteToBin(const std::string &file_path, void *data, size_t size) {
  std::ofstream out_file;
  out_file.open(file_path.c_str(), std::ios::binary);
  if (!out_file.good() || !out_file.is_open()) {
    return -1;
  }
  out_file.write(reinterpret_cast<char *>(data), size);
  return 0;
}

std::string GetAndroidPackageName();
std::string GetAndroidPackagePath();
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_COMMON_FILE_UTILS_H_
