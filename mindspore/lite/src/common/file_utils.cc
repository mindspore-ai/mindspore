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

#include <stdlib.h>
#include <fcntl.h>
#include <climits>
#include <cmath>
#include "src/common/file_utils.h"
#include "securec/include/securec.h"

namespace mindspore {
namespace lite {
#define MAX_FILENAME_LEN 1024
char *ReadFile(const char *file, size_t *size) {
  if (file == nullptr) {
    MS_LOG(ERROR) << "file is nullptr";
    return nullptr;
  }
  MS_ASSERT(size != nullptr);
  std::string realPath = RealPath(file);
  std::ifstream ifs(realPath);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << realPath << " is not exist";
    return nullptr;
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << realPath << " open failed";
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char[]> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "malloc buf failed, file: " << realPath;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), *size);
  ifs.close();

  return buf.release();
}

std::string RealPath(const char *path) {
  if (path == nullptr) {
    MS_LOG(ERROR) << "path is nullptr";
    return "";
  }
  if ((strlen(path)) >= PATH_MAX) {
    MS_LOG(ERROR) << "path is too long";
    return "";
  }
  auto resolvedPath = std::make_unique<char[]>(PATH_MAX);
  if (resolvedPath == nullptr) {
    MS_LOG(ERROR) << "new resolvedPath failed";
    return "";
  }
  char *real_path = realpath(path, resolvedPath.get());
  if (real_path == nullptr || strlen(real_path) == 0) {
    MS_LOG(ERROR) << "Proto file path is not valid";
    return "";
  }
  std::string res = resolvedPath.get();
  return res;
}

int WriteToBin(const std::string &file_path, void *data, size_t size) {
  std::ofstream out_file;

  out_file.open(file_path.c_str(), std::ios::binary);
  if (!out_file.good()) {
    return -1;
  }

  if (!out_file.is_open()) {
    out_file.close();
    return -1;
  }
  out_file.write(reinterpret_cast<char *>(data), size);
  return 0;
}

int CompareOutputData(float *output_data, float *correct_data, int data_size) {
  float error = 0;
  for (size_t i = 0; i < data_size; i++) {
    float abs = fabs(output_data[i] - correct_data[i]);
    if (abs > 0.00001) {
      error += abs;
    }
  }
  error /= data_size;
  if (error > 0.0001) {
    printf("has accuracy error!\n");
    printf("%f\n", error);
    return 1;
  }
  return 0;
}

void CompareOutput(float *output_data, std::string file_path) {
  size_t output_size;
  auto ground_truth = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &output_size));
  size_t output_num = output_size / sizeof(float);
  printf("output num : %zu\n", output_num);
  CompareOutputData(output_data, ground_truth, output_num);
}

// std::string GetAndroidPackageName() {
//  static std::string packageName;
//
//  if (!packageName.empty()) {
//    return packageName;
//  }
//
//  char cmdline[MAX_FILENAME_LEN] = {0};
//  int fd = open("/proc/self/cmdline", O_RDONLY);
//
//  if (fd >= 0) {
//    char ch;
//    int i = 0;
//    while (read(fd, &ch, sizeof(ch)) > 0 && !isspace(ch)) {
//      if (':' == ch) {
//        break;
//      }
//
//      if (('/' == ch) || ('\\' == ch)) {
//        (void)memset(cmdline, 0, sizeof(cmdline));
//        i = 0;
//      } else {
//        cmdline[i] = ch;
//        i++;
//      }
//    }
//    close(fd);
//  }
//  packageName = std::string(cmdline);
//  return packageName;
//}

// std::string GetAndroidPackagePath() {
//  std::string packageName = GetAndroidPackageName();
//  if (packageName.empty()) {
//    return "./";
//  }
//  return "/data/data/" + packageName + '/';
//}

}  // namespace lite
}  // namespace mindspore
