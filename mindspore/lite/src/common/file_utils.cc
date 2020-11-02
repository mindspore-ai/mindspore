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

#include "src/common/file_utils.h"
#include <fcntl.h>
#include <cstdlib>
#include <climits>
#include <cmath>
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
#ifdef _WIN32
  char *real_path = _fullpath(resolvedPath.get(), path, 1024);
#else
  char *real_path = realpath(path, resolvedPath.get());
#endif
  if (real_path == nullptr || strlen(real_path) == 0) {
    MS_LOG(ERROR) << "file path is not valid : " << path;
    return "";
  }
  std::string res = resolvedPath.get();
  return res;
}

int CompareOutputData(const float *output_data, size_t output_size, const float *correct_data, size_t data_size) {
  if (output_size != data_size) {
    printf("compare failed, output_size %zu isn't equal to data_size %zu.\n", output_size, data_size);
    return 0;
  }
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

int CompareOutput(const float *output_data, size_t output_num, const std::string &file_path) {
  size_t ground_truth_size = 0;
  auto ground_truth = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &ground_truth_size));
  size_t ground_truth_num = ground_truth_size / sizeof(float);
  printf("ground truth num : %zu\n", ground_truth_num);
  int res = CompareOutputData(output_data, output_num, ground_truth, ground_truth_num);
  delete[] ground_truth;
  return res;
}
}  // namespace lite
}  // namespace mindspore
