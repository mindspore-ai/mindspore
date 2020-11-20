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
#include "securec/include/securec.h"

namespace mindspore {
namespace lite {
char *ReadFile(const char *file, size_t *size) {
  if (file == nullptr) {
    MS_LOG(ERROR) << "file is nullptr";
    return nullptr;
  }
  MS_ASSERT(size != nullptr);
  std::string real_path = RealPath(file);
  std::ifstream ifs(real_path);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << real_path << " is not exist";
    return nullptr;
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << real_path << " open failed";
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char[]> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "malloc buf failed, file: " << real_path;
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
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "new resolved_path failed";
    return "";
  }
#ifdef _WIN32
  char *real_path = _fullpath(resolved_path.get(), path, 1024);
#else
  char *real_path = realpath(path, resolved_path.get());
#endif
  if (real_path == nullptr || strlen(real_path) == 0) {
    MS_LOG(ERROR) << "file path is not valid : " << path;
    return "";
  }
  std::string res = resolved_path.get();
  return res;
}
}  // namespace lite
}  // namespace mindspore
