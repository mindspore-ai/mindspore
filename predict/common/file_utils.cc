/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "common/file_utils.h"
#include <climits>

namespace mindspore {
namespace predict {
char *ReadFile(const char *file, size_t *size) {
  if (file == nullptr) {
    MS_LOGE("file is nullptr");
    return nullptr;
  }
  MS_ASSERT(size != nullptr);
  std::ifstream ifs(RealPath(file));
  if (!ifs.good()) {
    MS_LOGE("file: %s is not exist", file);
    return nullptr;
  }

  if (!ifs.is_open()) {
    MS_LOGE("file: %s open failed", file);
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    MS_LOGE("malloc buf failed, file:%s", file);
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
    MS_LOGE("path is nullptr");
    return "";
  }
  if ((strlen(path)) >= PATH_MAX) {
    MS_LOGE("path is too long");
    return "";
  }

  std::shared_ptr<char> resolvedPath(new (std::nothrow) char[PATH_MAX]{0});
  if (resolvedPath == nullptr) {
    MS_LOGE("new resolvedPath failed");
    return "";
  }

  auto ret = realpath(path, resolvedPath.get());
  if (ret == nullptr) {
    MS_LOGE("realpath failed");
    return "";
  }
  return resolvedPath.get();
}
}  // namespace predict
}  // namespace mindspore
