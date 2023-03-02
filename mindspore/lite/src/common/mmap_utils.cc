/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/common/mmap_utils.h"
#include "src/common/file_utils.h"
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#endif

namespace mindspore {
namespace lite {
void *ReadFileByMmap(const std::string &file, size_t *size) {
#if !defined(_WIN32) && !defined(_WIN64)
  auto real_path = RealPath(file.c_str());
  auto fd = open(real_path.c_str(), O_RDONLY);
  if (fd == -1) {
    MS_LOG(ERROR) << "Could not open " << file;
    return nullptr;
  }
  struct stat fd_stat;
  if (fstat(fd, &fd_stat) != 0) {
    MS_LOG(ERROR) << "Get fd stat error.";
    close(fd);
    return nullptr;
  }
  *size = fd_stat.st_size;
  auto mmap_buffers = mmap(nullptr, *size, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
  close(fd);
  if (mmap_buffers == MAP_FAILED) {
    MS_LOG(ERROR) << "Model mmap failed.";
    return nullptr;
  }
  return mmap_buffers;
#else
  MS_LOG(ERROR) << "Mmap is unsupported on windows.";
  return nullptr;
#endif
}

void UnmapMmapBuffer(void *buffer, size_t size) {
#if !defined(_WIN32) && !defined(_WIN64)
  (void)munmap(buffer, size);
#else
  MS_LOG(ERROR) << "Mmap is unsupported on windows.";
#endif
}
}  // namespace lite
}  // namespace mindspore
