/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SHARED_MEM_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SHARED_MEM_H_

#include <fcntl.h>

#include <string>

#include "minddata/dataset/util/status.h"

namespace mindspore::dataset {
std::string GenerateShmName();

class SharedMem {
 public:
  SharedMem() = delete;

  explicit SharedMem(const std::string &name, bool create, int fd, size_t size);

  ~SharedMem();

  void *Buf();

  std::string Name() const;

  int32_t Fd() const;

  size_t Size() const;

  void Close();

 private:
  std::string name_;
  bool create_;
  int32_t fd_;
  size_t size_;
  void *buf_;
};
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SHARED_MEM_H_
