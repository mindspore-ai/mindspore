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

#ifndef MINDSPORE_CORE_UTILS_SYSTEM_IO_HANDLE_H_
#define MINDSPORE_CORE_UTILS_SYSTEM_IO_HANDLE_H_

#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <utility>
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
using AsyncIOToken = size_t;
constexpr AsyncIOToken kInvalidAsyncIOToken = 0;

struct AsyncIOConf {
  size_t block_size;
  size_t queue_depth;
};

class AsyncIO {
 public:
  AsyncIO() = default;
  virtual ~AsyncIO() = default;
  virtual bool Init(const AsyncIOConf &conf) = 0;
  virtual bool Read(const std::string &file_name, void *data, size_t byte_num) = 0;
  virtual bool Write(const std::string &file_name, const void *data, size_t byte_num) = 0;
  virtual bool ReadAsync(const std::string &file_name, void *data, size_t byte_num, AsyncIOToken *token) = 0;
  virtual bool WriteAsync(const std::string &file_name, const void *data, size_t byte_num, AsyncIOToken *token) = 0;
  virtual bool Wait(AsyncIOToken token) = 0;
};

class BACKEND_EXPORT IOHandle {
 public:
  IOHandle() = default;
  ~IOHandle() = default;
  bool DeleteSwapFile(const std::string &file_name) const;
  bool CreateSwapFile(const std::string &file_name) const;
  void LoadAio(const std::string &aio_shared_lib_name, const std::string &instance_func_name);
  bool Read(const std::string &file_name, void *data, size_t byte_num);
  bool Write(const std::string &file_name, const void *data, size_t byte_num);
  bool ReadAsync(const std::string &file_name, void *data, size_t byte_num, AsyncIOToken *token);
  bool WriteAsync(const std::string &file_name, const void *data, size_t byte_num, AsyncIOToken *token);
  bool Wait(AsyncIOToken sync_token);

  void set_swap_path(const std::string &swap_path) { swap_path_ = swap_path; }

 private:
  std::string GetSwapFileWholeName(const std::string &file_name) const;

 private:
  std::string swap_path_;
  AsyncIO *aio_{nullptr};
};
using IOHandlePtr = std::shared_ptr<IOHandle>;
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_SYSTEM_IO_HANDLE_H_
