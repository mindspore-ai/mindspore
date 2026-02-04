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

#ifndef MINDSPORE_CCSRC_RUNTIMR_HARDWARE_ABSTRACT_GSM_AIO_PLUGIN_H_
#define MINDSPORE_CCSRC_RUNTIMR_HARDWARE_ABSTRACT_GSM_AIO_PLUGIN_H_

#include <libaio.h>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "runtime/hardware_abstract/memory_manager/io_handle.h"

#ifndef AIO_EXPORT
#define AIO_EXPORT __attribute__((visibility("default")))
#endif

namespace mindspore {
namespace device {
using IocbPtr = iocb *;

struct AIOContext {
 public:
  AIOContext() = default;
  ~AIOContext() = default;
  bool Initialize(size_t queue_depth);
  bool Finalize();

  io_context_t io_context_{nullptr};
  std::vector<IocbPtr> iocbs_;
  std::vector<io_event> events_;
  size_t queue_depth_{0};
};

class IocbHelper {
 public:
  IocbHelper(bool read, int fd, void *buf_base, size_t block_size, size_t byte_num, void *data,
             const std::vector<IocbPtr> &iocbs);
  size_t PrepareNextBatch();

 private:
  bool read_;
  int fd_;
  uint8_t *cur_buf_;
  size_t block_size_;
  size_t total_size_;
  void *data_;
  const std::vector<IocbPtr> &iocbs_;
  size_t batch_size_;
  size_t cur_batch_{0};
  int64_t cur_offset_{0};
  size_t last_block_size_;
  size_t total_block_num_;
  size_t last_batch_size_;
  size_t total_batch_num_;
};

using AsyncIOToken = size_t;

class AioPlugin : public AsyncIO {
 public:
  AioPlugin() = default;
  ~AioPlugin() override;
  static AioPlugin &GetInstance();
  bool Init(const AsyncIOConf &conf) override;
  bool Read(const std::string &file_name, void *data, size_t byte_num) override;
  bool Write(const std::string &file_name, const void *data, size_t byte_num) override;
  bool ReadAsync(const std::string &file_name, void *data, size_t byte_num, AsyncIOToken *token) override;
  bool WriteAsync(const std::string &file_name, const void *data, size_t byte_num, AsyncIOToken *token) override;
  bool Wait(AsyncIOToken token) override;

 private:
  bool FileAIOSync(bool read, const std::string &file_name, void *buf_base, size_t byte_num);
  bool FileAIOAsync(bool read, const std::string &file_name, void *buf_base, size_t byte_num, AsyncIOToken *token);
  int OpenFile(const std::string &file_name);
  bool CloseFile(int fd);
  bool CheckAIOContextValid();
  bool SubmitIocb(size_t num);
  int GetEvents(size_t min, size_t max, std::vector<size_t *> *pending_events);
  size_t UpdateAsyncEvents();
  AsyncIOToken GetNextAsyncToken();

 private:
  bool inited_{false};
  AsyncIOConf config_{};
  std::shared_ptr<AIOContext> aio_context_;
  AsyncIOToken next_token_{};
  std::queue<AsyncIOToken> token_pool_;
  std::mutex aio_mutex_;
  size_t *pending_events_;
};

extern "C" AsyncIO *get_aio_instance();
}  // namespace device
}  // namespace mindspore
#undef LinuxAIO
#endif  // MINDSPORE_CCSRC_RUNTIMR_HARDWARE_ABSTRACT_GSM_AIO_PLUGIN_H_
