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
#include "runtime/device/gsm/aio_plugin.h"

#include <fcntl.h>
#include <libaio.h>
#include <unistd.h>
#include <limits>
#include <memory>
#include <iostream>

namespace mindspore {
namespace device {
namespace {
inline int SizeToInt(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<int>::max)())) {
    std::cout << "The size_t value(" << u << ") exceeds the maximum value of int.";
  }
  return static_cast<int>(u);
}

inline int64_t SizeToLong(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<int64_t>::max)())) {
    std::cout << "The size_t value(" << u << ") exceeds the maximum value of int64_t.";
  }
  return static_cast<int64_t>(u);
}

inline size_t IntToSize(int u) {
  if (u < 0) {
    std::cout << "The int value(" << u << ") is less than 0.";
    return SIZE_MAX;
  }
  return static_cast<size_t>(u);
}
}  // namespace

constexpr int kIOSuccess = 0;
constexpr size_t kMaxPendingAIO = 1024;
bool AIOContext::Initialize(size_t queue_depth) {
  if (queue_depth == 0) {
    return false;
  }
  queue_depth_ = queue_depth;
  if (io_setup(SizeToInt(queue_depth), &io_context_) != kIOSuccess) {
    return false;
  }
  for (size_t i = 0; i < queue_depth_; ++i) {
    auto cb = new iocb;
    iocbs_.emplace_back(cb);
  }
  events_.resize(queue_depth_);
  return true;
}

bool AIOContext::Finalize() {
  for (auto cb : iocbs_) {
    delete cb;
  }
  iocbs_.clear();
  if (io_context_ != nullptr && io_destroy(io_context_) != kIOSuccess) {
    return false;
  }
  return true;
}

IocbHelper::IocbHelper(bool read, int fd, void *buf_base, size_t block_size, size_t byte_num, void *data,
                       const std::vector<IocbPtr> &iocbs)
    : read_(read),
      fd_(fd),
      cur_buf_(reinterpret_cast<uint8_t *>(buf_base)),
      block_size_(block_size),
      total_size_(byte_num),
      data_(data),
      iocbs_(iocbs) {
  batch_size_ = iocbs_.size();
  if (block_size_ == 0 || batch_size_ == 0) {
    return;
  }
  last_block_size_ = total_size_ % block_size_;
  total_block_num_ = last_block_size_ == 0 ? total_size_ / block_size_ : total_size_ / block_size_ + 1;
  last_block_size_ = last_block_size_ == 0 ? block_size_ : last_block_size_;
  last_batch_size_ = total_block_num_ % batch_size_;
  total_batch_num_ = last_batch_size_ == 0 ? total_block_num_ / batch_size_ : total_block_num_ / batch_size_ + 1;
  last_batch_size_ = last_batch_size_ == 0 ? batch_size_ : last_batch_size_;
}

size_t IocbHelper::PrepareNextBatch() {
  if (cur_batch_ == total_batch_num_) {
    return 0;
  }
  const bool last_batch = cur_batch_ == total_batch_num_ - 1;
  size_t cur_batch_size = last_batch ? last_batch_size_ : batch_size_;
  for (size_t i = 0; i < cur_batch_size; ++i) {
    const bool last_block = (last_batch && i == cur_batch_size - 1);
    size_t cur_block_size = last_block ? last_block_size_ : block_size_;
    const auto &iocb = iocbs_[i];
    if (read_) {
      io_prep_pread(iocb, fd_, cur_buf_, cur_block_size, cur_offset_);
    } else {
      io_prep_pwrite(iocb, fd_, cur_buf_, cur_block_size, cur_offset_);
    }
    iocb->data = data_;
    cur_offset_ += SizeToLong(block_size_);
    cur_buf_ += block_size_;
  }
  cur_batch_ += 1;
  return cur_batch_size;
}

AioPlugin &AioPlugin::GetInstance() {
  static AioPlugin instance;
  return instance;
}

bool AioPlugin::Init(const AsyncIOConf &conf) {
  if (inited_) {
    return true;
  }
  config_ = conf;
  aio_context_ = std::make_shared<AIOContext>();
  if (!aio_context_->Initialize(SizeToInt(config_.queue_depth))) {
    aio_context_->Finalize();
    aio_context_ = nullptr;
    return false;
  }
  pending_events_ = new size_t[kMaxPendingAIO];
  for (size_t i = 0; i < kMaxPendingAIO; ++i) {
    pending_events_[i] = 0;
  }
  inited_ = true;
  return true;
}

AioPlugin::~AioPlugin() {
  if (aio_context_ != nullptr) {
    aio_context_->Finalize();
  }
  if (pending_events_ != nullptr) {
    delete[] pending_events_;
    pending_events_ = nullptr;
  }
}

int AioPlugin::OpenFile(const std::string &file_name) {
  return open(file_name.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644);
}

bool AioPlugin::CloseFile(int fd) { return (close(fd) == 0); }

bool AioPlugin::CheckAIOContextValid() { return (aio_context_ != nullptr && aio_context_->io_context_ != nullptr); }

bool AioPlugin::SubmitIocb(size_t num) {
  const auto submit_ret = io_submit(aio_context_->io_context_, SizeToLong(num), aio_context_->iocbs_.data());
  return (submit_ret >= 0 && IntToSize(submit_ret) == num);
}

int AioPlugin::GetEvents(size_t min, size_t max, std::vector<size_t *> *pending_events) {
  if (max > aio_context_->events_.size()) {
    aio_context_->events_.resize(max);
  }
  const auto get_events_ret =
    io_getevents(aio_context_->io_context_, SizeToLong(min), SizeToLong(max), aio_context_->events_.data(), nullptr);
  if (get_events_ret < 0) {
    return get_events_ret;
  }
  const size_t events_num = IntToSize(get_events_ret);
  if (events_num < min) {
    return get_events_ret;
  }
  for (size_t i = 0; i < events_num; ++i) {
    if (aio_context_->events_[i].data != nullptr) {
      pending_events->emplace_back(reinterpret_cast<size_t *>(aio_context_->events_[i].data));
    }
  }
  return get_events_ret;
}

bool AioPlugin::FileAIOSync(bool read, const std::string &file_name, void *buf_base, size_t byte_num) {
  if (!CheckAIOContextValid()) {
    return false;
  }
  const auto fd = OpenFile(file_name);
  if (fd < 0) {
    return false;
  }
  std::lock_guard<std::mutex> lock(aio_mutex_);
  IocbHelper iocb_helper(read, fd, buf_base, config_.block_size, byte_num, nullptr, aio_context_->iocbs_);
  auto batch_size = iocb_helper.PrepareNextBatch();
  while (batch_size != 0) {
    if (!SubmitIocb(batch_size)) {
      return false;
    }
    std::vector<size_t *> pending_events;
    const auto get_events_ret = GetEvents(batch_size, batch_size, &pending_events);
    if (get_events_ret < 0 || IntToSize(get_events_ret) < batch_size) {
      return false;
    }
    batch_size = iocb_helper.PrepareNextBatch();
  }
  return CloseFile(fd);
}

size_t AioPlugin::UpdateAsyncEvents() {
  size_t update_num;
  size_t total_update_num = 0;
  do {
    std::vector<size_t *> pending_events;
    update_num = GetEvents(0, SizeToLong(config_.queue_depth), &pending_events);
    total_update_num += update_num;
    for (const auto ptr : pending_events) {
      *ptr -= 1;
    }
  } while (update_num == config_.queue_depth);
  return total_update_num;
}

bool AioPlugin::FileAIOAsync(bool read, const std::string &file_name, void *buf_base, size_t byte_num,
                             AsyncIOToken *token) {
  if (!CheckAIOContextValid()) {
    return false;
  }
  *token = GetNextAsyncToken();
  const auto fd = OpenFile(file_name);
  if (fd < 0) {
    return false;
  }
  std::lock_guard<std::mutex> lock(aio_mutex_);
  (void)UpdateAsyncEvents();
  auto ptr = pending_events_ + *token;
  IocbHelper iocb_helper(read, fd, buf_base, config_.block_size, byte_num, ptr, aio_context_->iocbs_);
  auto batch_size = iocb_helper.PrepareNextBatch();
  while (batch_size != 0) {
    if (!SubmitIocb(batch_size)) {
      return false;
    }
    pending_events_[*token] += batch_size;
    (void)UpdateAsyncEvents();
    batch_size = iocb_helper.PrepareNextBatch();
  }
  return true;
}

bool AioPlugin::Wait(AsyncIOToken token) {
  while (pending_events_[token] != 0) {
    std::vector<size_t *> pending_events;
    const auto ret = GetEvents(0, SizeToLong(config_.queue_depth), &pending_events);
    if (ret < 0) {
      return false;
    }
    std::lock_guard<std::mutex> lock(aio_mutex_);
    for (const auto ptr : pending_events) {
      *ptr -= 1;
    }
  }
  token_pool_.push(token);
  return true;
}

bool AioPlugin::Read(const std::string &file_name, void *data, size_t byte_num) {
  return FileAIOSync(true, file_name, data, byte_num);
}

bool AioPlugin::Write(const std::string &file_name, const void *data, size_t byte_num) {
  return FileAIOSync(false, file_name, const_cast<void *>(data), byte_num);
}

bool AioPlugin::ReadAsync(const std::string &file_name, void *data, size_t byte_num, AsyncIOToken *token) {
  return FileAIOAsync(true, file_name, data, byte_num, token);
}

bool AioPlugin::WriteAsync(const std::string &file_name, const void *data, size_t byte_num, AsyncIOToken *token) {
  return FileAIOAsync(false, file_name, const_cast<void *>(data), byte_num, token);
}

AsyncIOToken AioPlugin::GetNextAsyncToken() {
  if (!token_pool_.empty()) {
    const auto next_token = token_pool_.front();
    token_pool_.pop();
    return next_token;
  }
  AsyncIOToken next_token = ++next_token_;
  if (next_token >= kMaxPendingAIO) {
    while (token_pool_.empty()) {
      (void)UpdateAsyncEvents();
    }
    next_token = token_pool_.front();
    token_pool_.pop();
    --next_token_;
    return next_token;
  }
  return next_token;
}

AsyncIO *get_aio_instance() { return &AioPlugin::GetInstance(); }
}  // namespace device
}  // namespace mindspore
