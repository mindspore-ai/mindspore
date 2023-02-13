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

#include "runtime/device/gsm/io_handle.h"
#include <memory>
#include "utils/system/env.h"

namespace mindspore {
namespace device {
constexpr size_t kFileHeadOffset = 0;

bool IOHandle::Read(const std::string &file_name, void *data, size_t byte_num) {
  if (aio_ != nullptr) {
    return aio_->Read(file_name, data, byte_num);
  }
  const auto &fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  auto file = fs->CreateWriteFile(GetSwapFileWholeName(file_name));
  MS_EXCEPTION_IF_NULL(file);
  return file->PRead(data, byte_num, kFileHeadOffset);
}

bool IOHandle::Write(const std::string &file_name, const void *data, size_t byte_num) {
  if (aio_ != nullptr) {
    return aio_->Write(file_name, data, byte_num);
  }
  const auto &fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  auto file = fs->CreateWriteFile(GetSwapFileWholeName(file_name));
  MS_EXCEPTION_IF_NULL(file);
  return file->PWrite(data, byte_num, kFileHeadOffset);
}

bool IOHandle::ReadAsync(const std::string &file_name, void *data, size_t byte_num, AsyncIOToken *token) {
  if (aio_ != nullptr) {
    return aio_->ReadAsync(file_name, data, byte_num, token);
  }
  const auto &fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  auto file = fs->CreateWriteFile(GetSwapFileWholeName(file_name));
  MS_EXCEPTION_IF_NULL(file);
  *token = kInvalidAsyncIOToken;
  return file->PRead(data, byte_num, kFileHeadOffset);
}

bool IOHandle::WriteAsync(const std::string &file_name, const void *data, size_t byte_num, AsyncIOToken *token) {
  if (aio_ != nullptr) {
    return aio_->WriteAsync(file_name, data, byte_num, token);
  }
  const auto &fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  auto file = fs->CreateWriteFile(GetSwapFileWholeName(file_name));
  MS_EXCEPTION_IF_NULL(file);
  *token = kInvalidAsyncIOToken;
  return file->PWrite(data, byte_num, kFileHeadOffset);
}

bool IOHandle::Wait(AsyncIOToken token) { return aio_ == nullptr || aio_->Wait(token); }

bool IOHandle::DeleteSwapFile(const std::string &file_name) const {
  const auto &fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  return fs->DeleteFile(GetSwapFileWholeName(file_name));
}

bool IOHandle::CreateSwapFile(const std::string &file_name) const {
  const auto &fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  auto file = fs->CreateWriteFile(GetSwapFileWholeName(file_name));
  MS_EXCEPTION_IF_NULL(file);
  return true;
}

std::string IOHandle::GetSwapFileWholeName(const std::string &file_name) const {
  return swap_path_ + file_name + ".data";
}
}  // namespace device
}  // namespace mindspore
