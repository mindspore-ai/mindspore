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
#ifdef _MSC_VER
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <memory>
#include "utils/system/env.h"
#include "include/common/utils/offload_context.h"

namespace mindspore {
namespace device {
constexpr size_t kFileHeadOffset = 0;

void IOHandle::LoadAio(const std::string &aio_shared_lib_name, const std::string &instance_func_name) {
#ifdef _MSC_VER
  auto handle = LoadLibrary(aio_shared_lib_name.c_str());
  if (handle == nullptr) {
    MS_LOG(WARNING) << "Loading " << aio_shared_lib_name << " filed.";
  }
  auto get_aio_instance = reinterpret_cast<AsyncIO *(*)()>(GetProcAddress(handle, instance_func_name.c_str()));
  if (get_aio_instance == nullptr) {
    MS_LOG(WARNING) << "Getting function " << instance_func_name << " from " << aio_shared_lib_name << " failed.";
    return;
  }
#else
  auto handle = dlopen(aio_shared_lib_name.c_str(), RTLD_NOW);
  if (handle == nullptr) {
    MS_LOG(WARNING) << "Loading " << aio_shared_lib_name << " filed. Error message: " << dlerror();
    return;
  }
  auto get_aio_instance = reinterpret_cast<AsyncIO *(*)()>(dlsym(handle, instance_func_name.c_str()));
  if (get_aio_instance == nullptr) {
    MS_LOG(WARNING) << "Getting function " << instance_func_name << " from " << aio_shared_lib_name
                    << " failed. Error message: " << dlerror();
    return;
  }
#endif
  aio_ = get_aio_instance();
  MS_EXCEPTION_IF_NULL(aio_);
  const auto &offload_context = OffloadContext::GetInstance();
  MS_EXCEPTION_IF_NULL(offload_context);
  aio_->Init({offload_context->aio_block_size(), offload_context->aio_queue_depth()});
}

bool IOHandle::Read(const std::string &file_name, void *data, size_t byte_num) {
  if (aio_ != nullptr) {
    return aio_->Read(GetSwapFileWholeName(file_name), data, byte_num);
  }
  const auto &fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  auto file = fs->CreateWriteFile(GetSwapFileWholeName(file_name));
  MS_EXCEPTION_IF_NULL(file);
  return file->PRead(data, byte_num, kFileHeadOffset);
}

bool IOHandle::Write(const std::string &file_name, const void *data, size_t byte_num) {
  if (aio_ != nullptr) {
    return aio_->Write(GetSwapFileWholeName(file_name), data, byte_num);
  }
  const auto &fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  auto file = fs->CreateWriteFile(GetSwapFileWholeName(file_name));
  MS_EXCEPTION_IF_NULL(file);
  return file->PWrite(data, byte_num, kFileHeadOffset);
}

bool IOHandle::ReadAsync(const std::string &file_name, void *data, size_t byte_num, AsyncIOToken *token) {
  if (aio_ != nullptr) {
    return aio_->ReadAsync(GetSwapFileWholeName(file_name), data, byte_num, token);
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
    return aio_->WriteAsync(GetSwapFileWholeName(file_name), data, byte_num, token);
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
