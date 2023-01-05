/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORPRINT_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORPRINT_UTILS_H_

#include <map>
#include <string>
#include <thread>
#include <functional>
#include "acl/acl_tdt.h"

namespace mindspore::device::ascend {
class TensorPrint {
 public:
  explicit TensorPrint(const std::string &path, const acltdtChannelHandle *acl_handle)
      : print_file_path_(path), acl_handle_(acl_handle) {}
  ~TensorPrint() = default;
  void operator()();

 private:
  std::string print_file_path_;
  const acltdtChannelHandle *acl_handle_;
};

enum class ChannelType {
  kMbuf = 0,
  kTDT,
};

class AclHandle {
 public:
  static AclHandle &GetInstance();
  ChannelType GetChannelType() { return channel_type_; }
  acltdtChannelHandle *Get() { return acl_handle_; }
  bool CreateChannel(uint32_t deviceId, std::string name, size_t capacity = 16);

  ~AclHandle() = default;
  AclHandle(const AclHandle &) = delete;
  AclHandle &operator=(const AclHandle &) = delete;

 private:
  AclHandle() = default;

  acltdtChannelHandle *acl_handle_{nullptr};
  ChannelType channel_type_{ChannelType::kMbuf};
};

using PrintThreadCrt = std::function<std::thread(std::string &, acltdtChannelHandle *)>;
void CreateTensorPrintThread(const PrintThreadCrt &ctr);
void DestroyTensorPrintThread();
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORPRINT_UTILS_H_
