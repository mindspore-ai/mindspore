/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEBUG_RDR_MEM_ADDRESS_RECORDER_H_
#define MINDSPORE_CCSRC_DEBUG_RDR_MEM_ADDRESS_RECORDER_H_
#include <vector>
#include <string>
#include <map>
#include <set>
#include <memory>
#include <mutex>

#include "debug/rdr/base_recorder.h"

namespace mindspore {
namespace kernel {
class Address;
struct KernelLaunchInfo;
using AddressPtr = std::shared_ptr<Address>;
}  // namespace kernel
using AddressPtrList = std::vector<kernel::AddressPtr>;
struct MemInfo {
  AddressPtrList *inputs_;
  AddressPtrList *workspaces_;
  AddressPtrList *outputs_;
};
class MemAddressRecorder : public BaseRecorder {
 public:
  MemAddressRecorder() {}
  MemAddressRecorder(const std::string &module, const std::string &name) : BaseRecorder(module, name) {}
  ~MemAddressRecorder() {}

  virtual void Export();
  void SaveMemInfo(const std::string &op_name, const kernel::KernelLaunchInfo &mem_info);

  void Reset() {
    op_names_.clear();
    mem_info_stream_.str("");
  }
  void CleanUp();

 private:
  mutable std::mutex mtx_;
  bool printed_{false};

  std::set<std::string> op_names_;
  std::ostringstream mem_info_stream_;
};
using MemAddressRecorderPtr = std::shared_ptr<MemAddressRecorder>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_RDR_MEM_ADDRESS_RECORDER_H_
