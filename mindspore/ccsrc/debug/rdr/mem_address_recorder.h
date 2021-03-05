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
#include <memory>
#include <mutex>

#include "debug/rdr/base_recorder.h"

namespace mindspore {
namespace kernel {
class Address;
using AddressPtr = std::shared_ptr<Address>;
}  // namespace kernel
using AddressPtrList = std::vector<kernel::AddressPtr>;
struct GPUMemInfo {
  AddressPtrList *inputs_;
  AddressPtrList *workspaces_;
  AddressPtrList *outputs_;
};
class MemAddressRecorder : public BaseRecorder {
 public:
  static MemAddressRecorder &Instance();
  virtual void Export();
  void SaveMemInfo(const std::string &op_name, const GPUMemInfo &mem_info);
  void SetTag(const std::string &tag) { tag_ = tag; }

 private:
  MemAddressRecorder() {}
  MemAddressRecorder(const MemAddressRecorder &recorder);
  ~MemAddressRecorder() {}
  MemAddressRecorder &operator=(const MemAddressRecorder &recorder);

  mutable std::mutex mtx_;
  std::string mem_info_str_;
};
using MemAddressRecorderPtr = std::shared_ptr<MemAddressRecorder>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_RDR_MEM_ADDRESS_RECORDER_H_
