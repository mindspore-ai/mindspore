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
class GPUMemAddressRecorder : public BaseRecorder {
 public:
  GPUMemAddressRecorder() {}
  GPUMemAddressRecorder(const std::string &module, const std::string &name) : BaseRecorder(module, name) {}
  ~GPUMemAddressRecorder() {}

  virtual void Export();
  void SaveMemInfo(const std::string &op_name, const GPUMemInfo &mem_info, size_t id);
  void Reset(size_t nsize) {
    op_names_.resize(nsize);
    mem_info_inputs_.resize(nsize);
    mem_info_workspaces_.resize(nsize);
    mem_info_outputs_.resize(nsize);
  }

 private:
  mutable std::mutex mtx_;
  std::vector<std::string> op_names_;
  std::vector<AddressPtrList> mem_info_inputs_;
  std::vector<AddressPtrList> mem_info_workspaces_;
  std::vector<AddressPtrList> mem_info_outputs_;
};
using GPUMemAddressRecorderPtr = std::shared_ptr<GPUMemAddressRecorder>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_RDR_MEM_ADDRESS_RECORDER_H_
