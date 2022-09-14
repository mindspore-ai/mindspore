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
#include <string>
#include <map>
#include <set>
#include <memory>
#include <mutex>

#include "include/common/debug/rdr/base_recorder.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace kernel {
struct KernelLaunchInfo;
}  // namespace kernel
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

namespace RDR {
BACKEND_EXPORT bool RecordMemAddressInfo(const SubModuleId module, const std::string &name);
BACKEND_EXPORT bool UpdateMemAddress(const SubModuleId module, const std::string &name, const std::string &op_name,
                                     const kernel::KernelLaunchInfo &mem_info);
BACKEND_EXPORT void ClearMemAddressInfo();
}  // namespace RDR
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_RDR_MEM_ADDRESS_RECORDER_H_
