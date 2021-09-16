/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_KERNEL_MOD_H_

#include <vector>
#include <memory>
#include "runtime/device/ascend/ge_runtime/task_info.h"
#include "backend/kernel_compiler/kernel.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif

using TaskInfoPtr = std::shared_ptr<mindspore::ge::model_runner::TaskInfo>;
namespace mindspore {
namespace kernel {
class AscendKernelMod : public KernelMod {
 public:
  virtual std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                           const std::vector<AddressPtr> &, uint32_t) = 0;
  uint32_t block_dim() { return block_dim_; }
  uint32_t stream_id() { return stream_id_; }
  virtual bool NeedDump() {
#ifndef ENABLE_SECURITY
    const auto &dump_json = DumpJsonParser::GetInstance();
    return dump_json.NeedDump(fullname_) && dump_json.async_dump_enabled() && dump_json.op_debug_mode() == 0 &&
           !is_monad_;
#else
    return false;
#endif
  }

 protected:
  uint32_t block_dim_{1};
  uint32_t stream_id_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_KERNEL_MOD_H_
