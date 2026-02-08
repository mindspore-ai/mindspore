/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DUMP_CONTROL_H_
#define MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DUMP_CONTROL_H_

#include <string>
#include <vector>

#include "tools/visible.h"
#include "utils/ms_utils.h"

namespace mindspore {

class TOOLS_EXPORT DumpControl {
 public:
  static DumpControl &GetInstance() {
    static DumpControl instance;
    return instance;
  }
  ~DumpControl() = default;
  bool dynamic_switch() const { return dynamic_switch_; }
  bool dump_switch() const { return dump_switch_; }

  void SetDynamicDump() { dynamic_switch_ = true; }
  void DynamicDumpStart();
  void DynamicDumpStop();
  void SetInitialIteration(std::uint32_t initial_iteration);
  void UpdateUserDumpStep(const std::uint32_t step);

 private:
  DumpControl() = default;
  DISABLE_COPY_AND_ASSIGN(DumpControl);
  bool dynamic_switch_{false};
  bool dump_switch_{false};
};

}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DUMP_CONTROL_H_
