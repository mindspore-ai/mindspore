/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEPRECATED_INTERFACE_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEPRECATED_INTERFACE_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "runtime/hardware/device_context.h"
#include "runtime/device/memory_manager.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendDeviceContext;

class AscendDeprecatedInterface : public DeprecatedInterface {
 public:
  explicit AscendDeprecatedInterface(AscendDeviceContext *ascend_device_context)
      : ascend_device_context_(ascend_device_context) {}

  uint32_t InitCollective() override;
  void DumpProfileParallelStrategy(const FuncGraphPtr &func_graph) override;

 private:
  AscendDeviceContext *const ascend_device_context_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEPRECATED_INTERFACE_H_
