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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_UTILS_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_UTILS_H_

#include <string>
#include <set>
#include "common/util/error_manager/error_manager.h"
#include "plugin/device/ascend/hal/hardware/ascend_device_context.h"
#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace device {
namespace ascend {
template <typename Map, typename K = typename Map::key_type, typename V = typename Map::mapped_type>
std::string MapToString(const Map &value) {
  std::stringstream buffer;
  buffer << "{";
  for (auto it = value.begin(); it != value.end(); it++) {
    if (it != value.begin()) {
      buffer << ", ";
    }
    buffer << it->first << ": " << it->second;
  }
  buffer << "}";
  return buffer.str();
}

class ErrorManagerAdapter {
 public:
  ErrorManagerAdapter() = default;
  ~ErrorManagerAdapter() = default;
  static bool Init();
  static std::string GetErrorMessage(bool add_title = false);
  static std::string GetWarningMessage(bool add_title = false);
  static void BindToCurrentThread();

 private:
  static void MessageHandler(std::ostringstream *oss);

 private:
  static error_message::Context context_;
  static std::mutex initialized_mutex_;
  static bool initialized_;
};

bool IsGraphMode();
bool IsDynamicShapeGraph(const FuncGraphPtr &func_graph);

std::string GetSocVersion();
std::string GetAICoreNumber();
std::string GetAscendPath();

// Some NOP nodes have be hide in execution order, it doesn't have output device address, this function creates
// output device address for these nodes, and the output device address is the same with input device address.
void AssignOutputNopNodeDeviceAddress(const KernelGraphPtr &graph, const device::DeviceContext *device_context);

std::string GetErrorMsg(uint32_t rt_error_code);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_UTILS_H_
