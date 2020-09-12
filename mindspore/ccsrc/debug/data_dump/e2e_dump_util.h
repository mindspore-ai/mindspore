/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_E_2_E_DUMP_UTIL_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_E_2_E_DUMP_UTIL_H_

#include <string>
#include "backend/session/kernel_graph.h"
#include "runtime/device/device_address.h"
#ifndef ENABLE_DEBUGGER
class Debugger;
#endif
namespace mindspore {
class E2eDumpUtil {
 public:
  E2eDumpUtil() = default;
  ~E2eDumpUtil() = default;
  static bool DumpData(const session::KernelGraph *graph, Debugger *debugger = nullptr);

 private:
  static void DumpOutput(const session::KernelGraph *graph, const std::string &dump_path, Debugger *debugger);
  static void DumpInput(const session::KernelGraph *graph, const std::string &dump_path, Debugger *debugger);
  static void DumpParameters(const session::KernelGraph *graph, const std::string &dump_path, Debugger *debugger);

  static void GetFileKernelName(NotNull<std::string *> kernel_name);
  static void DumpMemToFile(const std::string &file_path, NotNull<const device::DeviceAddress *> addr, bool trans_flag,
                            const ShapeVector &int_shapes, const TypeId &type);
  static void DumpGPUMemToFile(const std::string &file_path, const std::string &original_kernel_name,
                               NotNull<const device::DeviceAddress *> addr, bool trans_flag,
                               const ShapeVector &int_shapes, const TypeId &type, size_t slot, Debugger *debugger);
  static void GetDumpIntShape(const AnfNodePtr &node, size_t index, bool trans_flag, NotNull<ShapeVector *> int_shapes);
  static bool IsDeviceTargetGPU();
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_E_2_E_DUMP_UTIL_H_
