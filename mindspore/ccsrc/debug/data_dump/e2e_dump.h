/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_E_2_E_DUMP_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_E_2_E_DUMP_H_

#include <dirent.h>
#include <stdlib.h>
#include <map>
#include <string>

#include "backend/session/kernel_graph.h"
#include "runtime/device/device_address.h"
#include "debug/data_dump/dump_json_parser.h"
#include "debug/data_dump/dump_utils.h"

#ifndef ENABLE_DEBUGGER
class Debugger;
#endif
namespace mindspore {
class E2eDump {
 public:
  E2eDump() = default;
  ~E2eDump() = default;
  static void DumpSetup(const session::KernelGraph *graph);

  static void UpdateIterGPUDump();

  static void DumpData(const session::KernelGraph *graph, uint32_t rank_id, const Debugger *debugger = nullptr);

  static bool DumpParametersAndConstData(const session::KernelGraph *graph, uint32_t rank_id, const Debugger *debugger);

  static bool DumpSingleNodeData(const CNodePtr &node, uint32_t graph_id, uint32_t rank_id,
                                 const Debugger *debugger = nullptr);

  static bool isDatasetGraph(const session::KernelGraph *graph);

  // Dump data when task error.
  static void DumpInputImpl(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                            std::string *kernel_name, const Debugger *debugger);

  static void DumpOutputImpl(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                             std::string *kernel_name, const Debugger *debugger);

  static bool DumpDirExists(const std::string &dump_path);

 private:
  static void DumpOutput(const session::KernelGraph *graph, const std::string &dump_path, const Debugger *debugger);

  static void DumpOutputSingleNode(const CNodePtr &node, const std::string &dump_path, const Debugger *debugger);

  static void DumpInput(const session::KernelGraph *graph, const std::string &dump_path, const Debugger *debugger);

  static void DumpInputSingleNode(const CNodePtr &node, const std::string &dump_path, const Debugger *debugger);

  static void DumpParametersAndConst(const session::KernelGraph *graph, const std::string &dump_path,
                                     const Debugger *debugger);

  static void DumpGPUMemToFile(const std::string &file_path, const std::string &original_kernel_name,
                               const device::DeviceAddress &addr, const ShapeVector &int_shapes,
                               const TypeId &host_type, const TypeId &device_type, bool trans_flag, size_t slot,
                               const Debugger *debugger);
  static bool IsDeviceTargetGPU();
  static void DumpSingleAnfNode(const AnfNodePtr &anf_node, const size_t output_index, const std::string &dump_path,
                                bool trans_flag, std::map<std::string, size_t> *const_map, const Debugger *debugger);

  static void UpdateIterDumpSetup(const session::KernelGraph *graph, bool sink_mode);

  inline static unsigned int starting_graph_id = INT32_MAX;
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_E_2_E_DUMP_UTIL_H_
