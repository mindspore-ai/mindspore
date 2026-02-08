/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_E_2_E_DUMP_H_
#define MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_E_2_E_DUMP_H_

#include <dirent.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "device_address/device_address.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "tools/data_dump/dump_json_parser.h"
#include "tools/data_dump/dump_utils.h"
#ifdef ENABLE_DEBUGGER
#include "tools/data_dump/debugger/debugger.h"
#endif
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "tools/visible.h"

#ifndef ENABLE_DEBUGGER
class Debugger;
#endif
namespace mindspore {
using mindspore::device::DeviceContext;

class TOOLS_EXPORT E2eDump {
 public:
  E2eDump() = default;
  ~E2eDump() = default;
  static void UpdateIterMindRTDump();

  static void DumpRunIter(const KernelGraphPtr &graph_ptr, uint32_t rank_id = 0);

  static void DumpConstantData(const session::KernelGraph *graph, const std::string &cst_dump_path,
                               const Debugger *debugger = nullptr);

  static void DumpConstantData(const session::KernelGraph *graph, uint32_t rank_id, const Debugger *debugger = nullptr);

  static void DumpParametersData(uint32_t rank_id, const Debugger *debugger);

  static bool DumpSingleNodeData(const CNodePtr &node, uint32_t graph_id, uint32_t rank_id,
                                 const Debugger *debugger = nullptr, const DeviceContext *device_context = nullptr);

  // Dump data when task error.
  static void DumpInputImpl(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                            std::string *kernel_name, const Debugger *debugger,
                            const DeviceContext *device_context = nullptr);

  static void DumpOutputImpl(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                             std::string *kernel_name, const Debugger *debugger,
                             const DeviceContext *device_context = nullptr);
  // Dump input/output data without additional check, used for exception case only
  static void DumpInputData(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                            std::string *kernel_name);
  static void DumpOutputData(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                             std::string *kernel_name);

  static bool IsDeviceTargetGPU();

  static bool IsDeviceTargetAscend();

  static bool IsMindRTKernelByKernel();

 private:
  static void DumpOutputSingleNode(const CNodePtr &node, const std::string &dump_path, const Debugger *debugger,
                                   const DeviceContext *device_context = nullptr);

  static void DumpInputSingleNode(const CNodePtr &node, const std::string &dump_path, const Debugger *debugger,
                                  const DeviceContext *device_context = nullptr);

  static void DumpArgsSingleNode(const CNodePtr &node, const std::string &dump_path, const Debugger *debugger);

  static void DumpMemFromTensorLoaderToFile(const Debugger *debugger, const std::string &file_path,
                                            const std::string &original_kernel_name, size_t slot);

  static void DumpSingleAnfNode(const AnfNodePtr &anf_node, const size_t output_index, const std::string &dump_path,
                                bool trans_flag, const Debugger *debugger);

  static void DumpSingleParameterNode(const AnfNodePtr &anf_node, const std::string &dump_path, bool trans_flag,
                                      const Debugger *debugger);

  inline static unsigned int starting_graph_id = INT32_MAX;
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_TOOLS_DATA_DUMP_E_2_E_DUMP_H_
