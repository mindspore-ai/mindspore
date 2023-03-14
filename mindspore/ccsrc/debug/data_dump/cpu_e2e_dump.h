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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_CPU_E_2_E_DUMP_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_CPU_E_2_E_DUMP_H_

#include <map>
#include <string>

#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/debug/data_dump/dump_utils.h"

namespace mindspore {
class CPUE2eDump {
 public:
  CPUE2eDump() = default;
  ~CPUE2eDump() = default;
  // Dump data when task error.
  static void DumpParameters(const session::KernelGraph *graph, uint32_t graph_id);

  static void DumpParametersData();

  static void DumpConstants(const session::KernelGraph *graph, uint32_t graph_id);

  static void DumpConstantsData();

  static void DumpCNodeData(const CNodePtr &node, uint32_t graph_id);

  static void DumpRunIter(const KernelGraphPtr &graph_ptr, uint32_t rank_id = 0);

 private:
  static void DumpCNodeInputs(const CNodePtr &node, const std::string &dump_path);

  static void DumpCNodeOutputs(const CNodePtr &node, const std::string &dump_path);

  static void DumpSingleAnfNode(const AnfNodePtr &anf_node, size_t output_index, const std::string &dump_path);

  static void DumpInputImpl(const CNodePtr &node, const std::string &dump_path, std::string *kernel_name);

  static void DumpOutputImpl(const CNodePtr &node, const std::string &dump_path, std::string *kernel_name);

  inline static unsigned int prev_run_iter_ = UINT32_MAX;
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_CPU_E_2_E_DUMP_H_
