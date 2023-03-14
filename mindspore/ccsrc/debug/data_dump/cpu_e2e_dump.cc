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

#include "debug/data_dump/cpu_e2e_dump.h"
#include <map>
#include <fstream>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_dump_utils.h"
#include "include/common/debug/common.h"
#include "mindspore/core/utils/file_utils.h"

namespace mindspore {
void CPUE2eDump::DumpCNodeData(const CNodePtr &node, uint32_t graph_id) {
  MS_EXCEPTION_IF_NULL(node);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  std::string kernel_name = GetKernelNodeName(node);
  if (!dump_json_parser.NeedDump(kernel_name)) {
    return;
  }

  MS_LOG(DEBUG) << "E2e dump CNode data start: " << kernel_name << ", current iteration is "
                << dump_json_parser.cur_dump_iter();
  std::string dump_path = GenerateDumpPath(graph_id);
  if (dump_json_parser.InputNeedDump()) {
    DumpCNodeInputs(node, dump_path);
  }
  if (dump_json_parser.OutputNeedDump()) {
    DumpCNodeOutputs(node, dump_path);
  }
}

void CPUE2eDump::DumpRunIter(const KernelGraphPtr &graph, uint32_t rank_id) {
  auto &json_parser = DumpJsonParser::GetInstance();
  // avoid dumping same iteration over and over
  if (!(json_parser.e2e_dump_enabled()) || json_parser.cur_dump_iter() == prev_run_iter_) {
    return;
  }
  std::string execution_order_path = json_parser.path() + "/rank_" + std::to_string(rank_id) + "/execution_order/";
  std::string file_name_to_check =
    execution_order_path + "/ms_global_execution_order_graph_" + std::to_string(graph->graph_id()) + ".csv";
  auto real_path = Common::CreatePrefixPath(file_name_to_check);
  if (!real_path.has_value()) {
    MS_LOG(WARNING) << "Check file path: " << file_name_to_check << " failed.";
    return;
  }
  std::string file_name = real_path.value();
  ChangeFileMode(file_name, S_IWUSR);
  std::ofstream fout(file_name, std::ofstream::app);
  if (!fout.is_open()) {
    MS_LOG(WARNING) << "Open file for saving graph global execution order failed.";
    return;
  }
  fout << std::to_string(json_parser.cur_dump_iter()) + "\n";
  fout.close();
  ChangeFileMode(file_name, S_IRUSR);
  prev_run_iter_ = json_parser.cur_dump_iter();
}

void CPUE2eDump::DumpCNodeInputs(const CNodePtr &node, const std::string &dump_path) {
  MS_EXCEPTION_IF_NULL(node);
  std::string kernel_name = GetKernelNodeName(node);
  MS_LOG(DEBUG) << "Start e2e dump CNode inputs data: " << kernel_name;
  DumpJsonParser::GetInstance().MatchKernel(kernel_name);
  DumpInputImpl(node, dump_path, &kernel_name);
}

void CPUE2eDump::DumpCNodeOutputs(const CNodePtr &node, const std::string &dump_path) {
  MS_EXCEPTION_IF_NULL(node);
  std::string kernel_name = GetKernelNodeName(node);
  MS_LOG(DEBUG) << "Start e2e dump CNode outputs data: " << kernel_name;
  DumpJsonParser::GetInstance().MatchKernel(kernel_name);
  DumpOutputImpl(node, dump_path, &kernel_name);
}

void CPUE2eDump::DumpInputImpl(const CNodePtr &node, const std::string &dump_path, std::string *kernel_name) {
  MS_EXCEPTION_IF_NULL(node);
  GetFileKernelName(NOT_NULL(kernel_name));
  auto input_size = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t j = 0; j < input_size; ++j) {
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, j);
    auto input = kernel_with_index.first;
    auto index = kernel_with_index.second;
    if (!AnfAlgo::OutputAddrExist(input, index)) {
      continue;
    }
    auto addr = AnfAlgo::GetOutputAddr(input, index);
    ShapeVector int_shapes;
    GetDumpIntShape(input, index, NOT_NULL(&int_shapes));
    auto type = common::AnfAlgo::GetOutputInferDataType(input, index);
    std::string op_type = common::AnfAlgo::GetCNodeName(node);
    std::string op_name = GetOpNameWithoutScope(*kernel_name);
    uint64_t timestamp = Common::GetTimeStamp();
    const uint32_t kTaskId = 0;
    const uint32_t kStreamId = 0;
    std::string file_path = dump_path + '/' + op_type + '.' + op_name + '.' + std::to_string(kTaskId) + '.' +
                            std::to_string(kStreamId) + '.' + std::to_string(timestamp) + ".input." + std::to_string(j);
    MS_EXCEPTION_IF_NULL(addr);
    DumpMemToFile(file_path, *addr, int_shapes, type);
  }
}

void CPUE2eDump::DumpOutputImpl(const CNodePtr &node, const std::string &dump_path, std::string *kernel_name) {
  MS_EXCEPTION_IF_NULL(node);
  GetFileKernelName(NOT_NULL(kernel_name));
  auto output_size = AnfAlgo::GetOutputTensorNum(node);
  for (size_t j = 0; j < output_size; ++j) {
    if (!AnfAlgo::OutputAddrExist(node, j)) {
      continue;
    }
    auto addr = AnfAlgo::GetOutputAddr(node, j);
    MS_EXCEPTION_IF_NULL(addr);
    ShapeVector int_shapes;
    GetDumpIntShape(node, j, NOT_NULL(&int_shapes));
    auto type = common::AnfAlgo::GetOutputInferDataType(node, j);
    std::string op_type = common::AnfAlgo::GetCNodeName(node);
    std::string op_name = GetOpNameWithoutScope(*kernel_name);
    const uint32_t kTaskId = 0;
    const uint32_t kStreamId = 0;
    uint64_t timestamp = Common::GetTimeStamp();
    std::string file_path = dump_path + '/' + op_type + '.' + op_name + '.' + std::to_string(kTaskId) + '.' +
                            std::to_string(kStreamId) + '.' + std::to_string(timestamp) + ".output." +
                            std::to_string(j);
    DumpMemToFile(file_path, *addr, int_shapes, type);
  }
}

void CPUE2eDump::DumpSingleAnfNode(const AnfNodePtr &anf_node, const size_t output_index,
                                   const std::string &dump_path) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!anf_node->isa<Parameter>() && !anf_node->isa<ValueNode>()) {
    return;
  }
  std::string node_name = GetKernelNodeName(anf_node);
  if (!dump_json_parser.NeedDump(node_name)) {
    return;
  }
  DumpJsonParser::GetInstance().MatchKernel(node_name);
  GetFileKernelName(NOT_NULL(&node_name));
  std::string dump_name = node_name;
  const std::string cst_prefix = "Default--";
  if (anf_node->isa<ValueNode>()) {
    if (dump_name.find(cst_prefix) == std::string::npos) {
      MS_LOG(INFO) << "Incorrect constant format: " << dump_name;
      return;
    }
    dump_name = node_name.substr(cst_prefix.length());
  }

  // check if output address exists, if not, return;
  if (!AnfAlgo::OutputAddrExist(anf_node, output_index)) {
    return;
  }
  auto addr = AnfAlgo::GetOutputAddr(anf_node, output_index);
  MS_EXCEPTION_IF_NULL(addr);
  if (addr->GetPtr() == nullptr) {
    return;
  }
  ShapeVector int_shapes;
  GetDumpIntShape(anf_node, output_index, NOT_NULL(&int_shapes));
  auto type = common::AnfAlgo::GetOutputInferDataType(anf_node, output_index);

  uint64_t timestamp = Common::GetTimeStamp();
  const uint32_t kTaskId = 0;
  const uint32_t kStreamId = 0;
  std::string file_path = dump_path + "/Parameter." + dump_name + '.' + std::to_string(kTaskId) + '.' +
                          std::to_string(kStreamId) + '.' + std::to_string(timestamp) + ".output.0";
  DumpMemToFile(file_path, *addr, int_shapes, type);
}

void CPUE2eDump::DumpParameters(const session::KernelGraph *graph, uint32_t graph_id) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.OutputNeedDump()) {
    return;
  }
  MS_LOG(INFO) << "Start e2e dump parameters.";
  const std::string &dump_path = GenerateDumpPath(graph_id);

  // dump parameters
  const auto &parameters = graph->inputs();
  for (auto &item : parameters) {
    DumpSingleAnfNode(item, kParameterOutputIndex, dump_path);
  }
}

void CPUE2eDump::DumpParametersData() {
  auto &graphs = DumpJsonParser::GetInstance().graphs();
  for (auto graph : graphs) {
    DumpParameters(graph, graph->graph_id());
  }
}

void CPUE2eDump::DumpConstants(const session::KernelGraph *graph, uint32_t graph_id) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.OutputNeedDump()) {
    return;
  }
  MS_LOG(INFO) << "Start e2e dump constant.";
  uint32_t cur_iteration = DumpJsonParser::GetInstance().cur_dump_iter();
  if (cur_iteration != 0) {
    return;
  }
  const std::string &dump_path = GenerateDumpPath(graph_id, 0, true);

  // dump constants
  const auto value_nodes = graph->graph_value_nodes();
  for (auto &item : value_nodes) {
    DumpSingleAnfNode(item, kValueNodeOutputIndex, dump_path);
  }
}

void CPUE2eDump::DumpConstantsData() {
  auto &graphs = DumpJsonParser::GetInstance().graphs();
  for (auto graph : graphs) {
    DumpConstants(graph, graph->graph_id());
  }
}
}  // namespace mindspore
