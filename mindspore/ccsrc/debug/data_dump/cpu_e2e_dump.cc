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
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {

void CPUE2eDump::DumpCNodeData(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  std::string kernel_name = node->fullname_with_scope();
  if (!dump_json_parser.NeedDump(kernel_name)) {
    return;
  }

  MS_LOG(DEBUG) << "E2e dump CNode data start: " << kernel_name << ", current iteration is "
                << dump_json_parser.cur_dump_iter();
  std::string dump_path = GenerateDumpPath();
  if (dump_json_parser.InputNeedDump()) {
    DumpCNodeInputs(node, dump_path);
  }
  if (dump_json_parser.OutputNeedDump()) {
    DumpCNodeOutputs(node, dump_path);
  }
}

void CPUE2eDump::DumpCNodeInputs(const CNodePtr &node, const std::string &dump_path) {
  MS_EXCEPTION_IF_NULL(node);
  std::string kernel_name = node->fullname_with_scope();
  MS_LOG(DEBUG) << "Start e2e dump CNode inputs data: " << kernel_name;
  DumpJsonParser::GetInstance().MatchKernel(kernel_name);
  DumpInputImpl(node, dump_path, &kernel_name);
}

void CPUE2eDump::DumpCNodeOutputs(const CNodePtr &node, const std::string &dump_path) {
  MS_EXCEPTION_IF_NULL(node);
  std::string kernel_name = node->fullname_with_scope();
  MS_LOG(DEBUG) << "Start e2e dump CNode outputs data: " << kernel_name;
  DumpJsonParser::GetInstance().MatchKernel(kernel_name);
  DumpOutputImpl(node, dump_path, &kernel_name);
}

void CPUE2eDump::DumpInputImpl(const CNodePtr &node, const std::string &dump_path, std::string *kernel_name) {
  MS_EXCEPTION_IF_NULL(node);
  GetFileKernelName(NOT_NULL(kernel_name));
  auto input_size = AnfAlgo::GetInputTensorNum(node);
  for (size_t j = 0; j < input_size; ++j) {
    auto kernel_with_index = AnfAlgo::GetPrevNodeOutput(node, j);
    auto input = kernel_with_index.first;
    auto index = kernel_with_index.second;
    if (!AnfAlgo::OutputAddrExist(input, index)) {
      continue;
    }
    auto addr = AnfAlgo::GetOutputAddr(input, index);
    std::string tensor_name = node->fullname_with_scope();

    ShapeVector int_shapes;
    GetDumpIntShape(input, index, NOT_NULL(&int_shapes));
    auto type = AnfAlgo::GetOutputInferDataType(input, index);
    std::string file_path = dump_path + '/' + *kernel_name + '_' + "input_" + std::to_string(j);
    DumpMemToFile(file_path, NOT_NULL(addr), int_shapes, type);
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
    ShapeVector int_shapes;
    GetDumpIntShape(node, j, NOT_NULL(&int_shapes));
    auto type = AnfAlgo::GetOutputInferDataType(node, j);
    std::string file_path = dump_path + '/' + *kernel_name + '_' + "output_" + std::to_string(j);
    DumpMemToFile(file_path, NOT_NULL(addr), int_shapes, type);
  }
}

void CPUE2eDump::DumpSingleAnfNode(const AnfNodePtr &anf_node, const size_t output_index, const std::string &dump_path,
                                   std::map<std::string, size_t> *const_map) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!anf_node->isa<Parameter>() && !anf_node->isa<ValueNode>()) {
    return;
  }
  std::string node_name = anf_node->fullname_with_scope();
  std::string dump_name = node_name;
  if (anf_node->isa<ValueNode>()) {
    auto iter = const_map->find(node_name);
    if (iter == const_map->end()) {
      return;
    }
    dump_name = std::string("cst") + std::to_string(iter->second);
  }

  if (!dump_json_parser.NeedDump(node_name)) {
    return;
  }
  DumpJsonParser::GetInstance().MatchKernel(node_name);
  GetFileKernelName(NOT_NULL(&node_name));
  // check if output address exists, if not, return;
  if (!AnfAlgo::OutputAddrExist(anf_node, output_index)) {
    return;
  }
  auto addr = AnfAlgo::GetOutputAddr(anf_node, output_index);
  MS_EXCEPTION_IF_NULL(addr);
  ShapeVector int_shapes;
  GetDumpIntShape(anf_node, output_index, NOT_NULL(&int_shapes));
  auto type = AnfAlgo::GetOutputInferDataType(anf_node, output_index);

  std::string file_path = dump_path + '/' + dump_name + '_' + "output_0";
  DumpMemToFile(file_path, NOT_NULL(addr), int_shapes, type);
}

void CPUE2eDump::DumpParametersAndConst(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Start e2e dump parameters and Const values";
  std::map<std::string, size_t> const_map;
  GetConstantId(graph, &const_map);
  const std::string &dump_path = GenerateDumpPath();

  // dump parameters
  const auto &parameters = graph->inputs();
  for (auto &item : parameters) {
    DumpSingleAnfNode(item, PARAMETER_OUTPUT_INDEX, dump_path, &const_map);
  }
  // dump const values
  auto value_nodes = graph->graph_value_nodes();
  for (const auto &value_node : value_nodes) {
    DumpSingleAnfNode(value_node, VALUE_NODE_OUTPUT_INDEX, dump_path, &const_map);
  }
}
}  // namespace mindspore
