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

#include "debug/data_dump/e2e_dump.h"

#include <unistd.h>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include "debug/data_dump/dump_json_parser.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/debug/anf_dump_utils.h"
#include "include/common/debug/common.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "include/common/utils/config_manager.h"
#include "utils/file_utils.h"
#include "debug/data_dump/tensor_stat_dump.h"
#include "abstract/utils.h"
#include "runtime/hardware/device_context_manager.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debug_services.h"
#include "debug/tensor_load.h"
#include "debug/debugger/debugger.h"
#endif

namespace mindspore {
std::string GenDataFilePath(const CNodePtr &node, const std::string &kernel_name, const std::string &dump_path,
                            size_t slot, bool is_input) {
  std::string op_type = common::AnfAlgo::GetCNodeName(node);
  std::string op_name = GetOpNameWithoutScope(kernel_name);
  uint64_t timestamp = Common::GetTimeStamp();
  uint32_t task_id = 0;
  uint32_t stream_id = 0;
  if (E2eDump::IsDeviceTargetAscend()) {
    stream_id = AnfAlgo::GetStreamId(node);
  }
  std::string tensor_type = is_input ? ".input." : ".output.";
  std::string file_path = dump_path + '/' + op_type + '.' + op_name + '.' + std::to_string(task_id) + '.' +
                          std::to_string(stream_id) + '.' + std::to_string(timestamp) + tensor_type +
                          std::to_string(slot);
  return file_path;
}

bool E2eDump::IsDeviceTargetGPU() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice;
}

bool E2eDump::IsDeviceTargetAscend() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice;
}

bool E2eDump::IsMindRTKernelByKernel() {
  return IsDeviceTargetGPU() || Debugger::GetInstance()->GetAscendKernelByKernelFlag();
}

/*
 * Feature group: Dump.
 * Target device group: GPU, Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: This function is for dumping tensor loaded to tensor_loader in memory to disk in GPU and Ascend machine.
 */
void E2eDump::DumpMemFromTensorLoaderToFile(const Debugger *debugger, const std::string &file_path,
                                            const std::string &original_kernel_name, size_t slot) {
#ifdef ENABLE_DEBUGGER
  MS_EXCEPTION_IF_NULL(debugger);
  auto ret = debugger->DumpTensorToFile(file_path, original_kernel_name, slot);
  if (!ret) {
    MS_LOG(INFO) << "DumpTensorToFile Failed: path:" << file_path;
  }
#endif
}

void E2eDump::DumpOutput(const session::KernelGraph *graph, const std::string &dump_path, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.OutputNeedDump()) {
    return;
  }
  MS_LOG(INFO) << "Start e2e dump output";
  bool trans_flag = dump_json_parser.trans_flag();
  const auto &apply_kernels = graph->execution_order();
  for (const auto &node : apply_kernels) {
    MS_EXCEPTION_IF_NULL(node);
    std::string kernel_name = GetKernelNodeName(node);
    if (!dump_json_parser.NeedDump(kernel_name)) {
      continue;
    }
    DumpJsonParser::GetInstance().MatchKernel(kernel_name);
    DumpOutputImpl(node, trans_flag, dump_path, &kernel_name, debugger);
  }
}

void E2eDump::DumpOutputSingleNode(const CNodePtr &node, const std::string &dump_path, const Debugger *debugger) {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.OutputNeedDump()) {
    return;
  }
  bool trans_flag = dump_json_parser.trans_flag();
  MS_EXCEPTION_IF_NULL(node);
  std::string kernel_name = GetKernelNodeName(node);
  if (!dump_json_parser.NeedDump(kernel_name)) {
    return;
  }
  DumpJsonParser::GetInstance().MatchKernel(kernel_name);
  DumpOutputImpl(node, trans_flag, dump_path, &kernel_name, debugger);
}

void E2eDump::DumpOutputImpl(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                             std::string *kernel_name, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(node);
  GetFileKernelName(NOT_NULL(kernel_name));
  auto output_size = AnfAlgo::GetOutputTensorNum(node);
  for (size_t j = 0; j < output_size; ++j) {
    if (!AnfAlgo::OutputAddrExist(node, j)) {
      continue;
    }
    auto addr = AnfAlgo::GetOutputAddr(node, j);
    std::string node_name = GetKernelNodeName(node);
    MS_EXCEPTION_IF_NULL(addr);
    ShapeVector int_shapes;
    GetDumpIntShape(node, j, NOT_NULL(&int_shapes), trans_flag);
    auto type = common::AnfAlgo::GetOutputInferDataType(node, j);
    std::string op_type = common::AnfAlgo::GetCNodeName(node);
    std::string op_name = GetOpNameWithoutScope(*kernel_name);
    uint32_t task_id = 0;
    uint32_t stream_id = 0;
    if (IsDeviceTargetAscend()) {
      stream_id = AnfAlgo::GetStreamId(node);
    }
    uint64_t timestamp = Common::GetTimeStamp();
    std::string file_path = dump_path + '/' + op_type + '.' + op_name + '.' + std::to_string(task_id) + '.' +
                            std::to_string(stream_id) + '.' + std::to_string(timestamp) + ".output." +
                            std::to_string(j);
    if (DumpJsonParser::GetInstance().IsStatisticDump() && IsMindRTKernelByKernel()) {
      TensorStatDump stat_dump(op_type, op_name, task_id, stream_id, timestamp, false, j, j);
      (void)stat_dump.DumpTensorStatsToFile(node_name, dump_path, debugger);
    }
    if (DumpJsonParser::GetInstance().IsTensorDump()) {
      if (IsMindRTKernelByKernel()) {
        DumpMemFromTensorLoaderToFile(debugger, file_path, node_name, j);
      } else {
        DumpMemToFile(file_path, *addr, int_shapes, type, trans_flag);
      }
    }
  }
}

void E2eDump::DumpOutputData(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                             std::string *kernel_name) {
  if (IsMindRTKernelByKernel()) {
    MS_LOG(INFO) << "DumpOutputData is only for graph mode on Ascend";
    return;
  }
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
    GetDumpIntShape(node, j, NOT_NULL(&int_shapes), trans_flag);
    auto type = common::AnfAlgo::GetOutputInferDataType(node, j);
    std::string file_path = GenDataFilePath(node, *kernel_name, dump_path, j, false);
    DumpMemToFile(file_path, *addr, int_shapes, type, trans_flag);
  }
}

void E2eDump::DumpInput(const session::KernelGraph *graph, const std::string &dump_path, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.InputNeedDump()) {
    return;
  }
  MS_LOG(INFO) << "Start e2e dump input";
  bool trans_flag = dump_json_parser.trans_flag();
  const auto &apply_kernels = graph->execution_order();
  for (const auto &node : apply_kernels) {
    MS_EXCEPTION_IF_NULL(node);
    std::string kernel_name = GetKernelNodeName(node);
    if (!dump_json_parser.NeedDump(kernel_name)) {
      continue;
    }
    DumpJsonParser::GetInstance().MatchKernel(kernel_name);
    DumpInputImpl(node, trans_flag, dump_path, &kernel_name, debugger);
  }
}

void E2eDump::DumpInputSingleNode(const CNodePtr &node, const std::string &dump_path, const Debugger *debugger) {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.InputNeedDump()) {
    return;
  }
  bool trans_flag = dump_json_parser.trans_flag();
  MS_EXCEPTION_IF_NULL(node);
  std::string kernel_name = GetKernelNodeName(node);
  if (!dump_json_parser.NeedDump(kernel_name)) {
    return;
  }
  DumpJsonParser::GetInstance().MatchKernel(kernel_name);
  DumpInputImpl(node, trans_flag, dump_path, &kernel_name, debugger);
}

void E2eDump::DumpInputImpl(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                            std::string *kernel_name, const Debugger *debugger) {
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
    std::string node_name = GetKernelNodeName(node);
    size_t slot = j;
    if (IsMindRTKernelByKernel()) {
      auto input_kernel = node->input(j + 1);
      std::string input_kernel_name = GetKernelNodeName(input_kernel);
      node_name = input_kernel_name;
      slot = 0;
    }
    ShapeVector int_shapes;
    GetDumpIntShape(input, index, NOT_NULL(&int_shapes), trans_flag);
    auto type = common::AnfAlgo::GetOutputInferDataType(input, index);
    std::string op_type = common::AnfAlgo::GetCNodeName(node);
    std::string op_name = GetOpNameWithoutScope(*kernel_name);
    uint64_t timestamp = Common::GetTimeStamp();
    uint32_t task_id = 0;
    uint32_t stream_id = 0;
    if (IsDeviceTargetAscend()) {
      stream_id = AnfAlgo::GetStreamId(node);
    }
    std::string file_path = dump_path + '/' + op_type + '.' + op_name + '.' + std::to_string(task_id) + '.' +
                            std::to_string(stream_id) + '.' + std::to_string(timestamp) + ".input." + std::to_string(j);
    auto addr = AnfAlgo::GetOutputAddr(input, index);
    MS_EXCEPTION_IF_NULL(addr);
    if (DumpJsonParser::GetInstance().IsStatisticDump() && IsMindRTKernelByKernel()) {
      TensorStatDump stat_dump(op_type, op_name, task_id, stream_id, timestamp, true, j, slot);
      (void)stat_dump.DumpTensorStatsToFile(node_name, dump_path, debugger);
    }
    if (DumpJsonParser::GetInstance().IsTensorDump()) {
      if (IsMindRTKernelByKernel()) {
        DumpMemFromTensorLoaderToFile(debugger, file_path, node_name, slot);
      } else {
        DumpMemToFile(file_path, *addr, int_shapes, type, trans_flag);
      }
    }
  }
}

void E2eDump::DumpInputData(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                            std::string *kernel_name) {
  if (IsMindRTKernelByKernel()) {
    MS_LOG(INFO) << "DumpInputData is only for graph mode on Ascend";
    return;
  }
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
    MS_EXCEPTION_IF_NULL(addr);
    ShapeVector int_shapes;
    GetDumpIntShape(input, index, NOT_NULL(&int_shapes), trans_flag);
    auto type = common::AnfAlgo::GetOutputInferDataType(input, index);
    std::string file_path = GenDataFilePath(node, *kernel_name, dump_path, j, true);
    DumpMemToFile(file_path, *addr, int_shapes, type, trans_flag);
  }
}

void E2eDump::DumpSingleAnfNode(const AnfNodePtr &anf_node, const size_t output_index, const std::string &dump_path,
                                bool trans_flag, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if ((!anf_node->isa<Parameter>() && !anf_node->isa<ValueNode>()) || IsValueNode<StringImm>(anf_node)) {
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
    trans_flag = false;
  }
  // check if output address exists, if not, return;
  if (!AnfAlgo::OutputAddrExist(anf_node, output_index)) {
    return;
  }
  auto addr = AnfAlgo::GetOutputAddr(anf_node, output_index);
  MS_EXCEPTION_IF_NULL(addr);
  ShapeVector int_shapes;
  GetDumpIntShape(anf_node, output_index, NOT_NULL(&int_shapes), trans_flag);
  auto type = common::AnfAlgo::GetOutputInferDataType(anf_node, output_index);
  uint64_t timestamp = Common::GetTimeStamp();
  uint32_t task_id = 0;
  uint32_t stream_id = 0;
  std::string file_path = dump_path + "/Parameter." + dump_name + '.' + std::to_string(task_id) + '.' +
                          std::to_string(stream_id) + '.' + std::to_string(timestamp) + ".output.0";
  if (IsDeviceTargetGPU()) {
    if (dump_json_parser.IsStatisticDump()) {
      TensorStatDump stat_dump("Parameter", dump_name, task_id, stream_id, timestamp, false, 0, 0);
      (void)stat_dump.DumpTensorStatsToFile(node_name, dump_path, debugger);
    }
    if (dump_json_parser.IsTensorDump()) {
      DumpMemFromTensorLoaderToFile(debugger, file_path, node_name, 0);
    }
  } else {
    // On Ascend, saving statistic data is only supported npy format.
    if (dump_json_parser.IsStatisticDump() && dump_json_parser.IsNpyFormat()) {
      // On Ascend kernel by kernel mode, load tensor data into debugger first.
      auto format = kOpFormat_DEFAULT;
      std::string tensor_name = node_name + ":0";
      uint32_t root_graph_id = debugger->GetCurrentRootGraphId();
      bool ret = addr->LoadMemToHost(tensor_name, 0, format, int_shapes, type, 0, true, root_graph_id, false, true);
      if (!ret) {
        MS_LOG(ERROR) << "LoadMemToHost failed, tensor_name: " << tensor_name;
      } else {
        TensorStatDump stat_dump("Parameter", dump_name, task_id, stream_id, timestamp, false, 0, 0);
        (void)stat_dump.DumpTensorStatsToFile(node_name, dump_path, debugger);
      }
    }
    if (dump_json_parser.IsTensorDump()) {
      DumpMemToFile(file_path, *addr, int_shapes, type, trans_flag);
    }
  }
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: This function is similar to DumpSingleAnfNode function but it is only for dumping parameters in mindRT.
 * This function uses GetParameterInfo to get dump info for the parameter node.
 */
void E2eDump::DumpSingleParameterNode(const AnfNodePtr &anf_node, const std::string &dump_path, bool trans_flag,
                                      const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  std::string node_name = GetKernelNodeName(anf_node);
  if (!anf_node->isa<Parameter>() || !dump_json_parser.NeedDump(node_name) || !dump_json_parser.OutputNeedDump()) {
    return;
  }
  DumpJsonParser::GetInstance().MatchKernel(node_name);
  GetFileKernelName(NOT_NULL(&node_name));
  ShapeVector int_shapes;
  TypeId type;
  TypeId device_type;
  auto addr = GetParameterInfo(anf_node, NOT_NULL(&int_shapes), NOT_NULL(&type), NOT_NULL(&device_type));
  if (addr == nullptr || addr->GetPtr() == nullptr) {
    MS_LOG(DEBUG) << "Skip node: " << node_name << ". Parameter data is not available for mindRT.";
    return;
  }
  uint64_t timestamp = Common::GetTimeStamp();
  uint32_t task_id = 0;
  uint32_t stream_id = 0;
  std::string file_path = dump_path + "/Parameter." + node_name + '.' + std::to_string(task_id) + '.' +
                          std::to_string(stream_id) + '.' + std::to_string(timestamp) + ".output.0";
  if (IsDeviceTargetGPU()) {
    if (dump_json_parser.IsStatisticDump()) {
      TensorStatDump stat_dump("Parameter", node_name, task_id, stream_id, timestamp, false, 0, 0);
      (void)stat_dump.DumpTensorStatsToFile(node_name, dump_path, debugger);
    }
    if (dump_json_parser.IsTensorDump()) {
      DumpMemFromTensorLoaderToFile(debugger, file_path, node_name, 0);
    }
  } else {
    // On Ascend, saving statistic data is only supported npy format.
    if (dump_json_parser.IsStatisticDump() && dump_json_parser.IsNpyFormat()) {
      // On Ascend kernel by kernel mode, load tensor data into debugger first.
      auto format = kOpFormat_DEFAULT;
      std::string tensor_name = node_name + ":0";
      uint32_t root_graph_id = debugger->GetCurrentRootGraphId();
      bool ret = addr->LoadMemToHost(tensor_name, 0, format, int_shapes, type, 0, true, root_graph_id, false, true);
      if (!ret) {
        MS_LOG(ERROR) << "LoadMemToHost failed, tensor_name: " << tensor_name;
      }
      TensorStatDump stat_dump("Parameter", node_name, task_id, stream_id, timestamp, false, 0, 0);
      (void)stat_dump.DumpTensorStatsToFile(node_name, dump_path, debugger);
    }
    if (dump_json_parser.IsTensorDump()) {
      DumpMemToFile(file_path, *addr, int_shapes, type, trans_flag);
    }
  }
}

void E2eDump::DumpParameters(const session::KernelGraph *graph, const std::string &dump_path,
                             const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.OutputNeedDump()) {
    return;
  }
  MS_LOG(INFO) << "Start e2e dump parameters";
  bool trans_flag = dump_json_parser.trans_flag();

  // dump parameters
  const auto &parameters = graph->inputs();
  for (auto &item : parameters) {
    DumpSingleAnfNode(item, PARAMETER_OUTPUT_INDEX, dump_path, trans_flag, debugger);
  }
}

void E2eDump::DumpConstantData(const session::KernelGraph *graph, uint32_t rank_id, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!IsDeviceTargetGPU() || !dump_json_parser.e2e_dump_enabled()) {
    return;
  }
  uint32_t graph_id = graph->graph_id();
  std::string cst_path = GenerateDumpPath(graph_id, rank_id, true);
  if (!Common::FileExists(cst_path)) {
    DumpConstantData(graph, cst_path, debugger);
  }
}

void E2eDump::DumpConstantData(const session::KernelGraph *graph, const std::string &cst_dump_path,
                               const Debugger *debugger) {
  // Dump constant to npy file
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  MS_LOG(INFO) << "DumpConstants. Current iteration is " << dump_json_parser.cur_dump_iter();
  MS_LOG(INFO) << "Current graph id is " << graph->graph_id();
  if (!dump_json_parser.OutputNeedDump()) {
    return;
  }
  const auto value_nodes = graph->graph_value_nodes();
  for (auto &item : value_nodes) {
    DumpSingleAnfNode(item, VALUE_NODE_OUTPUT_INDEX, cst_dump_path, false, debugger);
  }
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime.
 * Description: This function is for updating dump iteration for GPU and ascend old runtime.
 */
void E2eDump::UpdateIterOldRTDump(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  uint32_t graph_id = graph->graph_id();
  if (IsDeviceTargetGPU()) {
    if (starting_graph_id == INT32_MAX) {
      starting_graph_id = graph_id;
    } else if (starting_graph_id == graph_id && !MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
      // Update dump iter for mindrt runtime is done using UpdateIterGPUDump().
      // Update dump iter for GPU old runtime.
      dump_json_parser.UpdateDumpIter();
    }
    return;
  }
  // If device target is Ascend
  if (graph->IsDatasetGraph()) {
    MS_LOG(INFO) << "No need to update iteration for dataset graph.";
    return;
  }

  // In multi network scripts, dump iter is equal to the number of networks that have been executed so far.
  dump_json_parser.UpdateDumpIter();
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: This function is for updating dump iteration for GPU and ascend MindRT dump. Please note that dump with
 * dataset_sink_mode = True is not supported for GPU.
 */
void E2eDump::UpdateIterMindRTDump() {
  auto debugger = Debugger::GetInstance();
  // Dataset graph is always the first graph in the list when dataset_sink_mode is true.
  auto graph_list = debugger->GetStepGraphPtrList();
  if (graph_list.empty()) {
    MS_LOG(INFO) << "The graph list is empty.";
    return;
  }
  auto graph = graph_list[0];
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice && graph->IsDatasetGraph()) {
    MS_LOG(INFO) << "No need to update iteration for dataset graph.";
    return;
  }
  // update dump iter for GPU and kernel by kernel ascend dump.
  DumpJsonParser::GetInstance().UpdateDumpIter();
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Generates graph history files (dumping all the iteration numbers in which the graph was executed) for
 * the given graph and rank_id. If dataset_sink_mode is true for async dump in ascend, this function is called once per
 * each epoch and dumps all the iterations in the epoch to the graph history file.
 */
void E2eDump::DumpRunIter(const KernelGraphPtr &graph, uint32_t rank_id) {
  auto &json_parser = DumpJsonParser::GetInstance();
  if (!(json_parser.async_dump_enabled() || json_parser.e2e_dump_enabled())) {
    return;
  }
  bool sink_mode =
    (ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE || graph->IsDatasetGraph());
  auto iter_num = SizeToInt(LongToSize(ConfigManager::GetInstance().iter_num()));
  if (graph->IsDatasetGraph()) {
    MS_LOG(INFO) << "graph: " << graph->graph_id() << " is dataset graph, not creating graph history file.";
    return;
  }
  if (!Debugger::GetInstance()->GetAscendKernelByKernelFlag() && !IsDeviceTargetGPU() &&
      (graph->graph_id() != graph->root_graph_id())) {
    // when device target is ascend, we only dump graph run iter for the root graph.
    return;
  }
  std::string execution_order_path = json_parser.path() + "/rank_" + std::to_string(rank_id) + "/execution_order/";
  std::string graph_str =
    IsDeviceTargetGPU() ? std::to_string(graph->graph_id()) : std::to_string(graph->root_graph_id());
  std::string file_name_to_check = execution_order_path + "/ms_global_execution_order_graph_" + graph_str + ".csv";
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
  if (sink_mode && json_parser.async_dump_enabled() && !Debugger::GetInstance()->GetAscendKernelByKernelFlag()) {
    // for async dump when sink_mode = true, cur_dump_iter() = current_epoch
    // dump history for all iterations in the epoch
    Debugger::GetInstance()->UpdateGraphIterMap(graph->graph_id(), iter_num);
    auto graph_iter_map = Debugger::GetInstance()->GetGraphIterMap();
    auto step_per_epoch = IntToSize(graph_iter_map[graph->graph_id()]);
    for (size_t i = 0; i < step_per_epoch; i++) {
      auto step = (json_parser.cur_dump_iter() * step_per_epoch) + i;
      fout << (std::to_string(step) + "\n");
    }
  } else {
    fout << std::to_string(json_parser.cur_dump_iter()) + "\n";
  }
  fout.close();
  ChangeFileMode(file_name, S_IRUSR);
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: This function is for dumping the whole graph. It is used for old runtime in GPU and Ascend and
 * super-kernel mindRT in Ascend.
 */
void E2eDump::DumpData(const session::KernelGraph *graph, uint32_t rank_id, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  bool success = false;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  uint32_t graph_id = graph->graph_id();
  if (!dump_json_parser.e2e_dump_enabled()) {
    return;
  }

  if (dump_json_parser.GetIterDumpFlag()) {
    MS_LOG(INFO) << "Start e2e dump. Current iteration is " << dump_json_parser.cur_dump_iter();
    MS_LOG(INFO) << "Current graph id is " << graph_id;
    std::string dump_path = GenerateDumpPath(graph_id, rank_id);
    if (dump_json_parser.IsStatisticDump()) {
      (void)TensorStatDump::OpenStatisticsFile(dump_path);
    }
    DumpInput(graph, dump_path, debugger);
    DumpOutput(graph, dump_path, debugger);
    if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
      // Dump parameters for old runtime. For mindRT it is done in PostExecuteGraphDebugger.
      DumpParameters(graph, dump_path, debugger);
      // DumpConstantData for GPU old runtime.
      DumpConstantData(graph, rank_id, debugger);
    }
    if (dump_json_parser.IsStatisticDump()) {
      CsvWriter::GetInstance().CloseFile();
    }
    success = true;
  }

  if (success) {
    MS_LOG(DEBUG) << "E2eDump Dump Data completed!";
  } else {
    MS_LOG(DEBUG) << "E2eDump Dump has not occurred!";
  }
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: This function is for dumping a single node. It is used for mindrt in GPU and Ascend kernel-by-kernel.
 */
bool E2eDump::DumpSingleNodeData(const CNodePtr &node, uint32_t graph_id, uint32_t rank_id, const Debugger *debugger) {
  bool success = false;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (dump_json_parser.DumpEnabledForIter()) {
    std::string dump_path = GenerateDumpPath(graph_id, rank_id);
    DumpInputSingleNode(node, dump_path, debugger);
    DumpOutputSingleNode(node, dump_path, debugger);
    success = true;
  }
  return success;
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: This function is for dumping all the parameters in the current root graph for GPU, Ascend superkernel
 * (e2e dump) and Ascend kernel-by-kernel (e2e and async dump).
 */
void E2eDump::DumpParametersData(uint32_t rank_id, const Debugger *debugger) {
  uint32_t root_graph_id = debugger->GetCurrentRootGraphId();
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if ((dump_json_parser.async_dump_enabled() && !debugger->GetAscendKernelByKernelFlag()) ||
      (dump_json_parser.async_dump_enabled() && dump_json_parser.op_debug_mode() > 0)) {
    // Dump parameters for mindRT in async dump only for kernel by kernel mode.
    return;
  }
  if (dump_json_parser.DumpEnabledForIter()) {
    MS_LOG(INFO) << "DumpParameters. Current iteration is " << dump_json_parser.cur_dump_iter();
    MS_LOG(INFO) << "Current root graph id is " << root_graph_id;
    std::string dump_path = GenerateDumpPath(root_graph_id, rank_id);
    bool trans_flag = dump_json_parser.trans_flag();
    for (auto &item : debugger->GetParametersMindRT()) {
      DumpSingleParameterNode(item, dump_path, trans_flag, debugger);
    }
  }
}
}  // namespace mindspore
