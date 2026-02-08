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

#include "tools/data_dump/e2e_dump.h"

#include <unistd.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/utils/anfalgo.h"
#include "include/utils/common.h"
#include "utils/config_manager.h"
#include "utils/convert_utils.h"
#include "ir/tensor_new.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_dump_utils.h"
#include "tools/data_dump/common/csv_writer.h"
#include "tools/data_dump/debugger/debugger_utils.h"
#include "tools/data_dump/dump_json_parser.h"
#include "tools/data_dump/tensor_stat_dump.h"
#include "utils/file_utils.h"
#include "utils/ms_context.h"
#ifdef ENABLE_DEBUGGER
#include "ops/op_def.h"
#include "tools/data_dump/debug_services.h"
#include "tools/data_dump/debugger/debugger.h"
#include "tools/data_dump/tensor_load.h"
#endif

namespace mindspore {

TypeId ConvertStringToTypeId(const std::string &dtype) {
  const std::map<std::string, TypeId> kDbgDataTypeToStringMap = {
    {"bool", TypeId::kNumberTypeBool},        {"int8", TypeId::kNumberTypeInt16},
    {"int16", TypeId::kNumberTypeInt16},      {"int32", TypeId::kNumberTypeInt32},
    {"int64", TypeId::kNumberTypeInt64},      {"uint8", TypeId::kNumberTypeUInt8},
    {"uint16", TypeId::kNumberTypeUInt16},    {"uint32", TypeId::kNumberTypeUInt32},
    {"uint64", TypeId::kNumberTypeUInt64},    {"float16", TypeId::kNumberTypeFloat16},
    {"float32", TypeId::kNumberTypeFloat32},  {"float64", TypeId::kNumberTypeFloat64},
    {"bfloat16", TypeId::kNumberTypeBFloat16}};
  auto iter_type = kDbgDataTypeToStringMap.find(dtype);
  if (iter_type == kDbgDataTypeToStringMap.end()) {
    return TypeId::kTypeUnknown;
  }
  return iter_type->second;
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
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  return IsDeviceTargetGPU() || debugger->GetAscendKernelByKernelFlag();
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
    MS_VLOG(VL_DUMP) << "DumpTensorToFile Failed: path:" << file_path;
  }
#endif
}

void E2eDump::DumpOutputSingleNode(const CNodePtr &node, const std::string &dump_path, const Debugger *debugger,
                                   const DeviceContext *device_context) {
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
  DumpOutputImpl(node, trans_flag, dump_path, &kernel_name, debugger, device_context);
}

void E2eDump::DumpOutputImpl(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                             std::string *kernel_name, const Debugger *debugger, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(node);
  std::string stat_op_name = *kernel_name;
  GetFileKernelName(NOT_NULL(kernel_name));
  auto output_size = AnfAlgo::GetOutputTensorNum(node);
  std::vector<size_t> valid_indexes;
  if (device_context != nullptr) {
    valid_indexes = GetValidDumpIndex(node, output_size, false, device_context);
  }
  for (size_t j = 0; j < output_size; ++j) {
    if ((device_context != nullptr &&
         std::find(valid_indexes.begin(), valid_indexes.end(), j) == valid_indexes.end())) {
      continue;
    }
    std::string node_name = GetKernelNodeName(node);
    auto type = common::AnfAlgo::GetOutputInferDataType(node, j);
    std::string op_type = common::AnfAlgo::GetCNodeName(node);
    std::string op_name = *kernel_name;
    uint32_t task_id = 0;
    uint32_t stream_id = 0;
    if (IsDeviceTargetAscend()) {
      stream_id = AnfAlgo::GetStreamId(node);
    }
    uint64_t timestamp = Common::GetTimeStamp();
    std::string file_path = dump_path + '/' + op_type + '.' + op_name + '.' + std::to_string(task_id) + '.' +
                            std::to_string(stream_id) + '.' + std::to_string(timestamp) + ".output." +
                            std::to_string(j);
    // Tensor must be saved before statistic. Because the tensor would be changed in DumpTensorStatsToFile when data
    // type is int4, if tensor saved after statistic, the tensor value would be wrong.
    if (DumpJsonParser::GetInstance().IsTensorDump()) {
      if (IsMindRTKernelByKernel()) {
        DumpMemFromTensorLoaderToFile(debugger, file_path, node_name, j);
      } else {
        if (!AnfAlgo::OutputAddrExist(node, j, true)) {
          continue;
        }
        auto kt = AnfAlgo::GetOutputKernelTensor(node, j);
        MS_EXCEPTION_IF_NULL(kt);
        ShapeVector int_shapes;
        GetDumpIntShape(node, j, NOT_NULL(&int_shapes), trans_flag);
        DumpMemToFile(file_path, kt, int_shapes, type, trans_flag);
      }
    }
    if (DumpJsonParser::GetInstance().IsStatisticDump() && IsMindRTKernelByKernel()) {
      TensorStatDump stat_dump(op_type, stat_op_name, task_id, stream_id, timestamp, false, j, j);
      (void)stat_dump.DumpTensorStatsToFile(node_name, dump_path, debugger);
    }
  }
}

void E2eDump::DumpInputSingleNode(const CNodePtr &node, const std::string &dump_path, const Debugger *debugger,
                                  const DeviceContext *device_context) {
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
  DumpInputImpl(node, trans_flag, dump_path, &kernel_name, debugger, device_context);
}

tensor::TensorPtr GetConvertedTensorFromIgnoredInput(const AnfNodePtr node, size_t idx) {
  auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, idx);
  MS_EXCEPTION_IF_NULL(kernel_with_index.first);
  if (!kernel_with_index.first->isa<ValueNode>()) {
    MS_VLOG(VL_DUMP) << "Prim init args is not value node for idx: " << idx;
    return nullptr;
  }
  std::shared_ptr<tensor::Tensor> converted_tensor = nullptr;
  auto input = kernel_with_index.first->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(input);
  const auto &input_value = input->value();
  MS_EXCEPTION_IF_NULL(input_value);
  if (input_value->isa<Scalar>()) {
    converted_tensor = ScalarToTensor(input_value->cast<ScalarPtr>());
  } else if (input_value->isa<ValueSequence>()) {
    converted_tensor = SequenceToTensor(input_value->cast<ValueSequencePtr>());
  } else {
    MS_VLOG(VL_DUMP) << "Prim init args is not scalar or valuesequence for idx: " << idx;
  }
  return converted_tensor;
}

void E2eDump::DumpArgsSingleNode(const CNodePtr &node, const std::string &dump_path, const Debugger *debugger) {
  auto op_name = GetKernelNodeName(node);
  int start_index = static_cast<int>(op_name.rfind('/')) + 1;
  int end_index = static_cast<int>(op_name.rfind('-'));
  if (end_index == -1) {
    end_index = static_cast<int>(op_name.length());
  }
  std::string op_t = op_name.substr(start_index, end_index - start_index);
  auto op_def = mindspore::ops::GetOpDef(op_t);
  nlohmann::json json;
  if (!op_def) {
    auto prim_node = GetCNodePrimitive(node);
    if (prim_node != nullptr) {
      auto prim_attrs = prim_node->attrs();
      for (const auto &entry : prim_attrs) {
        json[entry.first] = entry.second->ToString();
      }
    }
  } else {
    int idx = 0;
    for (const auto &op_arg : op_def->args_) {
      ++idx;
      if (op_arg.as_init_arg_) {
        auto input_kernel = node->input(idx);
        std::string input_kernel_name = GetKernelNodeName(input_kernel);
        string input_tensor_name = input_kernel_name + ':' + "0";
        auto arg_name = op_arg.arg_name_;
        auto t_data = debugger->GetTensor(input_tensor_name);
        std::shared_ptr<tensor::Tensor> converted_tensor = nullptr;
        if (t_data == nullptr) {
          MS_VLOG(VL_DUMP) << "Dump args single node input idx: " << idx
                           << ", use host value for node: " << node->fullname_with_scope();
          converted_tensor = GetConvertedTensorFromIgnoredInput(node, idx - 1);
          if (converted_tensor == nullptr) {
            continue;
          }
        } else {
          std::string type = t_data->GetTypeString();
          converted_tensor =
            tensor::from_buffer(ConvertStringToTypeId(type), t_data->GetShape(),
                                static_cast<void *>(const_cast<char *>(t_data->GetDataPtr())), t_data->GetByteSize());
        }

        json[arg_name] = converted_tensor->DataToString(false);
      }
    }
  }

  std::string scope_name = node->fullname_with_scope();
  std::replace(scope_name.begin(), scope_name.end(), '.', '_');
  std::replace(scope_name.begin(), scope_name.end(), '/', '_');

  constexpr int kJsonIndent = 4;
  std::string file_path = dump_path + op_t + "." + scope_name + ".json";
  auto realpath = Common::CreatePrefixPath(file_path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get realpath failed, path=" << file_path;
    return;
  }

  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream outFile(realpath.value());
  if (!outFile.is_open()) {
    MS_LOG(ERROR) << "Could not open file" << file_path;
    return;
  }
  outFile << json.dump(kJsonIndent);
  outFile.close();
  ChangeFileMode(realpath.value(), S_IRUSR);
}

void E2eDump::DumpInputImpl(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                            std::string *kernel_name, const Debugger *debugger, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(node);
  std::string stat_op_name = *kernel_name;
  GetFileKernelName(NOT_NULL(kernel_name));
  auto input_size = common::AnfAlgo::GetInputTensorNum(node);
  std::vector<size_t> valid_indexes;
  if (device_context != nullptr) {
    valid_indexes = GetValidDumpIndex(node, input_size, true, device_context);
  }
  for (size_t j = 0; j < input_size; ++j) {
    if (device_context != nullptr && std::find(valid_indexes.begin(), valid_indexes.end(), j) == valid_indexes.end()) {
      continue;
    }
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, j);
    auto input = kernel_with_index.first;
    auto index = kernel_with_index.second;
    std::string node_name = GetKernelNodeName(node);
    size_t slot = j;
    if (IsMindRTKernelByKernel()) {
      auto input_kernel = node->input(j + 1);
      std::string input_kernel_name = GetKernelNodeName(input_kernel);
      node_name = input_kernel_name;
      slot = 0;
    }
    auto type = common::AnfAlgo::GetOutputInferDataType(input, index);
    std::string op_type = common::AnfAlgo::GetCNodeName(node);
    std::string op_name = *kernel_name;
    uint64_t timestamp = Common::GetTimeStamp();
    uint32_t task_id = 0;
    uint32_t stream_id = 0;
    if (IsDeviceTargetAscend()) {
      stream_id = AnfAlgo::GetStreamId(node);
    }
    std::string file_path = dump_path + '/' + op_type + '.' + op_name + '.' + std::to_string(task_id) + '.' +
                            std::to_string(stream_id) + '.' + std::to_string(timestamp) + ".input." + std::to_string(j);
    if (DumpJsonParser::GetInstance().IsTensorDump()) {
      if (IsMindRTKernelByKernel()) {
        DumpMemFromTensorLoaderToFile(debugger, file_path, node_name, slot);
      } else {
        ShapeVector int_shapes;
        if (!AnfAlgo::OutputAddrExist(input, index)) {
          continue;
        }
        auto kt = AnfAlgo::GetOutputKernelTensor(input, index);
        MS_EXCEPTION_IF_NULL(kt);
        GetDumpIntShape(input, index, NOT_NULL(&int_shapes), trans_flag);
        DumpMemToFile(file_path, kt, int_shapes, type, trans_flag);
      }
    }
    if (DumpJsonParser::GetInstance().IsStatisticDump() && IsMindRTKernelByKernel()) {
      TensorStatDump stat_dump(op_type, stat_op_name, task_id, stream_id, timestamp, true, j, slot);
      (void)stat_dump.DumpTensorStatsToFile(node_name, dump_path, debugger);
    }
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
  const std::string cst_prefix = "Default_";
  if (anf_node->isa<ValueNode>()) {
    if (dump_name.find(cst_prefix) == std::string::npos) {
      MS_VLOG(VL_DUMP) << "Incorrect constant format: " << dump_name;
      return;
    }
    dump_name = node_name.substr(cst_prefix.length());
    trans_flag = false;
  }
  // check if output address exists, if not, return;
  if (!AnfAlgo::OutputAddrExist(anf_node, output_index)) {
    return;
  }
  auto kt = AnfAlgo::GetOutputKernelTensor(anf_node, output_index);
  MS_EXCEPTION_IF_NULL(kt);
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
      bool ret = LoadMemToHost(AnfAlgo::GetOutputKernelTensor(anf_node, output_index).get(), tensor_name, format,
                               int_shapes, type, 0, true, root_graph_id, false, true);
      if (!ret) {
        MS_LOG(ERROR) << "LoadMemToHost failed, tensor_name: " << tensor_name;
      } else {
        TensorStatDump stat_dump("Parameter", dump_name, task_id, stream_id, timestamp, false, 0, 0);
        (void)stat_dump.DumpTensorStatsToFile(node_name, dump_path, debugger);
      }
    }
    if (dump_json_parser.IsTensorDump()) {
      DumpMemToFile(file_path, kt, int_shapes, type, trans_flag);
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
  auto kt = GetParameterInfo(anf_node, NOT_NULL(&int_shapes), NOT_NULL(&type), NOT_NULL(&device_type));
  if (kt == nullptr || kt->device_address() == nullptr || kt->device_address()->GetPtr() == nullptr) {
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
      bool ret = LoadMemToHost(kt.get(), tensor_name, format, int_shapes, type, 0, true, root_graph_id, false, true);
      if (!ret) {
        MS_LOG(ERROR) << "LoadMemToHost failed, tensor_name: " << tensor_name;
      }
      TensorStatDump stat_dump("Parameter", node_name, task_id, stream_id, timestamp, false, 0, 0);
      (void)stat_dump.DumpTensorStatsToFile(node_name, dump_path, debugger);
    }
    if (dump_json_parser.IsTensorDump()) {
      DumpMemToFile(file_path, kt, int_shapes, type, trans_flag);
    }
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
  MS_VLOG(VL_DUMP) << "DumpConstants. Current iteration is " << dump_json_parser.cur_dump_iter();
  MS_VLOG(VL_DUMP) << "Current graph id is " << graph->graph_id();
  if (!dump_json_parser.OutputNeedDump()) {
    return;
  }
  const auto value_nodes = graph->graph_value_nodes();
  for (auto &item : value_nodes) {
    DumpSingleAnfNode(item, kValueNodeOutputIndex, cst_dump_path, false, debugger);
  }
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
  MS_EXCEPTION_IF_NULL(debugger);
  // Dataset graph is always the first graph in the list when dataset_sink_mode is true.
  auto graph_list = debugger->GetStepGraphPtrList();
  if (graph_list.empty()) {
    MS_VLOG(VL_DUMP) << "The graph list is empty.";
    return;
  }
  auto graph = graph_list[0];
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice && graph->IsDatasetGraph()) {
    MS_VLOG(VL_DUMP) << "No need to update iteration for dataset graph.";
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
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string backend = context->backend_policy();
  if (backend == "ge") {
    MS_VLOG(VL_DUMP) << "On Ascend910B or Ascend910_93 platform, execution_order is not support to dump.";
    return;
  }
  bool sink_mode =
    (ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE || graph->IsDatasetGraph());
  auto iter_num = SizeToInt(LongToSize(ConfigManager::GetInstance().iter_num()));
  if (graph->IsDatasetGraph()) {
    MS_VLOG(VL_DUMP) << "graph: " << graph->graph_id() << " is dataset graph, not creating graph history file.";
    return;
  }
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (!debugger->GetAscendKernelByKernelFlag() && !IsDeviceTargetGPU() &&
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
  if (sink_mode && json_parser.async_dump_enabled() && !debugger->GetAscendKernelByKernelFlag()) {
    // for async dump when sink_mode = true, cur_dump_iter() = current_epoch
    // dump history for all iterations in the epoch
    debugger->UpdateGraphIterMap(graph->graph_id(), iter_num);
    auto graph_iter_map = debugger->GetGraphIterMap();
    auto step_per_epoch = IntToSize(graph_iter_map[graph->graph_id()]);
    for (size_t i = 0; i < step_per_epoch; i++) {
      auto step = (json_parser.cur_dump_iter() * step_per_epoch) + i;
      fout << (std::to_string(step) + "\n");
    }
  } else {
    fout << std::to_string(json_parser.cur_dump_iter()) << "\n";
  }
  fout.close();
  ChangeFileMode(file_name, S_IRUSR);
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: This function is for dumping a single node. It is used for mindrt in GPU and Ascend kernel-by-kernel.
 */
bool E2eDump::DumpSingleNodeData(const CNodePtr &node, uint32_t graph_id, uint32_t rank_id, const Debugger *debugger,
                                 const DeviceContext *device_context) {
  bool success = false;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (dump_json_parser.DumpEnabledForIter()) {
    std::string dump_path = GenerateDumpPath(graph_id, rank_id);
    DumpInputSingleNode(node, dump_path, debugger, device_context);
    DumpOutputSingleNode(node, dump_path, debugger, device_context);
    if (dump_json_parser.save_args_flag()) {
      DumpArgsSingleNode(node, dump_path, debugger);
    }
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
    MS_VLOG(VL_DUMP) << "DumpParameters. Current iteration is " << dump_json_parser.cur_dump_iter();
    MS_VLOG(VL_DUMP) << "Current root graph id is " << root_graph_id;
    std::string dump_path = GenerateDumpPath(root_graph_id, rank_id);
    bool trans_flag = dump_json_parser.trans_flag();
    for (auto &item : debugger->GetParametersMindRT()) {
      DumpSingleParameterNode(item, dump_path, trans_flag, debugger);
    }
  }
}
}  // namespace mindspore
