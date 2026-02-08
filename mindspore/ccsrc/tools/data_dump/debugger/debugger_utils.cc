/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#include "tools/data_dump/debugger/debugger_utils.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "include/backend/common/device_address_utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "include/utils/anfalgo.h"
#include "include/utils/common.h"
#include "utils/config_manager.h"
#include "ir/tensor_new.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_dump_utils.h"
#include "proto/debug_graph.pb.h"
#include "runtime/hardware_abstract/utils.h"
#include "tools/data_dump/common/csv_writer.h"
#include "tools/data_dump/debugger/debugger.h"
#include "tools/data_dump/device_statistic/kernel_launcher.h"
#include "tools/data_dump/dump_json_parser.h"
#include "tools/data_dump/dump_utils.h"
#include "tools/data_dump/e2e_dump.h"
#include "tools/data_dump/overflow_counter.h"
#include "tools/data_dump/tensor_info_collect.h"
#include "tools/data_dump/tensor_stat_dump.h"
#include "tools/data_dump/tensor_statistic.h"
#include "tools/data_dump/utils.h"

constexpr int kFailure = 1;
constexpr int kQint4ShapeModify = 2;
constexpr auto kInput = "input";
constexpr auto kOutput = "output";

using mindspore::kernel::AddressPtr;
using AddressPtrList = std::vector<mindspore::kernel::AddressPtr>;
using KernelGraph = mindspore::session::KernelGraph;
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;
using KernelTensorPtr = std::shared_ptr<KernelTensor>;

namespace mindspore {
using mindspore::TensorInfoCommForDump;
using mindspore::TensorInfoForDump;

inline mindspore::tensor::TensorPtr KernelTensor2Tensor(KernelTensorPtr, const TypeId, const ShapeVector &);
inline string TensorToString(mindspore::tensor::TensorPtr tensor);

namespace {
std::vector<size_t> GetIgnoredIndexesForInput(const CNodePtr &cnode, const DeviceContext *device_context) {
  std::vector<size_t> ignored_indexes;
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  if (kernel_mod != nullptr) {
    MS_EXCEPTION_IF_NULL(device_context);
    auto kernel_executor = device_context->GetKernelExecutor();
    MS_EXCEPTION_IF_NULL(kernel_executor);
    ignored_indexes = kernel_executor->GetLaunchIgnoredInputAddressIdx(cnode);
  }
  return ignored_indexes;
}

std::vector<size_t> GetIgnoredIndexesForOutput(const CNodePtr &cnode, const DeviceContext *device_context) {
  std::vector<size_t> ignored_indexes;
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  static string ignore_useless_output_env = common::GetEnv("MINDSPORE_DUMP_IGNORE_USELESS_OUTPUT", "1");
  static bool warn_once = true;
  if (warn_once && ignore_useless_output_env != "0" && ignore_useless_output_env != "1") {
    MS_LOG(WARNING) << "Invalid value for environment variable 'MINDSPORE_DUMP_IGNORE_USELESS_OUTPUT'. "
                    << "Expected value is either '0' or '1', but got '" << ignore_useless_output_env << "'. "
                    << "The default value '1' will be used. Please correct the setting to avoid this warning.";
    warn_once = false;
  }
  static bool enable_useless_output = ignore_useless_output_env == "0";
  static bool log_once = true;
  if (log_once) {
    MS_VLOG(VL_DUMP) << "MINDSPORE_DUMP_IGNORE_USELESS_OUTPUT=" << ignore_useless_output_env << ". "
                     << "Invalid outputs will " << (enable_useless_output ? "" : "not ") << "be dumped.";
    log_once = false;
  }
  if (!enable_useless_output && kernel_mod != nullptr) {
    ignored_indexes = kernel_mod->GetUseLessOutputIdx();
  }
  return ignored_indexes;
}
};  // namespace

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU, Ascend.
 * Runtime category: MindRT.
 * Description: Get Valid Tensor indexes.
 */
std::vector<size_t> GetValidDumpIndex(const CNodePtr &cnode, size_t index_size, bool is_input,
                                      const DeviceContext *device_context, const std::vector<KernelTensor *> &tensors) {
  std::vector<size_t> valid_indexes;
  valid_indexes.reserve(index_size);
  std::vector<size_t> ignored_indexes =
    is_input ? GetIgnoredIndexesForInput(cnode, device_context) : GetIgnoredIndexesForOutput(cnode, device_context);
  std::set<size_t> ignored_indexes_set(ignored_indexes.begin(), ignored_indexes.end());
  for (size_t index = 0; index < index_size; ++index) {
    if (ignored_indexes_set.find(index) != ignored_indexes_set.end()) {
      MS_VLOG(VL_DUMP) << cnode->fullname_with_scope() << (is_input ? " input" : " output") << ", index " << index
                       << " this tensor is set to be ignored, currently skipped.";
      continue;
    }
    if (index >= tensors.size()) {
      valid_indexes.push_back(index);
      continue;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(index < tensors.size(), "Index out of range. Index: " + std::to_string(index) +
                                                         ", tensors size: " + std::to_string(tensors.size()));
    auto tensor = tensors[index];
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->device_ptr() == nullptr) {
      MS_VLOG(VL_DUMP) << cnode->fullname_with_scope() << (is_input ? " input" : " output") << ", index " << index
                       << " deviceaddress is nullptr.";
      continue;
    }
    if (tensor->tensor_storage_info() && tensor->dtype_id() == kNumberTypeInt4) {
      MS_LOG(WARNING) << cnode->fullname_with_scope() << (is_input ? " input" : " output") << ", index " << index
                      << " deviceaddress is not contiguous and the tensor type is int4 . Dump currently does not "
                         "support non-contiguous int4 data and is "
                         "currently skipped.";
      continue;
    }
    // Check if the current dump mode is exception dump.
    bool is_exception_dump = DumpJsonParser::GetInstance().op_debug_mode() == DumpJsonParser::DUMP_LITE_EXCEPTION;
    if (tensor->GetDeviceType() != device_context->GetDeviceType() && !is_exception_dump) {
      MS_LOG(WARNING) << cnode->fullname_with_scope() << (is_input ? " input" : " output") << ", index " << index
                      << " tensor's target device(" << tensor->GetDeviceType() << ") is different from running device("
                      << device_context->GetDeviceType() << "), currently skipped.";
      continue;
    }
    valid_indexes.push_back(index);
  }
  return valid_indexes;
}

bool IsDeviceTargetGPU() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice;
}

bool GetTransFlag() {
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (IsDeviceTargetGPU()) {
    return true;
  }
  return DumpJsonParser::GetInstance().trans_flag();
}

uint32_t GetSampleMode() {
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (IsDeviceTargetGPU()) {
    return 0;
  }
  return DumpJsonParser::GetInstance().sample_mode();
}

uint32_t GetSampleNum() {
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (IsDeviceTargetGPU() || !GetSampleMode()) {
    return 0;
  }
  return DumpJsonParser::GetInstance().sample_num();
}

size_t ModifySize(const TypeId &host_type, const size_t &host_size) {
  if (host_type == kNumberTypeInt4) {
    return host_size / kQint4ShapeModify;
  }
  return host_size;
}

inline TypeId GetInputKernelType(const AnfNodePtr &input_kernel, bool trans_flag) {
  auto device_type = AnfAlgo::GetOutputDeviceDataType(input_kernel, kParameterOutputIndex);
  auto host_type = common::AnfAlgo::GetOutputInferDataType(input_kernel, kParameterOutputIndex);
  auto type = trans_flag ? host_type : device_type;
  return type;
}

inline TypeId GetOutputKernelType(const CNodePtr &cnode, size_t j, bool trans_flag) {
  auto device_type = AnfAlgo::GetOutputDeviceDataType(cnode, j);
  auto host_type = common::AnfAlgo::GetOutputInferDataType(cnode, j);
  auto type = trans_flag ? host_type : device_type;
  return type;
}

inline ShapeVector SampleDumpShape(const ShapeVector &dump_shape) {
  auto sample_mode = GetSampleMode();
  auto sample_num = GetSampleNum();
  if (sample_mode == DumpJsonParser::DUMP_HEAD_AND_TAIL && SizeOf(dump_shape) > sample_num) {
    ShapeVector sample_shape = {sample_num};
    return sample_shape;
  }
  return dump_shape;
}

inline ShapeVector GetOutputKernelShapeVec(const CNodePtr &cnode, KernelTensor *kernel_tensor, size_t j,
                                           bool trans_flag) {
  auto dump_shape = kernel_tensor->GetShapeVector();
  if (!trans_flag) {
    dump_shape = AnfAlgo::GetOutputDeviceShape(cnode, j, dump_shape);
  }
  dump_shape = SampleDumpShape(dump_shape);
  return dump_shape;
}

inline ShapeVector GetInputKernelShapeVec(const AnfNodePtr &input_kernel, KernelTensor *kernel_tensor, size_t j,
                                          bool trans_flag) {
  auto dump_shape = kernel_tensor->GetShapeVector();
  if (!trans_flag) {
    dump_shape = AnfAlgo::GetOutputDeviceShape(input_kernel, kParameterOutputIndex, dump_shape);
  }
  dump_shape = SampleDumpShape(dump_shape);
  return dump_shape;
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU, Ascend.
 * Runtime category: MindRT.
 * Description: Get kernel inputs from device_tensors and load the inputs from device to host.
 */
void LoadInputs(const CNodePtr &cnode, std::vector<KernelTensor *> kernel_tensors, uint32_t root_graph_id,
                const DeviceContext *device_context, const bool trans_flag, const uint32_t sample_mode,
                const uint32_t sample_num, const bool async_copy) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(device_context);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  std::vector<size_t> ignored_address;
  if (kernel_mod != nullptr) {
    MS_EXCEPTION_IF_NULL(device_context);
    auto kernel_executor = device_context->GetKernelExecutor();
    MS_EXCEPTION_IF_NULL(kernel_executor);
    ignored_address = kernel_executor->GetLaunchIgnoredInputAddressIdx(cnode);
  }

  auto input_size = kernel_tensors.size();
  std::vector<size_t> valid_indexes = GetValidDumpIndex(cnode, input_size, true, device_context, kernel_tensors);
  for (size_t index : valid_indexes) {
    auto input_kernel = cnode->input(index + 1);
    std::string input_kernel_name = GetKernelNodeName(input_kernel);
    auto type = GetInputKernelType(input_kernel, trans_flag);
    // For example, this happens with the Depend op
    if (type == kMetaTypeNone) {
      continue;
    }
    auto host_format = kOpFormat_DEFAULT;
    auto device_format =
      E2eDump::IsDeviceTargetGPU() ? kOpFormat_DEFAULT : AnfAlgo::GetOutputFormat(input_kernel, kParameterOutputIndex);

    string input_tensor_name = input_kernel_name + ':' + "0";
    MS_EXCEPTION_IF_NULL(kernel_tensors[index]);
    auto dump_shape = GetInputKernelShapeVec(input_kernel, kernel_tensors[index], index, trans_flag);

    auto ret = LoadMemToHost(kernel_tensors[index], input_tensor_name, host_format, dump_shape, type, 0, true,
                             root_graph_id, false, trans_flag, async_copy);
    if (!ret) {
      MS_LOG(WARNING) << "LoadMemToHost failed: tensor_name:" << input_tensor_name << ", host_format:" << host_format
                      << ", device_format:" << device_format << ".";
    }
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU, Ascend.
 * Runtime category: MindRT.
 * Description: Get kernel outputs from device_tensors and load the inputs from device to host.
 */
void LoadOutputs(const CNodePtr &cnode, std::vector<KernelTensor *> kernel_tensors, uint32_t root_graph_id,
                 const DeviceContext *device_context, const bool trans_flag, const uint32_t sample_mode,
                 const uint32_t sample_num) {
  auto output_size = AnfAlgo::GetOutputTensorNum(cnode);
  auto node_name = common::AnfAlgo::GetCNodeName(cnode);
  std::string kernel_name = GetKernelNodeName(cnode);
  std::vector<size_t> valid_indexes = GetValidDumpIndex(cnode, output_size, false, device_context, kernel_tensors);
  for (size_t index : valid_indexes) {
    auto type = GetOutputKernelType(cnode, index, trans_flag);
    // For example, this happens with the Depend op
    if (type == kMetaTypeNone) {
      continue;
    }

    auto host_format = kOpFormat_DEFAULT;
    auto device_format = E2eDump::IsDeviceTargetGPU() ? kOpFormat_DEFAULT : AnfAlgo::GetOutputFormat(cnode, index);

    string tensor_name = kernel_name + ':' + std::to_string(index);
    MS_EXCEPTION_IF_NULL(kernel_tensors[index]);
    auto dump_shape = GetOutputKernelShapeVec(cnode, kernel_tensors[index], index, trans_flag);

    auto ret = LoadMemToHost(kernel_tensors[index], tensor_name, host_format, dump_shape, type, index, false,
                             root_graph_id, false, trans_flag);
    if (!ret) {
      MS_LOG(WARNING) << "LoadMemToHost failed: tensor_name:" << tensor_name << ", host_format:" << host_format
                      << ", device_format:" << device_format << ".!";
    }
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Returns true if the node needs to be read for Dump or online debugger. This function is used by GPU
 * and Ascend kernel-by-kernel mindRT.
 */
bool CheckReadData(const CNodePtr &cnode) {
  auto debugger = Debugger::GetInstance();
  if (!debugger) {
    return false;
  }
  bool read_data = false;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool dump_enabled = dump_json_parser.DumpEnabledForIter();
  std::string kernel_name = GetKernelNodeName(cnode);
  if (dump_enabled) {
    if (dump_json_parser.NeedDump(kernel_name)) {
      read_data = true;
    }
  }
  MS_LOG(DEBUG) << cnode->fullname_with_scope() << " need dump " << read_data;
  return read_data;
}

bool CheckOverFlow(const DeviceContext *device_context, std::vector<KernelTensor *> output_kernel_tensors) {
  if (output_kernel_tensors.empty()) {
    return false;
  }
  const auto &stream_id = output_kernel_tensors[0]->stream_id();

  uint32_t set_overflow_num = DumpJsonParser::GetInstance().overflow_number();
  uint32_t overflow_cont = OverflowCounter::GetInstance().getCount();
  bool is_overflow = false;
  bool sync_ok = device_context->device_res_manager_->SyncAllStreams();
  if (!sync_ok) {
    MS_LOG(EXCEPTION) << "Sync stream error! Overflow check op launcher failed";
  }
  if (set_overflow_num == 0) {
    is_overflow = datadump::CalCheckOverflow(device_context, output_kernel_tensors, stream_id);
  } else if (overflow_cont < set_overflow_num) {
    is_overflow = datadump::CalCheckOverflow(device_context, output_kernel_tensors, stream_id);
    if (is_overflow) {
      OverflowCounter::GetInstance().addCount();
    }
  }
  return is_overflow;
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Load inputs and outputs of the given node if needed and dump them if dump is enabled, then it performs
 * PostExecuteNode function on the given node for GPU.
 */
void ReadDataAndDump(const CNodePtr &cnode, std::vector<KernelTensor *> input_kernel_tensors,
                     std::vector<KernelTensor *> output_kernel_tensors, const DeviceContext *device_context,
                     const bool abnormal_dump) {
  auto debugger = Debugger::GetInstance();
  if (!debugger) {
    return;
  }
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (dump_json_parser.op_debug_mode() == DumpJsonParser::DUMP_BOTH_OVERFLOW) {
    auto output_size = output_kernel_tensors.size();
    std::vector<size_t> valid_indexes =
      GetValidDumpIndex(cnode, output_size, false, device_context, output_kernel_tensors);
    std::vector<KernelTensor *> valid_output_tensors;
    std::transform(valid_indexes.begin(), valid_indexes.end(), std::back_inserter(valid_output_tensors),
                   [&output_kernel_tensors](auto index) { return output_kernel_tensors[index]; });
    if (!CheckOverFlow(device_context, valid_output_tensors)) {
      return;
    }
  }
  auto kernel_graph = std::dynamic_pointer_cast<KernelGraph>(cnode->func_graph());
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto root_graph_id = kernel_graph->root_graph_id();
  bool trans_flag = GetTransFlag();
  uint32_t sample_mode = GetSampleMode();
  uint32_t sample_num = GetSampleNum();
  if (dump_json_parser.InputNeedDump()) {
    if (DumpJsonParser::GetInstance().IsDeviceCalcStats()) {
      datadump::DumpKernelTensorStats(device_context, input_kernel_tensors, true, cnode, root_graph_id);
    } else {
      bool async_copy = !abnormal_dump;
      LoadInputs(cnode, input_kernel_tensors, root_graph_id, device_context, trans_flag, sample_mode, sample_num,
                 async_copy);
    }
  }
  if (dump_json_parser.OutputNeedDump()) {
    if (DumpJsonParser::GetInstance().IsDeviceCalcStats()) {
      datadump::DumpKernelTensorStats(device_context, output_kernel_tensors, false, cnode, root_graph_id);
    } else if (!abnormal_dump) {
      LoadOutputs(cnode, output_kernel_tensors, root_graph_id, device_context, trans_flag, sample_mode, sample_num);
    }
  }
  // Dump kernel
  if (!DumpJsonParser::GetInstance().IsDeviceCalcStats()) {
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto graph_id = kernel_graph->graph_id();
    // for GPU, nodes are dumped in graph_id directory.
    if (IsDeviceTargetGPU()) {
      debugger->DumpSingleNode(cnode, graph_id);
    } else {
      // for Ascend, node are dumped in root_graph_id directory.
      debugger->DumpSingleNode(cnode, root_graph_id, device_context);
    }
    debugger->ClearCurrentData();
  }
}

inline std::shared_ptr<TensorData> PrepareStatTensorData(mindspore::tensor::TensorPtr out_tensor,
                                                         const TensorInfoForDump &tensor_info) {
  std::shared_ptr<TensorData> tensor_data = std::make_shared<TensorData>();
  tensor_data->SetTensor(out_tensor);
  tensor_data->SetDataPtr(static_cast<char *>(out_tensor->data_c()));
  auto byte_size = LongToSize(out_tensor->DataNBytes());
  if (tensor_info.host_type == kNumberTypeInt4) {
    uint32_t int4_nums_per_byte = 2;
    byte_size = byte_size / int4_nums_per_byte;
  }
  tensor_data->SetByteSize(byte_size);
  tensor_data->SetType(tensor_info.host_type);
  tensor_data->SetShape(out_tensor->shape());
  tensor_data->SetFormat(tensor_info.format);
  return tensor_data;
}

void DumpTensorToFile(std::string file_path, mindspore::tensor::TensorPtr out_tensor, TypeId host_type,
                      size_t host_size, ShapeVector host_shape) {
  if (host_type == kNumberTypeInt4) {
    auto int8_tensor = tensor::from_spec(TypeId::kNumberTypeInt8, host_shape, device::DeviceType::kCPU);
    bool split_succeed =
      SplitInt8ToInt4x2(out_tensor->data_c(), host_size, int8_tensor->data_c(), int8_tensor->DataSize());
    if (!split_succeed) {
      return;
    }
    DumpJsonParser::DumpToFile(file_path, int8_tensor->data_c(), int8_tensor->Size(), int8_tensor->shape_c(),
                               static_cast<TypeId>(int8_tensor->data_type_c()));
  } else if (host_type == TypeId::kNumberTypeBFloat16) {
    std::shared_ptr<tensor::Tensor> float32_tensor =
      std::make_shared<tensor::Tensor>(*out_tensor, TypeId::kNumberTypeFloat32);
    DumpJsonParser::DumpToFile(file_path, float32_tensor->data_c(), float32_tensor->Size(), float32_tensor->shape_c(),
                               static_cast<TypeId>(float32_tensor->data_type_c()));
  } else {
    DumpJsonParser::DumpToFile(file_path, out_tensor->data_c(), host_size, host_shape, host_type);
  }
}

KernelTensorPtr HandleOverflow(const std::vector<TensorInfoForDump> &tensor_info_list,
                               const DeviceContext *device_context, uint32_t stream_id,
                               const TensorInfoCommForDump &tensor_info_comm, uint32_t set_overflow_num) {
  if (OverflowCounter::GetInstance().getCount() >= set_overflow_num && set_overflow_num != 0) {
    return nullptr;
  }

  std::vector<KernelTensor *> kernel_tensors;
  for (const auto &tensor_info : tensor_info_list) {
    if (tensor_info.io == kOutput) {
      kernel_tensors.push_back(tensor_info.kernel_tensor);
    }
  }
  return datadump::CalCheckOverflowAsync(device_context, kernel_tensors, stream_id);
}

bool ProcessOverflow(const KernelTensorPtr &overflow_kernel_tensor, uint32_t set_overflow_num) {
  mindspore::tensor::TensorPtr my_overflow =
    KernelTensor2Tensor(overflow_kernel_tensor, kNumberTypeBool, ShapeVector());
  bool is_overflow = (TensorToString(my_overflow) == "True");
  if (is_overflow && (set_overflow_num == 0 || OverflowCounter::GetInstance().getCount() < set_overflow_num)) {
    OverflowCounter::GetInstance().addCount();
  } else {
    is_overflow = false;
  }
  return is_overflow;
}

ShapeVector GetOriOutputTensorShape(const TensorInfoForDump &tensor_info, TypeId host_type) {
  auto tensor_storage_info = tensor_info.kernel_tensor->tensor_storage_info();
  auto host_shape = tensor_storage_info == nullptr ? tensor_info.host_shape : tensor_storage_info->ori_shape;
  if (host_type == kNumberTypeInt4 && !GetSampleNum()) {
    constexpr int64_t kNumber2 = 2;
    host_shape.back() *= kNumber2;
  }
  return host_shape;
}

void LaunchDumpCallback(const std::vector<TensorInfoForDump> &tensor_info_list, const DeviceContext *device_context,
                        uint32_t stream_id, const TensorInfoCommForDump &tensor_info_comm) {
  bool dump_tensor = DumpJsonParser::GetInstance().IsTensorDump();
  bool overflow_flag = (DumpJsonParser::GetInstance().op_debug_mode() == DumpJsonParser::DUMP_BOTH_OVERFLOW);
  uint32_t set_overflow_num = DumpJsonParser::GetInstance().overflow_number();
  KernelTensorPtr overflow_kernel_tensor;

  if (overflow_flag) {
    overflow_kernel_tensor =
      HandleOverflow(tensor_info_list, device_context, stream_id, tensor_info_comm, set_overflow_num);
  }

  bool dump_host_stat =
    (DumpJsonParser::GetInstance().IsStatisticDump() && !DumpJsonParser::GetInstance().IsDeviceCalcStats());
  if (!dump_tensor && !dump_host_stat) {
    return;
  }
  device::CallbackFunc callback_func = [tensor_info_list, device_context, stream_id, tensor_info_comm, dump_tensor,
                                        dump_host_stat, overflow_flag, set_overflow_num, overflow_kernel_tensor]() {
    if (overflow_flag) {
      bool is_overflow = ProcessOverflow(overflow_kernel_tensor, set_overflow_num);
      if (!is_overflow) {
        return;
      }
    }
    for (const auto &tensor_info : tensor_info_list) {
      MS_EXCEPTION_IF_NULL(tensor_info.kernel_tensor);

      auto host_type = tensor_info.host_type;
      if (host_type > TypeId::kNumberTypeEnd || host_type < TypeId::kNumberTypeBegin ||
          host_type == kNumberTypeComplex64) {
        MS_VLOG(VL_DUMP) << "Cannot create tensor with type: " << TypeIdLabel(host_type);
        continue;
      }

      uint64_t timestamp = Common::GetTimeStamp();
      std::string type_str = TypeIdToString(host_type);
      transform(type_str.begin(), type_str.end(), type_str.begin(), tolower);
      std::string file_path = tensor_info_comm.file_path_prefix + '.' + std::to_string(timestamp) + '.' +
                              tensor_info.io + '.' + std::to_string(tensor_info.io_index) + '.' + tensor_info.format +
                              "." + type_str;
      auto host_shape = GetOriOutputTensorShape(tensor_info, host_type);
      mindspore::tensor::TensorPtr out_tensor = tensor::from_spec(host_type, host_shape, device::DeviceType::kCPU);
      MS_EXCEPTION_IF_NULL(out_tensor);
      size_t host_size = LongToSize(out_tensor->DataNBytes());
      if (host_size == 0) {
        std::string file_name = tensor_info_comm.file_path_prefix;
        if (file_name.rfind("/") != std::string::npos) {
          file_name = file_path.substr(file_name.rfind("/") + 1);
        }
        MS_LOG(WARNING) << "Dump tensor size is 0 for tensor: " << file_name << ". Skip it";
        continue;
      }
      host_size = ModifySize(host_type, host_size);
      size_t device_size = tensor_info.device_size;
      if (host_size > device_size) {
        MS_LOG(ERROR) << "Dump host size " << host_size << " greater than device size " << device_size;
        continue;
      }
      auto device_tensor = tensor_info.kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_tensor);
      device::DeviceContextKey host_key = {device_tensor->GetDeviceType(), device_tensor->device_id()};
      device::DeviceContext *host_context =
        device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
      MS_EXCEPTION_IF_NULL(host_context);
      MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
      auto ret = host_context->device_res_manager_->CopyDirectly(
        out_tensor->data_c(), host_size, tensor_info.device_ptr, device_size, device::CopyType::kD2H);
      MS_LOG(DEBUG) << "Callback aclrtmemcpy for " << file_path << ". result is: " << ret << file_path;
      auto tensor_storage_info = tensor_info.kernel_tensor->tensor_storage_info();
      // Convert the non-contiguous Tensor to contiguous
      if (tensor_storage_info != nullptr) {
        out_tensor = ExtractContiguousTensor(out_tensor, tensor_info.host_shape, tensor_info.host_type,
                                             tensor_storage_info->storage_offset, tensor_storage_info->strides);
        host_size = LongToSize(out_tensor->DataNBytes());
      }
      // Tensor must be saved before statistic. Because the tensor would be changed in DumpTensorStatsToFile when data
      // type is int4, if tensor saved after statistic, the tensor value would be wrong.
      if (dump_tensor) {
        DumpTensorToFile(file_path, out_tensor, host_type, host_size, out_tensor->shape());
      }

      if (dump_host_stat) {
        auto tensor_data = PrepareStatTensorData(out_tensor, tensor_info);

        bool is_input = (tensor_info.io == kInput);
        TensorStatDump stat_dump(tensor_info_comm.op_type, tensor_info_comm.stat_op_name, tensor_info_comm.task_id,
                                 stream_id, timestamp, is_input, tensor_info.io_index, 0);
        stat_dump.DumpTensorStatsToFile(tensor_info_comm.dump_path, tensor_data);
      }
    }
  };

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  auto callback_ret = host_context->device_res_manager_->LaunchCallback(callback_func, stream_id, true);
  if (!callback_ret) {
    MS_LOG(ERROR) << "Async dump callback launch fail.";
  }
}

void PrepareInputDataViaCallback(const CNodePtr &cnode, const DeviceContext *device_context,
                                 const std::vector<KernelTensor *> &input_kernel_tensors,
                                 std::vector<TensorInfoForDump> *tensor_info_list) {
  bool trans_flag = GetTransFlag();

  std::vector<size_t> valid_indexes =
    GetValidDumpIndex(cnode, input_kernel_tensors.size(), true, device_context, input_kernel_tensors);

  for (size_t index : valid_indexes) {
    auto input_kernel = cnode->input(index + 1);
    MS_EXCEPTION_IF_NULL(input_kernel_tensors[index]);
    auto &device_tensor = input_kernel_tensors[index]->device_address();
    MS_EXCEPTION_IF_NULL(device_tensor);

    auto type = GetInputKernelType(input_kernel, trans_flag);
    // For example, this happens with the Depend op
    if (type == kMetaTypeNone) {
      continue;
    }

    auto dump_shape = GetInputKernelShapeVec(input_kernel, input_kernel_tensors[index], index, trans_flag);
    auto host_format = kOpFormat_DEFAULT;
    auto format = trans_flag ? host_format : kernel::GetFormatFromEnumToStr(input_kernel_tensors[index]->format());

    tensor_info_list->emplace_back(TensorInfoForDump(kInput, index, format, type, dump_shape, device_tensor->GetSize(),
                                                     input_kernel_tensors[index]));
  }
}

void PrepareOutputDataViaCallback(const CNodePtr &cnode, const DeviceContext *device_context,
                                  const std::vector<KernelTensor *> &output_kernel_tensors,
                                  std::vector<TensorInfoForDump> *tensor_info_list) {
  auto output_size = AnfAlgo::GetOutputTensorNum(cnode);
  auto node_name = common::AnfAlgo::GetCNodeName(cnode);
  bool trans_flag = GetTransFlag();

  std::string kernel_name = GetKernelNodeName(cnode);
  std::vector<size_t> valid_indexes =
    GetValidDumpIndex(cnode, output_size, false, device_context, output_kernel_tensors);
  for (size_t index : valid_indexes) {
    auto type = GetOutputKernelType(cnode, index, trans_flag);
    // For example, this happens with the Depend op
    if (type == kMetaTypeNone) {
      continue;
    }

    MS_EXCEPTION_IF_NULL(output_kernel_tensors[index]);

    auto &device_tensor = output_kernel_tensors[index]->device_address();
    MS_EXCEPTION_IF_NULL(device_tensor);

    auto dump_shape = GetOutputKernelShapeVec(cnode, output_kernel_tensors[index], index, trans_flag);

    auto host_format = kOpFormat_DEFAULT;
    auto format = trans_flag ? host_format : kernel::GetFormatFromEnumToStr(output_kernel_tensors[index]->format());
    tensor_info_list->emplace_back(TensorInfoForDump(kOutput, index, format, type, dump_shape, device_tensor->GetSize(),
                                                     output_kernel_tensors[index]));
  }
}

TensorInfoCommForDump GetTensorInfoCommFromCnode(const CNodePtr &cnode) {
  auto kernel_graph = std::dynamic_pointer_cast<KernelGraph>(cnode->func_graph());
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto root_graph_id = kernel_graph->root_graph_id();

  uint32_t rank_id = datadump::GetRankID();
  std::string dump_path = GenerateDumpPath(root_graph_id, rank_id);
  std::string op_type = common::AnfAlgo::GetCNodeName(cnode);
  std::string op_name = GetKernelNodeName(cnode);
  std::string stat_op_name = op_name;
  GetFileKernelName(NOT_NULL(&op_name));

  uint32_t task_id = 0;
  auto stream_id = AnfAlgo::GetStreamId(cnode);

  TensorInfoCommForDump tensor_info_comm(dump_path, op_type, op_name, stat_op_name, task_id, stream_id);
  return tensor_info_comm;
}

inline mindspore::tensor::TensorPtr KernelTensor2Tensor(KernelTensorPtr kernel_tensor, const TypeId host_type,
                                                        const ShapeVector &host_shape) {
  if (!kernel_tensor) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  const void *src = kernel_tensor->device_ptr();
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);

  mindspore::tensor::TensorPtr out_tensor = tensor::from_spec(host_type, host_shape, device::DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(out_tensor);
  size_t host_size = LongToSize(out_tensor->DataNBytes());
  if (host_size == 0) {
    MS_LOG(WARNING) << "kernel tensor size is 0, skip it.";
    return out_tensor;
  }
  device::DeviceContextKey host_key = {device_tensor->GetDeviceType(), device_tensor->device_id()};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  host_context->device_res_manager_->CopyDirectly(out_tensor->data_c(), host_size, src, host_size,
                                                  device::CopyType::kD2H);
  return out_tensor;
}

inline string TensorToString(mindspore::tensor::TensorPtr tensor) {
  if (!tensor) {
    return "null";
  }
  return tensor->DataToString(false);
}

inline void Write2File(const TensorInfoForDump &tensor_info, uint32_t stream_id,
                       const TensorInfoCommForDump &tensor_info_comm) {
  string node_name = tensor_info_comm.stat_op_name;
  string node_type = tensor_info_comm.op_type;

  const string csv_header = CsvHeaderUtil::GetInstance().GetStatCsvHeader();
  const std::vector<string> &stat_name_list = DumpJsonParser::GetInstance().statistic_category();

  string filename = tensor_info_comm.dump_path + "/" + "statistic.csv";
  CsvWriter csv;
  std::lock_guard<std::mutex> lock(CsvFileMutexManager::GetInstance().GetCsvMutex(filename));
  if (!csv.OpenFile(filename, csv_header)) {
    MS_LOG(WARNING) << "filename is " << filename;
    MS_LOG(WARNING) << "Open statistic dump file failed, skipping current statistics";
    return;
  }
  uint64_t timestamp = Common::GetTimeStamp();
  std::string host_type = TypeIdToString(tensor_info.host_type, true);

  if (tensor_info.device_size == 0) {
    std::string file_name = tensor_info_comm.file_path_prefix;
    if (file_name.rfind("/") != std::string::npos) {
      file_name = file_name.substr(file_name.rfind("/") + 1);
    }
    file_name = file_name + '.' + std::to_string(timestamp) + '.' + tensor_info.io + '.' +
                std::to_string(tensor_info.io_index) + '.' + tensor_info.format + "." + host_type;
    MS_LOG(WARNING) << "Dump tensor size is 0 for tensor: " << file_name << ". Skip it";
    return;
  }

  csv.WriteToCsv(node_type);
  csv.WriteToCsv(node_name);
  csv.WriteToCsv(0);
  csv.WriteToCsv(stream_id);
  csv.WriteToCsv(timestamp);
  csv.WriteToCsv(tensor_info.io);
  csv.WriteToCsv(tensor_info.io_index);
  csv.WriteToCsv(tensor_info.device_size);
  csv.WriteToCsv(host_type);
  csv.WriteToCsv(ShapeToString(tensor_info.host_shape));

  for (const auto &name : stat_name_list) {
    auto it = tensor_info.stat_results.find(name);
    if (it == tensor_info.stat_results.end()) {
      csv.WriteToCsv("null");
      continue;
    }
    const TensorInfoForDump::KernelTensorMeta &kernel_tensor_meta = it->second;
    auto kernel_tensor = kernel_tensor_meta.tensor;
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto dtype_id = kernel_tensor_meta.dtype;
    auto shape = kernel_tensor_meta.shape;
    auto tensor = KernelTensor2Tensor(kernel_tensor, dtype_id, shape);
    csv.WriteToCsv(TensorToString(tensor));
  }
  csv.WriteToCsv("", true);
  csv.CloseFile();
}

void LaunchDeviceStatCallback(std::vector<TensorInfoForDump> *tensor_info_vec_ptr, const DeviceContext *device_context,
                              uint32_t stream_id, const TensorInfoCommForDump &tensor_info_comm) {
  const std::vector<std::string> &stat_name_list = DumpJsonParser::GetInstance().statistic_category();
  std::vector<TensorInfoForDump> &tensor_info_vec = *tensor_info_vec_ptr;
  // launch statistic kernel
  for (auto &tensor_info : tensor_info_vec) {
    auto kernel_tensor = tensor_info.kernel_tensor;
    for (auto &name : stat_name_list) {
      auto result = datadump::CalStatisticAsync(name, device_context, kernel_tensor, stream_id);
      if (result.empty() || !result.back()) {
        continue;
      }
      auto stat_res = result.back();
      result.pop_back();
      MS_EXCEPTION_IF_NULL(stat_res);
      tensor_info.stat_results.emplace(
        name, TensorInfoForDump::KernelTensorMeta{stat_res, stat_res->dtype_id(), stat_res->GetShapeVector()});
      tensor_info.workspace[name] = result;
    }
  }

  device::CallbackFunc callback_func = [tensor_info_vec, tensor_info_comm, stream_id]() mutable {
    for (auto &tensor_info : tensor_info_vec) {
      Write2File(tensor_info, stream_id, tensor_info_comm);
    }
  };

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  auto callback_ret = host_context->device_res_manager_->LaunchCallback(callback_func, stream_id, true);
  if (!callback_ret) {
    MS_LOG(ERROR) << "Async device statistic dump callback launch fail.";
  }
}

void DumpDataViaCallback(const CNodePtr &cnode, const std::vector<KernelTensor *> &input_kernel_tensors,
                         const std::vector<KernelTensor *> &output_kernel_tensors,
                         const DeviceContext *device_context) {
  TensorInfoCommForDump tensor_info_comm = GetTensorInfoCommFromCnode(cnode);
  auto stream_id = tensor_info_comm.stream_id;

  std::vector<TensorInfoForDump> tensor_info_list;
  if (DumpJsonParser::GetInstance().InputNeedDump()) {
    PrepareInputDataViaCallback(cnode, device_context, input_kernel_tensors, &tensor_info_list);
  }
  if (DumpJsonParser::GetInstance().OutputNeedDump()) {
    PrepareOutputDataViaCallback(cnode, device_context, output_kernel_tensors, &tensor_info_list);
  }
  bool calc_device_stat = DumpJsonParser::GetInstance().IsDeviceCalcStats();
  if (calc_device_stat) {
    LaunchDeviceStatCallback(&tensor_info_list, device_context, stream_id, tensor_info_comm);
  } else {
    LaunchDumpCallback(tensor_info_list, device_context, stream_id, tensor_info_comm);
  }
}

}  // namespace mindspore
