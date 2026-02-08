/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/symmetric_memory/symmetric_memory_kernel_plugin.h"

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "kernel/ascend/symmetric_memory/symmetric_memory_kernel_in_out_map.h"
#include "plugin/ascend/kernel_executor/kernel_select_ascend.h"
#include "kernel/ascend/symmetric_memory/symmetric_memory_kernel_mod.h"
#include "kernel/ascend/symmetric_memory/symmetric_memory_helper.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "op_def/math_op_name.h"
#include "op_def/nn_op_name.h"
#include "acl/acl_base.h"
#include "utils/phase.h"
#include "utils/ms_context.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_q.h"

namespace mindspore::kernel {
namespace {
constexpr SubModuleId kSymmetricMemorySubModuleId = SubModuleId::SM_KERNEL;

static const std::unordered_map<mindspore::symmetricmemory::LogLevel, mindspore::MsLogLevel> kLogLevelMap = {
  {mindspore::symmetricmemory::LogLevel::DEBUG, mindspore::MsLogLevel::kDebug},
  {mindspore::symmetricmemory::LogLevel::INFO, mindspore::MsLogLevel::kInfo},
  {mindspore::symmetricmemory::LogLevel::WARNING, mindspore::MsLogLevel::kWarning},
  {mindspore::symmetricmemory::LogLevel::ERROR, mindspore::MsLogLevel::kError},
  {mindspore::symmetricmemory::LogLevel::EXCEPTION, mindspore::MsLogLevel::kException}};

void SymmetricMemoryLog(const std::string &value, const char *file_path, int32_t line, const char *func_name,
                        mindspore::symmetricmemory::LogLevel level) {
  MsLogLevel ms_level;
  auto it = kLogLevelMap.find(level);
  if (it != kLogLevelMap.end()) {
    ms_level = it->second;
  } else {
    ms_level = MsLogLevel::kWarning;
    MS_LOG(WARNING) << "LogLevel can not find in MsLogLevel, LogLevel is '" << level << "', and set 'WARNING' level.";
  }
  if (ms_level == MsLogLevel::kException) {
    mindspore::LogWriter(mindspore::LocationInfo(file_path, line, func_name), mindspore::kException,
                         kSymmetricMemorySubModuleId, mindspore::NoExceptionType, false, nullptr) ^
      mindspore::LogStream() << value;
  } else if (static_cast<int>(ms_level) >= mindspore::g_ms_submodule_log_levels[kSymmetricMemorySubModuleId] &&
             static_cast<int>(ms_level) <= static_cast<int>(mindspore::this_thread_max_log_level)) {
    mindspore::LogWriter(mindspore::LocationInfo(file_path, line, func_name), ms_level, kSymmetricMemorySubModuleId,
                         mindspore::NoExceptionType, false, nullptr) < mindspore::LogStream() << value;
  }
}

bool IsKernelGraphOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &outputs = common::AnfAlgo::GetAllOutputIndexByReturnTypes(func_graph->output());
  return std::find_if(outputs.begin(), outputs.end(), [&node](const auto &output) {
           const auto &real_pair = common::AnfAlgo::VisitKernelWithReturnType(node, 0);
           return output.first == node || (real_pair.first == output.first && real_pair.second == output.second);
         }) != outputs.end();
}

bool IsNeedInsertTransDataForGraphOut(const AnfNodePtr &node, const std::vector<std::string> &output_formats) {
  // output is graph output & format is nz
  if (IsKernelGraphOutput(node) &&
      std::any_of(output_formats.begin(), output_formats.end(),
                  [](const std::string &format) { return !CheckDefaultSupportFormat(format); })) {
    return true;
  }
  return false;
}

void GetMsTypesList(const CNodePtr &kernel, std::vector<TypeId> *ms_in_dtypes, std::vector<TypeId> *ms_out_dtypes) {
  auto input_num = common::AnfAlgo::GetInputTensorNum(kernel);
  auto output_num = AnfUtils::GetOutputTensorNum(kernel);

  for (size_t i = 0; i < input_num; i++) {
    auto cur_input_type = mindspore::device::ascend::GetInputDeviceType(kernel, i);
    if (mindspore::device::ascend::IsEmptyTupleInput(kernel, i, cur_input_type)) {
      cur_input_type = TypeId::kNumberTypeInt64;
    }
    (void)ms_in_dtypes->push_back(cur_input_type);
  }

  for (size_t i = 0; i < output_num; i++) {
    (void)ms_out_dtypes->push_back(common::AnfAlgo::GetOutputInferDataType(kernel, i));
  }
  return;
}

}  // namespace

void SymmetricMemoryKernelPlugin::InitInternalLog() { SetLogFunction(&mindspore::kernel::SymmetricMemoryLog); }
KernelModPtr SymmetricMemoryKernelPlugin::BuildKernel(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);

  std::string op_fullname = anf_node->fullname_with_scope();
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  // Easy to compare accuracy and performance, later changed to debug
  KernelModPtr kernel_ptr;
  if (Factory<SymmetricMemoryKernelMod>::Instance().IsRegistered(opname)) {
    MS_LOG(INFO) << "Supported by SymmetricMemoryKernel: " << opname;
    kernel_ptr = std::static_pointer_cast<KernelMod>(Factory<SymmetricMemoryKernelMod>::Instance().Create(opname));
  }

  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "symmetric_memory can't find Kernel[" << opname << "]";
    return nullptr;
  }
  kernel_ptr->set_fullname(op_fullname);
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);
  auto prim_ptr = common::AnfAlgo::GetCNodePrimitive(anf_node);
  if (prim_ptr == nullptr) {
    MS_LOG(ERROR) << "symmetric_memory can't find primitive Kernel[" << opname << "]";
    return nullptr;
  }
  if (!kernel_ptr->Init(prim_ptr, input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG_WITH_NODE(EXCEPTION, anf_node) << "#dmsg#Kernel build failed:#dmsg#Initialize symmetric_memory kernel op["
                                          << anf_node->fullname_with_scope() << "] failed.";
  }

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckResizeCondition(cnode)) {
    if (kernel_ptr->Resize(input_kernel_tensors, output_kernel_tensors) == KRET_RESIZE_FAILED) {
      MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#symmetric_memory kernel op[" << cnode->fullname_with_scope()
                        << "] Resize failed.";
    }
  }

  return kernel_ptr;
}

bool SymmetricMemoryKernelPlugin::IsRegisteredKernel(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  std::vector<TypeId> ms_in_dtypes;
  std::vector<TypeId> ms_out_dtypes;
  GetMsTypesList(cnode, &ms_in_dtypes, &ms_out_dtypes);
  if (Factory<SymmetricMemoryKernelMod>::Instance().IsRegistered(opname)) {
    auto symmetric_memory_op_name = TransSymmetricMemoryOpName(opname);
    auto symmetric_memory_in_dtypes =
      SymmetricMemoryKernelModInOutMap::GetInstance()->MapSymmetricMemoryInputDtypes(opname, ms_in_dtypes);
    auto symmetric_memory_out_dtypes =
      SymmetricMemoryKernelModInOutMap::GetInstance()->MapSymmetricMemoryOutputDtypes(opname, ms_out_dtypes);
    return symmetricmemory::IsSymmetricMemoryKernelDtypesSupported(symmetric_memory_op_name, symmetric_memory_in_dtypes,
                                                                   symmetric_memory_out_dtypes);
  }

  return false;
}

void SymmetricMemoryKernelPlugin::GetValidKernelBuildInfoWithInternalFormat(const AnfNodePtr &node,
                                                                            std::vector<std::string> *input_formats,
                                                                            std::vector<std::string> *output_formats) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input_formats);
  MS_EXCEPTION_IF_NULL(output_formats);

  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);

  for (size_t i = 0; i < input_num; ++i) {
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
    auto first_node = kernel_with_index.first;
    if (first_node->isa<ValueNode>()) {
      auto value_node = first_node->cast<ValueNodePtr>();
      auto value = value_node->value();
      if (value->isa<None>()) {
        continue;
      }
    }
    std::string input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    if (!CheckDefaultSupportFormat(input_format)) {
      MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg# kernel op[" << node->fullname_with_scope()
                        << "] input format " << input_format << " is not supported.";
    }
  }
  // check if graph output is nz format
  if (IsNeedInsertTransDataForGraphOut(node, *output_formats)) {
    MS_LOG(EXCEPTION) << "#dmsg#Kernel build failed:#dmsg# kernel op[" << node->fullname_with_scope()
                      << "] special output format is not supported.";
  }
}

MS_KERNEL_PLUGIN_FACTORY_REG(SymmetricMemoryKernelPlugin, SymmetricMemoryKernelPlugin);
}  // namespace mindspore::kernel
