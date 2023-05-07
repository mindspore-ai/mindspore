/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/hardware/ge_kernel_executor.h"
#include <utility>
#include <algorithm>
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/hardware/ascend_graph_optimization.h"
#include "plugin/device/ascend/hal/device/kernel_build_ascend.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_load.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "include/common/utils/parallel_context.h"
#include "acl/acl_rt.h"

#ifndef ENABLE_SECURITY
#include "toolchain/adx_datadump_callback.h"
#include "plugin/device/ascend/hal/profiler/memory_profiling.h"
#include "utils/anf_utils.h"
#include "plugin/device/ascend/hal/profiler/ascend_profiling.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "include/backend/optimizer/helper.h"
#include "debug/data_dump/e2e_dump.h"
#include "debug/data_dump/overflow_dumper.h"
#include "kernel/acl/acl_kernel_build.h"
#include "kernel/aicpu/aicpu_kernel_build.h"
#include "kernel/aicpu/aicpu_kernel_metadata.h"
#include "kernel/host/host_kernel_build.h"
#include "kernel/host/host_kernel_metadata.h"
#include "kernel/kernel_build_info.h"
#include "transform/acl_ir/acl_helper.h"
#include "include/common/debug/anf_ir_dump.h"

using Adx::AdxRegDumpProcessCallBack;
using mindspore::device::ascend::ProfilingManager;
using mindspore::profiler::ascend::MemoryProfiling;
#endif

namespace mindspore::device::ascend {
namespace {
std::pair<KernelType, std::vector<std::shared_ptr<kernel::KernelBuildInfo>>> QueryKernelType(const AnfNodePtr &node) {
  auto kernel_type = transform::AclHelper::GetKernelInfoFromGe(node);
  if (kernel_type != KernelType::UNKNOWN_KERNEL_TYPE) {
    return {kernel_type, {}};
  }
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list{};
  opt::ConvertAttrAndInputBeforeAicpuKernelSelect(cnode);
  kernel::AicpuMetadataInfo(cnode, &kernel_info_list);
  if (!kernel_info_list.empty()) {
    return {KernelType::AICPU_KERNEL, kernel_info_list};
  }
  kernel::HostMetadataInfo(cnode, &kernel_info_list);
  if (!kernel_info_list.empty()) {
    return {KernelType::HOST_KERNEL, kernel_info_list};
  }
  return {KernelType::UNKNOWN_KERNEL_TYPE, {}};
}

std::string KernelSelectDebugString(const kernel::KernelBuildInfo *build_info,
                                    const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list) {
  std::ostringstream output_buffer;
  output_buffer << std::endl;
  output_buffer << "need build info: " << std::endl;
  output_buffer << build_info->ToString() << std::endl;
  output_buffer << "candidate build info list: " << std::endl;
  for (const auto &info : kernel_info_list) {
    output_buffer << info->ToString() << std::endl;
  }
  return output_buffer.str();
}

using AclKernelFormatList = std::vector<std::pair<std::vector<string>, std::vector<string>>>;
AclKernelFormatList GetValidFormat(const AnfNodePtr &node, size_t input_num, size_t output_num) {
  std::vector<std::string> inputs_format(input_num, kOpFormat_DEFAULT);
  std::vector<std::string> outputs_format(output_num, kOpFormat_DEFAULT);
  return {std::make_pair(inputs_format, outputs_format)};
}
}  // namespace
namespace {
constexpr uint32_t kFirstItem = 0;

TypeId GetInputDeviceType(const AnfNodePtr &kernel_node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  TypeId type = kTypeUnknown;
  auto [input_node, idx] = common::AnfAlgo::GetPrevNodeOutput(kernel_node, input_idx);
  MS_EXCEPTION_IF_NULL(input_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(input_node->kernel_info());
  if (kernel_info != nullptr && kernel_info->select_kernel_build_info() != nullptr) {
    type = AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_idx);
    if (type == kTypeUnknown) {
      MS_LOG(INFO) << "This node kernel build info in valid, it may be parameter or value node: "
                   << kernel_node->DebugString() << ", idx: " << input_idx
                   << ", input node: " << input_node->DebugString();
      type = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_idx);
      auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
      MS_EXCEPTION_IF_NULL(build_info);
      build_info->SetOutputDeviceType(type, idx);
    }
  } else {
    MS_LOG(INFO) << "Node no build info, node name: " << kernel_node->DebugString() << ", idx: " << input_idx
                 << ", input node: " << input_node->DebugString();
    type = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_idx);
  }
  return type;
}

void GenerateKernelBuildInfo(const AnfNodePtr &kernel, const KernelType &kernel_type) {
  std::vector<std::string> input_formats;
  std::vector<std::string> output_formats;
  std::vector<std::string> input_reshape_types;
  std::vector<std::string> output_reshape_types;
  auto input_num = common::AnfAlgo::GetInputTensorNum(kernel);
  auto output_num = AnfUtils::GetOutputTensorNum(kernel);
  if (kernel_type == ACL_KERNEL) {
    transform::AclHelper::GetValidKernelBuildInfo(kernel, &input_formats, &output_formats, &input_reshape_types,
                                                  &output_reshape_types);
  } else {
    auto cand_format = GetValidFormat(kernel, input_num, output_num);
    if (cand_format.empty()) {
      MS_LOG(EXCEPTION) << "The kernel: " << kernel->fullname_with_scope()
                        << " does not have a supported dynamic shape format on the Ascend platform.";
    }
    input_formats = cand_format.at(kFirstItem).first;
    output_formats = cand_format.at(kFirstItem).second;
    input_reshape_types.assign(input_num, "");
    output_reshape_types.assign(output_num, "");
    for (size_t i = 0; i < common::AnfAlgo::GetInputTensorNum(kernel); i++) {
      auto input_format = AnfAlgo::GetPrevNodeOutputFormat(kernel, i);
      if (!transform::AclHelper::CheckDefaultSupportFormat(input_format)) {
        MS_LOG(EXCEPTION) << "Aicpu kernel input not support this format: " << input_format
                          << ", kernel: " << kernel->fullname_with_scope() << ", input idx: " << i;
      }
    }
  }
  std::vector<TypeId> input_types;
  input_types.reserve(input_num);
  std::vector<TypeId> output_types;
  output_types.reserve(output_num);
  std::vector<kernel::KernelObjectType> input_object_types;
  input_object_types.reserve(input_num);
  std::vector<kernel::KernelObjectType> output_object_types;
  output_object_types.reserve(output_num);

  for (size_t i = 0; i < input_num; i++) {
    input_types.push_back(GetInputDeviceType(kernel, i));
    // no tuple in PyNative dynamic shape
    input_object_types.push_back(kernel::KernelObjectType::TENSOR);
  }
  for (size_t i = 0; i < output_num; i++) {
    output_types.push_back(common::AnfAlgo::GetOutputInferDataType(kernel, i));
    // no tuple in PyNative dynamic shape
    output_object_types.push_back(kernel::KernelObjectType::TENSOR);
  }
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetKernelType(kernel_type);
  builder->SetInputsFormat(input_formats);
  builder->SetInputsDeviceType(input_types);
  builder->SetInputsKernelObjectType(input_object_types);
  builder->SetOutputsFormat(output_formats);
  builder->SetOutputsDeviceType(output_types);
  builder->SetOutputsKernelObjectType(output_object_types);
  builder->SetInputsReshapeType(input_reshape_types);
  builder->SetOutputsReshapeType(output_reshape_types);
  if (input_formats.size() != input_types.size() || input_formats.size() != input_object_types.size()) {
    MS_LOG(EXCEPTION) << "The input buildInfo size kernel: " << kernel->fullname_with_scope()
                      << "is not equal, input_formats size: " << input_formats.size()
                      << ", input_types size: " << input_types.size()
                      << ", input_object_types size: " << input_object_types.size();
  }
  if (output_formats.size() != output_types.size() || output_formats.size() != output_object_types.size()) {
    MS_LOG(EXCEPTION) << "The output buildInfo size kernel: " << kernel->fullname_with_scope()
                      << "is not equal, output_formats size: " << output_formats.size()
                      << ", output_types size: " << output_types.size()
                      << ", output_object_types size: " << output_object_types.size();
  }
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel.get());
}

bool GenerateKernelMod(const std::vector<CNodePtr> &kernels) {
  for (const auto &kernel : kernels) {
    kernel::KernelModPtr kernel_mod_ptr = nullptr;
    if (AnfAlgo::GetKernelType(kernel) == KernelType::ACL_KERNEL) {
      kernel_mod_ptr = kernel::AclOpBuild(kernel);
    } else if (AnfAlgo::GetKernelType(kernel) == KernelType::AICPU_KERNEL) {
      kernel_mod_ptr = kernel::AicpuOpBuild(kernel);
    } else if (AnfAlgo::GetKernelType(kernel) == KernelType::HOST_KERNEL) {
      kernel_mod_ptr = kernel::HostOpBuild(kernel);
    } else {
      MS_LOG(EXCEPTION) << "The kernel: " << kernel->fullname_with_scope() << " kernel build failed, kernel type: "
                        << kernel::KernelTypeLabel(AnfAlgo::GetKernelType(kernel));
    }
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, kernel.get());
  }
  return true;
}
}  // namespace

void GeKernelExecutor::Initialize() {
  if (initialized_) {
    return;
  }
  res_manager_ = dynamic_cast<AscendDeviceResManager *>(device_context_->device_res_manager_.get());
  MS_EXCEPTION_IF_NULL(res_manager_);
  initialized_ = true;
}

void GeKernelExecutor::Destroy() {
  if (!initialized_) {
    return;
  }
  AscendGraphOptimization::GetInstance().Reset();
  res_manager_ = nullptr;
  initialized_ = false;
}

void GeKernelExecutor::UnifyMindIR(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  AscendGraphOptimization::GetInstance().UnifyMindIR(graph);
}

void GeKernelExecutor::AddMindIRPass(const KernelGraphPtr &graph) const {
  AscendGraphOptimization::GetInstance().AscendMindIRPass(graph);
}

void GeKernelExecutor::OptimizeGraph(const FuncGraphPtr &graph) const {
  // will be cached by OpCompileInfo
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  AscendGraphOptimization::GetInstance().OptimizeACLGraph(kernel_graph);
  // select kernel
  const auto &kernels = kernel_graph->execution_order();
  for (const auto &kernel : kernels) {
    auto [kernel_type, kernel_info_list] = QueryKernelType(kernel);
    if (kernel_type == KernelType::UNKNOWN_KERNEL_TYPE) {
      MS_LOG(EXCEPTION) << "Query kernel type failed, node name: " << kernel->fullname_with_scope()
                        << ", node info: " << kernel->DebugString();
    }
    // in this func, no select process, acl/aicpu/host kernel may not support pre node's format.
    GenerateKernelBuildInfo(kernel, kernel_type);
    if (kernel_type != ACL_KERNEL) {
      auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel->kernel_info());
      MS_EXCEPTION_IF_NULL(kernel_info);
      auto build_info = kernel_info->select_kernel_build_info();
      MS_EXCEPTION_IF_NULL(build_info);
      bool find_valid = std::any_of(kernel_info_list.begin(), kernel_info_list.end(),
                                    [&build_info](const kernel::KernelBuildInfoPtr &item) {
                                      MS_EXCEPTION_IF_NULL(item);
                                      return item->IsSimilarityKernelBuildInfo(*build_info);
                                    });
      if (!find_valid) {
        std::string kernel_type_str = (kernel_type == AICPU_KERNEL) ? "AICPU_KERNEL" : "HOST_KERNEL";
        MS_LOG(EXCEPTION) << "Invalid Kernel Build Info! Kernel type: " << kernel_type_str
                          << ", node: " << kernel->fullname_with_scope()
                          << KernelSelectDebugString(build_info, kernel_info_list);
      }
    }
  }
}

void GeKernelExecutor::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  // build kernel mod
  MS_LOG(DEBUG) << "Status record: start create kernel.";
  PROF_START(create_kernel);
  auto ret = GenerateKernelMod(nodes);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Kernel build error.";
  }
  PROF_END(create_kernel);
  MS_LOG(DEBUG) << "Status record: end create kernel.";
}

void GeKernelExecutor::LaunchDeviceLibrary() {
  MS_LOG(DEBUG) << "Status record: start launch device library.";
  auto ret = mindspore::kernel::AicpuOpKernelLoad::GetInstance().LaunchAicpuKernelSo();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Cust aicpu kernel so load failed.";
  }
  MS_LOG(DEBUG) << "Status record: end launch device library.";
}

void GeKernelExecutor::PreprocessBeforeRun(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  // nop op -> memcpy
  const auto &nodes = kernel_graph->execution_order();
  for (const auto &node : nodes) {
    auto op_name = common::AnfAlgo::GetCNodeName(node);
    // Save the nop_op that needs to be memcpy
    static mindspore::HashSet<std::string> nop_nodes = {prim::kPrimReshape->name(), prim::kPrimExpandDims->name(),
                                                        prim::kPrimSqueeze->name(), prim::kPrimFlatten->name(),
                                                        prim::kPrimFlattenGrad->name()};
    // If the 2nd input of reshape is not a value node, then there are two inputs to select the host reshape operator
    bool is_host_reshape_op = false;
    if (op_name == prim::kPrimReshape->name()) {
      auto kernel_mod = AnfAlgo::GetKernelMod(node);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      is_host_reshape_op = kernel_mod->GetKernelModType() == kernel::KernelModType::HostKernelMod;
    }
    bool is_nop_op = nop_nodes.find(op_name) != nop_nodes.end();
    bool is_transpose_nop = (op_name == prim::kPrimTranspose->name() || op_name == prim::kPrimTransposeD->name()) &&
                            common::AnfAlgo::HasNodeAttr(kAttrNopOp, node);
    bool is_dynamic_shape_skip_execute = AnfAlgo::IsDynamicShapeSkipExecute(node);
    if (is_dynamic_shape_skip_execute || is_transpose_nop || (is_nop_op && !is_host_reshape_op)) {
      nop_op_to_memcpy_.insert(node);
    }
  }

  // load aicpu so
  LaunchDeviceLibrary();
  // build kernel mod
  CreateKernel(nodes);
}

bool GeKernelExecutor::PySyncRuning() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if ((ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) &&
      ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) &&
      !res_manager_->SyncStream(kDefaultStreamIndex)) {
    return false;
  }
  return true;
}

bool GeKernelExecutor::MemoryCopyAsync(const CNodePtr &node, const vector<AddressPtr> &inputs,
                                       const vector<AddressPtr> &outputs) const {
  MS_LOG(DEBUG) << "Launch MemoryCopyAsync instead for kernel " << node->fullname_with_scope();
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(INFO) << "Kernel " << node->fullname_with_scope() << " input output size should be 1 but"
                 << " input size is:" << inputs.size() << " output size is:" << outputs.size();
  }

  const auto stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  MS_EXCEPTION_IF_NULL(stream);
  aclError status = aclrtMemcpyAsync(outputs[0]->addr, outputs[0]->size, inputs[0]->addr, inputs[0]->size,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (status != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "MemCpyAsync op aclrtMemcpyAsync failed, ret:" << status << " destMax:" << outputs[0]->size
                  << " count:" << inputs[0]->size;
    return false;
  }
  return true;
}

bool GeKernelExecutor::LaunchKernel(const CNodePtr &kernel, const vector<AddressPtr> &inputs,
                                    const vector<AddressPtr> &workspace, const vector<AddressPtr> &outputs,
                                    size_t stream_id) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto graph_id = AnfAlgo::GetGraphId(kernel.get());
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  KernelType kernel_type = AnfAlgo::GetKernelType(kernel);
  MS_EXCEPTION_IF_NULL(kernel);
  (void)res_manager_->BindDeviceToCurrentThread(false);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  // Stream id may not be assigned in some scenarios, such as PyNative. Use the default stream in those cases.
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  }
  MS_EXCEPTION_IF_NULL(stream);
#ifdef ENABLE_DEBUGGER
  if (DumpJsonParser::GetInstance().async_dump_enabled()) {
    auto register_dumper = debug::OverflowDumper::GetInstance(kAscendDevice);
    register_dumper->Init();
    register_dumper->OpDebugRegisterForStream(kernel);
  }
#endif
  // launch kernel
  if (nop_op_to_memcpy_.find(kernel) != nop_op_to_memcpy_.end()) {
    if (!MemoryCopyAsync(kernel, inputs, outputs)) {
      MS_LOG(ERROR) << "Memory copy failed for kernel " << kernel->fullname_with_scope();
      return false;
    }
  } else {
    MS_LOG(DEBUG) << "Begin launch kernel: " << kernel->fullname_with_scope();
    bool ret = kernel_mod->Launch(inputs, workspace, outputs, stream);
    MS_LOG(DEBUG) << "End launch kernel: " << kernel->fullname_with_scope();
    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
      return false;
    }
  }
#ifdef ENABLE_DEBUGGER
  if (DumpJsonParser::GetInstance().async_dump_enabled()) {
    auto kernel_dumper = debug::OverflowDumper::GetInstance(kAscendDevice);
    kernel_dumper->OpLoadDumpInfo(kernel);
  }
#endif
#ifndef ENABLE_SECURITY
  auto ascend_instance = profiler::ascend::AscendProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(ascend_instance);
  if (ProfilingManager::GetInstance().IsProfilingInitialized()) {
    ascend_instance->GetNodeTaskIdStreamId(kernel, graph_id, UintToInt(device_id), kernel_type, kernel_mod->task_id());
  }
#endif
  // for PyNative Sync Run mode
  return PySyncRuning();
}
}  // namespace mindspore::device::ascend
