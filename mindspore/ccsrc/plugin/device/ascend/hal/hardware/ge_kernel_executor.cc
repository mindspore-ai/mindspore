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
#include "include/common/utils/parallel_context.h"
#include "acl/acl_rt.h"
#include "acl/acl_op_compiler.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/hardware/ascend_graph_optimization.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/hardware/ge_graph_optimization.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_load.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_attr_and_input_convert_regist.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel_metadata.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel_build.h"

#ifndef ENABLE_SECURITY
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/hal/profiler/memory_profiling.h"
#include "plugin/device/ascend/hal/profiler/ascend_profiling.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "plugin/device/ascend/hal/device/ascend_kernel_task.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "plugin/device/ascend/optimizer/ascend_backend_optimization.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_build.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_build.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_build.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_metadata.h"
#include "plugin/device/ascend/kernel/host/host_kernel_build.h"
#include "plugin/device/ascend/kernel/host/host_kernel_metadata.h"
#include "kernel/kernel_build_info.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/ge_adapter_info.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/backend/debug/data_dump/overflow_dumper.h"
#include "include/backend/debug/profiler/profiling.h"
#include "utils/anf_utils.h"

using mindspore::device::ascend::ProfilingManager;
using mindspore::profiler::ascend::MemoryProfiling;
#endif

namespace mindspore::device::ascend {
namespace {
static const HashMap<::ge::DataType, std::string> kGeTypeToString = {{::ge::DataType::DT_BOOL, "bool"},
                                                                     {::ge::DataType::DT_INT8, "int8"},
                                                                     {::ge::DataType::DT_INT16, "int16"},
                                                                     {::ge::DataType::DT_INT32, "int32"},
                                                                     {::ge::DataType::DT_INT64, "int64"},
                                                                     {::ge::DataType::DT_UINT8, "uint8"},
                                                                     {::ge::DataType::DT_UINT16, "uint16"},
                                                                     {::ge::DataType::DT_UINT32, "uint32"},
                                                                     {::ge::DataType::DT_UINT64, "uint64"},
                                                                     {::ge::DataType::DT_FLOAT16, "float16"},
                                                                     {::ge::DataType::DT_FLOAT, "float"},
                                                                     {::ge::DataType::DT_DOUBLE, "double"},
                                                                     {::ge::DataType::DT_STRING, "string"},
                                                                     {::ge::DataType::DT_COMPLEX64, "complex64"},
                                                                     {::ge::DataType::DT_COMPLEX128, "complex128"},
                                                                     {::ge::DataType::DT_BF16, "bf16"}};
std::string ConvertGeTypeToString(::ge::DataType type) {
  if (kGeTypeToString.count(type) != 0) {
    return kGeTypeToString.at(type);
  }
  return "";
}

void PrintQueryAclTypeErr(const AnfNodePtr &node, const transform::ErrorAclType acl_err_type) {
  std::stringstream ss;
  ss << "The current [" << node->fullname_with_scope()
     << "] operator did not match any operator prototype library. The reason is:" << std::endl;

  switch (acl_err_type) {
    case transform::kUnknownOp: {
      ss << "The current operator needs to be supplemented with an adapter, please check in `transform` directory."
         << std::endl;
      break;
    }
    case transform::kInValidType: {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      ss << "The supported input and output data types for the current operator are:" << std::endl;
      std::string name = GetCNodeFuncName(cnode);
      const auto &info = transform::GeAdapterManager::GetInstance().GetInfo(name, true);
      const auto &input_supported_dtypes = info->input_supported_dtypes();
      for (auto [index, dtypes] : input_supported_dtypes) {
        ss << "InputDesc [" << index << "] support {";
        for (auto dtype : dtypes) {
          ss << ConvertGeTypeToString(dtype) << ",";
        }
        ss << "}" << std::endl;
      }
      const auto &output_supported_dtypes = info->output_supported_dtypes();
      for (auto [index, dtypes] : output_supported_dtypes) {
        ss << "OutputDesc [" << index << "] support {";
        for (auto dtype : dtypes) {
          ss << ConvertGeTypeToString(dtype) << ",";
        }
        ss << "}" << std::endl;
      }
      ss << std::endl;
      ss << "But current operator's input and output data types is:" << std::endl;
      size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
      size_t output_num = AnfUtils::GetOutputTensorNum(node);
      for (size_t i = 0; i < input_num; ++i) {
        ss << "InputDesc [" << i << "] is ";
        ss << TypeIdToString(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, i)) << std::endl;
      }
      for (size_t i = 0; i < output_num; ++i) {
        ss << "InputDesc [" << i << "] is ";
        ss << TypeIdToString(common::AnfAlgo::GetOutputInferDataType(node, i)) << std::endl;
      }
      break;
    }
    case transform::kSpecialOp: {
      ss << "The current operator is specified not to select ACL. Please contact the relevant engineer for help."
         << std::endl;
      break;
    }
    default:
      return;
  }

  MS_LOG(ERROR) << ss.str();
}

std::pair<KernelType, std::vector<std::shared_ptr<kernel::KernelBuildInfo>>> QueryKernelType(const AnfNodePtr &node) {
  transform::ErrorAclType acl_err_type = transform::ErrorAclType::kNormalOp;
  auto kernel_type = transform::AclHelper::GetKernelInfoFromGe(node, &acl_err_type);
  // Todo: add datatype and format filter
  if (kernel_type != KernelType::UNKNOWN_KERNEL_TYPE && kernel::IsRegisteredAclnnOp(node)) {
    return {KernelType::OPAPI_KERNEL, {}};
  }
  if (kernel_type != KernelType::UNKNOWN_KERNEL_TYPE && kernel_type != KernelType::HCCL_KERNEL) {
    return {kernel_type, {}};
  }
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list{};
  if (kernel_type == KernelType::HCCL_KERNEL) {
    kernel::HcclMetadataInfo(cnode, &kernel_info_list);
    return {KernelType::HCCL_KERNEL, kernel_info_list};
  }
  kernel::ConvertAttrAndInputBeforeAicpuKernelSelect(cnode);
  kernel::AicpuMetadataInfo(cnode, &kernel_info_list);
  if (!kernel_info_list.empty()) {
    return {KernelType::AICPU_KERNEL, kernel_info_list};
  }
  kernel::HostMetadataInfo(cnode, &kernel_info_list);
  if (!kernel_info_list.empty()) {
    return {KernelType::HOST_KERNEL, kernel_info_list};
  }
  PrintQueryAclTypeErr(node, acl_err_type);
  return {KernelType::UNKNOWN_KERNEL_TYPE, {}};
}

std::string KernelSelectDebugString(const kernel::KernelBuildInfo *build_info,
                                    const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list) {
  MS_EXCEPTION_IF_NULL(build_info);
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
AclKernelFormatList GetValidFormat(size_t input_num, size_t output_num) {
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
      MS_LOG(DEBUG) << "This node kernel build info in valid, it may be parameter or value node: "
                    << kernel_node->DebugString() << ", idx: " << input_idx
                    << ", input node: " << input_node->DebugString();
      type = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_idx);
      auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
      MS_EXCEPTION_IF_NULL(build_info);
      build_info->SetOutputDeviceType(type, idx);
    }
  } else {
    MS_LOG(DEBUG) << "Node no build info, node name: " << kernel_node->DebugString() << ", idx: " << input_idx
                  << ", input node: " << input_node->DebugString();
    type = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_idx);
  }
  return type;
}

void GenerateKernelBuildInfo(const AnfNodePtr &kernel, const KernelType &kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel);
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
    auto cand_format = GetValidFormat(input_num, output_num);
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
      if ((!transform::AclHelper::CheckDefaultSupportFormat(input_format)) && (kernel_type != HCCL_KERNEL)) {
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
    (void)input_types.push_back(GetInputDeviceType(kernel, i));
    // no tuple in PyNative dynamic shape
    (void)input_object_types.push_back(kernel::KernelObjectType::TENSOR);
  }
  for (size_t i = 0; i < output_num; i++) {
    (void)output_types.push_back(common::AnfAlgo::GetOutputInferDataType(kernel, i));
    // no tuple in PyNative dynamic shape
    (void)output_object_types.push_back(kernel::KernelObjectType::TENSOR);
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
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfAlgo::GetKernelMod(kernel)) {
      continue;
    }
    kernel::KernelModPtr kernel_mod_ptr = nullptr;
    if (AnfAlgo::GetKernelType(kernel) == KernelType::ACL_KERNEL) {
      kernel_mod_ptr = kernel::AclOpBuild(kernel);
    } else if (AnfAlgo::GetKernelType(kernel) == KernelType::AICPU_KERNEL) {
      kernel_mod_ptr = kernel::AicpuOpBuild(kernel);
    } else if (AnfAlgo::GetKernelType(kernel) == KernelType::HOST_KERNEL) {
      kernel_mod_ptr = kernel::HostOpBuild(kernel);
    } else if (AnfAlgo::GetKernelType(kernel) == KernelType::HCCL_KERNEL) {
      kernel_mod_ptr = kernel::HcclOpBuild(kernel);
    } else if (AnfAlgo::GetKernelType(kernel) == KernelType::OPAPI_KERNEL) {
      kernel_mod_ptr = kernel::AclnnOpBuild(kernel);
    } else {
      MS_LOG(EXCEPTION) << "The kernel: " << kernel->fullname_with_scope() << " kernel build failed, kernel type: "
                        << kernel::KernelTypeLabel(AnfAlgo::GetKernelType(kernel));
    }
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, kernel.get());
  }
  return true;
}

bool GraphWithNoRealKernel(const KernelGraphPtr &kernel_graph) {
  const auto &nodes = kernel_graph->execution_order();
  for (auto &node : nodes) {
    if (AnfUtils::IsRealKernel(node)) {
      return false;
    }
  }
  return true;
}

pynative::KernelTaskPtr GetTaskByTaskType(const pynative::KernelTaskType &task_type,
                                          const std::shared_ptr<pynative::KernelTaskContext> &context) {
  switch (task_type) {
    case pynative::KernelTaskType::kCONTIGUOUS_TASK:
      return std::make_shared<AscendContiguousKernelTask>(context);
    case pynative::KernelTaskType::kCOPY_TASK:
      return std::make_shared<AscendCopyWithSliceKernelTask>(context);
    default:
      MS_LOG(EXCEPTION) << "KernelTaskType is invalid, task_type:" << task_type;
  }
}

void SetAclOpPrecisionMode() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto op_precision_mode = ms_context->get_param<std::string>(MS_CTX_OP_PRECISION_MODE);
  if (op_precision_mode.empty()) {
    return;
  }
  MS_LOG(INFO) << "Set ACL_OP_PRECISION_MODE: " << op_precision_mode;
  auto ret = aclSetCompileopt(aclCompileOpt::ACL_OP_PRECISION_MODE, op_precision_mode.c_str());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set op precision mode failed! Error flag is " << ret;
  }
}
}  // namespace

void GeKernelExecutor::Initialize() {
  if (initialized_) {
    return;
  }
  MS_EXCEPTION_IF_NULL(device_context_);
  res_manager_ = device_context_->device_res_manager_.get();
  MS_EXCEPTION_IF_NULL(res_manager_);
  graph_executor_ = dynamic_cast<GeGraphExecutor *>(device_context_->graph_executor_.get());
  // not check graph executor, may use in ascend device context
  SetAclOpPrecisionMode();
  initialized_ = true;
}

void GeKernelExecutor::Destroy() {
  if (!initialized_) {
    return;
  }
  res_manager_ = nullptr;
  graph_executor_ = nullptr;
  initialized_ = false;
}

void GeKernelExecutor::UnifyMindIR(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  GEGraphOptimization::GetInstance().UnifyMindIR(graph);
}

void GeKernelExecutor::AddMindIRPass(const KernelGraphPtr &graph) const {
  GEGraphOptimization::GetInstance().GEMindIRPass(graph);
}

void GeKernelExecutor::OptimizeGraph(const FuncGraphPtr &graph) const {
  // will be cached by OpCompileInfo
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // GE graph run mode do optimize in ProcessBeforeRun
  if (kernel_graph->is_graph_run_mode() && IsEnableRefMode()) {
    return;
  }
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "GeOptimizeGraph", 1, 0, 0);
  GEGraphOptimization::GetInstance().OptimizeACLGraph(kernel_graph);
  // select kernel
  const auto &kernels = kernel_graph->execution_order();
  for (const auto &kernel : kernels) {
    auto [kernel_type, kernel_info_list] = QueryKernelType(kernel);
    if (kernel_type == KernelType::UNKNOWN_KERNEL_TYPE) {
      MS_EXCEPTION(TypeError) << "Query kernel type failed, node name: " << kernel->fullname_with_scope()
                              << ", node info: " << kernel->DebugString();
    }
    // in this func, no select process, acl/aicpu/host kernel may not support pre node's format.
    GenerateKernelBuildInfo(kernel, kernel_type);
    if (kernel_type != ACL_KERNEL && kernel_type != OPAPI_KERNEL) {
      auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel->kernel_info());
      MS_EXCEPTION_IF_NULL(kernel_info);
      auto build_info = kernel_info->select_kernel_build_info();
      MS_EXCEPTION_IF_NULL(build_info);
      bool find_valid = std::any_of(kernel_info_list.begin(), kernel_info_list.end(),
                                    [&build_info](const kernel::KernelBuildInfoPtr &item) {
                                      MS_EXCEPTION_IF_NULL(item);
                                      return item->IsSimilarityKernelBuildInfo(*build_info);
                                    });
      if ((!find_valid) && (kernel_type != HCCL_KERNEL)) {
        MS_EXCEPTION(TypeError) << "Invalid Kernel Build Info! Kernel type: " << kernel::KernelTypeLabel(kernel_type)
                                << ", node: " << kernel->fullname_with_scope()
                                << KernelSelectDebugString(build_info, kernel_info_list);
      }
    }
  }
  GEGraphOptimization::GetInstance().OptimizeACLGraphAfterKernelSelect(kernel_graph);
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "GeOptimizeGraph", 1, 0, 1);
}

void GeKernelExecutor::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  if (!nodes.empty() && IsEnableRefMode()) {
    auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(nodes[0]->func_graph());
    MS_EXCEPTION_IF_NULL(kernel_graph);
    // Not create kernel when use GE
    if (!kernel_graph->is_from_single_op() && kernel_graph->is_graph_run_mode()) {
      return;
    }
  }
  // build kernel mod
  MS_LOG(DEBUG) << "Status record: start create kernel.";
  profiler::CollectHostInfo("Ascend", "CreateKernel", "CreateGeKernel", 1, 0, 0);
  PROF_START(create_kernel);
  auto ret = GenerateKernelMod(nodes);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Kernel build error.";
  }
  PROF_END(create_kernel);
  profiler::CollectHostInfo("Ascend", "CreateKernel", "CreateGeKernel", 1, 0, 1);
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
  profiler::CollectHostInfo("Ascend", "PreprocessBeforeRun", "GePreprocess", 1, 0, 0);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  // use GE
  if (kernel_graph->is_graph_run_mode() && IsEnableRefMode()) {
    if (GraphWithNoRealKernel(kernel_graph)) {
      return;
    }
    MS_EXCEPTION_IF_NULL(graph_executor_);
    graph_executor_->PreprocessBeforeRun(kernel_graph);
    profiler::CollectHostInfo("Ascend", "PreprocessBeforeRun", "GePreprocess", 1, 0, 1);
    return;
  }

  // nop op -> memcpy
  const auto &nodes = kernel_graph->execution_order();
  for (const auto &node : nodes) {
    auto op_name = common::AnfAlgo::GetCNodeName(node);
    // If the 2nd input of reshape is not a value node, then there are two inputs to select the host reshape operator
    bool is_host_reshape_op = false;
    if (op_name == prim::kPrimReshape->name()) {
      auto kernel_mod = AnfAlgo::GetKernelMod(node);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      is_host_reshape_op = kernel_mod->GetKernelModType() == kernel::KernelModType::HostKernelMod;
    }
    bool is_nop_op = transform::AclHelper::IsNopNode(node);
    bool is_transpose_nop = (op_name == prim::kPrimTranspose->name() || op_name == prim::kPrimTransposeD->name()) &&
                            common::AnfAlgo::HasNodeAttr(kAttrNopOp, node);
    bool is_dynamic_shape_skip_execute = AnfAlgo::IsDynamicShapeSkipExecute(node);
    if (is_dynamic_shape_skip_execute || is_transpose_nop || (is_nop_op && !is_host_reshape_op)) {
      nop_op_to_memcpy_.insert(node);
    }
  }

  // load aicpu so
  LaunchDeviceLibrary();
  profiler::CollectHostInfo("Ascend", "PreprocessBeforeRun", "GePreprocess", 1, 0, 1);
}

bool GeKernelExecutor::PySyncRuning() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_EXCEPTION_IF_NULL(res_manager_);
  if ((ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) &&
      ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) &&
      !res_manager_->SyncStream(kDefaultStreamIndex)) {
    return false;
  }
  return true;
}

bool GeKernelExecutor::MemoryCopyAsync(const CNodePtr &node, const vector<AddressPtr> &inputs,
                                       const vector<AddressPtr> &outputs) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Launch MemoryCopyAsync instead for kernel " << node->fullname_with_scope();
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(DEBUG) << "Kernel " << node->fullname_with_scope() << " input output size should be 1 but"
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
  MS_EXCEPTION_IF_NULL(kernel);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto graph_id = AnfAlgo::GetGraphId(kernel.get());
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  KernelType kernel_type = AnfAlgo::GetKernelType(kernel);
  MS_EXCEPTION_IF_NULL(res_manager_);
  (void)res_manager_->BindDeviceToCurrentThread(false);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  // Stream id may not be assigned in some scenarios, such as PyNative. Use the default stream in those cases.
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  }
  MS_EXCEPTION_IF_NULL(stream);
  bool is_dynamic_shape_skip_execute = AnfAlgo::IsDynamicShapeSkipExecute(kernel);
  if (is_dynamic_shape_skip_execute) {
    nop_op_to_memcpy_.insert(kernel);
  }
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
    kernel_type = RT_KERNEL;
  } else {
    MS_LOG(DEBUG) << "Begin launch kernel: " << kernel->fullname_with_scope();
    bool ret = kernel_mod->Launch(inputs, workspace, outputs, stream);
    MS_LOG(DEBUG) << "End launch kernel: " << kernel->fullname_with_scope();
    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
      res_manager_->ResetStreamAndCtx();
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

bool GeKernelExecutor::ExecuteKernelTask(const pynative::KernelTaskType &task_type,
                                         const device::DeviceAddressPtrList &input_addr_list,
                                         const TensorStorageInfoPtrList &input_storage_list,
                                         const device::DeviceAddressPtrList &output_addr_list) const {
  auto stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);

  auto task_context = std::make_shared<pynative::KernelTaskContext>(device_context_, input_addr_list,
                                                                    input_storage_list, output_addr_list, stream);

  auto task = GetTaskByTaskType(task_type, task_context);
  MS_EXCEPTION_IF_NULL(task);
  auto ret = task->RunWithRet();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Exec task failed, task_type:" << task_type;
  }
  return ret;
}

}  // namespace mindspore::device::ascend
