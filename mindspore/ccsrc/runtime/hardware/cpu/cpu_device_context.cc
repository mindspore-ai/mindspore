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

#include "runtime/hardware/cpu/cpu_device_context.h"
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"
#include "runtime/device/cpu/cpu_memory_manager.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "runtime/device/cpu/kernel_select_cpu.h"
#include "utils/trace_base.h"
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/common/pass_manager.h"
#include "backend/optimizer/common/common_backend_optimization.h"
#include "backend/optimizer/cpu/insert_cast_cpu.h"
#include "backend/optimizer/cpu/insert_format_transform_op.h"
#include "backend/optimizer/pass/replace_node_by_proxy.h"
#include "backend/optimizer/pass/erase_visit_attr.h"
#include "profiler/device/cpu/cpu_profiling.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif
#ifdef PLATFORM_86
#include <pmmintrin.h>
#endif

namespace mindspore {
namespace device {
namespace cpu {
using mindspore::kernel::KernelBuildInfo;

#ifdef PLATFORM_86
// Whether need set the flush zero mode in the kernel launch.
static bool flush_zero_mode_enable{false};
#endif

void CPUDeviceContext::Initialize() {
  if (initialized_) {
    return;
  }

  mem_manager_ = std::make_shared<CPUMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);

#ifndef ENABLE_SECURITY
  // Dump json config file if dump is enabled.
  auto rank_id = GetRankID();
  auto &json_parser = DumpJsonParser::GetInstance();
  json_parser.Parse();
  json_parser.CopyDumpJsonToDir(rank_id);
  json_parser.CopyMSCfgJsonToDir(rank_id);
#endif

  initialized_ = true;
}

void CPUDeviceContext::Destroy() {
  // Release memory.
  if (mem_manager_ != nullptr) {
    mem_manager_->FreeDeviceMemory();
    mem_manager_ = nullptr;
  }
}

bool CPUDeviceContext::AllocateMemory(DeviceAddress *const &address, size_t size) const {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto device_ptr = mem_manager_->MallocMemFromMemPool(size);
  if (!device_ptr) {
    return false;
  }
  address->ptr_ = device_ptr;
  address->size_ = size;
  address->from_mem_pool_ = true;
  return true;
}

void CPUDeviceContext::FreeMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(address->ptr_);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (!address->from_mem_pool()) {
    return;
  }
  mem_manager_->FreeMemFromMemPool(address->ptr_);
  address->ptr_ = nullptr;
}

DeviceAddressPtr CPUDeviceContext::CreateDeviceAddress(void *const device_ptr, size_t device_size, const string &format,
                                                       TypeId type_id) const {
  return std::make_shared<CPUDeviceAddress>(device_ptr, device_size, format, type_id);
}

void CPUDeviceContext::OptimizeGraph(const KernelGraphPtr &graph) const {
  // Update Graph Dynamic Shape Attr.
  UpdateGraphDynamicShapeAttr(NOT_NULL(graph));

  SetOperatorInfo(graph->execution_order());
  OptimizeGraphImpl(graph);

  // Run final optimization.
  opt::CommonFinalOptimization(graph);
}

void CPUDeviceContext::OptimizeSingleOpGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  SetOperatorInfo(graph->execution_order());
  OptimizeGraphImpl(graph);
}

void CPUDeviceContext::OptimizeGraphImpl(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertFormatTransformOpCPU>("insert_format_transform_op_cpu"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
  graph->SetExecOrderByDefault();
}

void CPUDeviceContext::UpdateGraphDynamicShapeAttr(const NotNull<KernelGraphPtr> &graph) const {
  for (const auto &cnode : graph->execution_order()) {
    if (AnfAlgo::IsNodeDynamicShape(cnode)) {
      AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(true), cnode);
      MS_LOG(INFO) << "Set Dynamic Shape Attr to Node:" << cnode->fullname_with_scope();
    }
  }
  graph->UpdateGraphDynamicAttr();
}

namespace {
void SetControlOpInfo(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<std::string> inputs_format;
  std::vector<TypeId> inputs_type;
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    (void)inputs_format.emplace_back(kOpFormat_DEFAULT);
    inputs_type.push_back(AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_index));
  }
  std::vector<std::string> outputs_format;
  std::vector<TypeId> outputs_type;
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    (void)outputs_format.emplace_back(kOpFormat_DEFAULT);
    outputs_type.push_back(AnfAlgo::GetOutputInferDataType(kernel_node, output_index));
  }

  auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
  builder->SetInputsFormat(inputs_format);
  builder->SetInputsDeviceType(inputs_type);
  builder->SetOutputsFormat(outputs_format);
  builder->SetOutputsDeviceType(outputs_type);

  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel_node.get());
}
}  // namespace

void CPUDeviceContext::SetOperatorInfo(const std::vector<CNodePtr> &nodes) const {
  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!AnfAlgo::IsControlOpExecInBackend(node)) {
      SetKernelInfo(node);
    } else {
      SetControlOpInfo(node);
    }
  }
}

void CPUDeviceContext::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (AnfAlgo::IsControlOpExecInBackend(node)) {
      continue;
    }
    std::string kernel_name = AnfAlgo::GetCNodeName(node);
    std::shared_ptr<kernel::CPUKernel> cpu_kernel = kernel::CPUKernelFactory::GetInstance().Create(kernel_name, node);
    if (!cpu_kernel) {
      MS_LOG(EXCEPTION) << "Build cpu operator[" << node->fullname_with_scope() << "] failed";
    }

#ifdef PLATFORM_86
    // Some CPU kernels need set the flush zero mode to improve performance.
    if (!flush_zero_mode_enable &&
        (kOpNeedSetFlushZeroModeList.find(kernel_name) != kOpNeedSetFlushZeroModeList.end())) {
      flush_zero_mode_enable = true;
    }
#endif

    cpu_kernel->Init(node);
    AnfAlgo::SetKernelMod(cpu_kernel, node.get());
  }
}

namespace {
void ProcessCast(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertCastCPU>("insert_cast_cpu"));
  MS_LOG(INFO) << "Insert cast pass";
  pm->AddPass(std::make_shared<opt::EraseVisitAttr>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
  graph->SetExecOrderByDefault();
}
}  // namespace

void CPUDeviceContext::PreprocessBeforeRunGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  ProcessCast(graph);

  // Remove reorder after PS feature finish adapting push/pull in auto_monad.
  auto execution_order = graph->execution_order();
  AnfAlgo::ReorderPosteriorExecList(NOT_NULL(&execution_order));
  graph->set_execution_order(execution_order);
}

void CPUDeviceContext::PreprocessBeforeRunSingleOpGraph(const KernelGraphPtr &graph) const { ProcessCast(graph); }

bool CPUDeviceContext::LaunchKernel(const CNodePtr &kernel, const std::vector<AddressPtr> &inputs,
                                    const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs,
                                    bool) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_LOG(DEBUG) << "Launch kernel: " << kernel->fullname_with_scope();
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto cpu_kernel_mod = dynamic_cast<kernel::CPUKernel *>(kernel_mod);
  MS_EXCEPTION_IF_NULL(cpu_kernel_mod);

#ifdef PLATFORM_86
  // Some CPU kernels need set the flush zero mode to improve performance.
  if (flush_zero_mode_enable) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  }
#endif

  // Some CPU kernels can't initialize kernel and launch kernel in different thread, so reinitialize the kernels before
  // launch.
  if (kOpNotSupportMultiThreadExecList.find(AnfAlgo::GetCNodeName(kernel)) != kOpNotSupportMultiThreadExecList.end()) {
    cpu_kernel_mod->InitKernel(kernel);
  }
#ifndef ENABLE_SECURITY
  const auto &profiler_inst = profiler::cpu::CPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);
  if (profiler_inst->GetEnableFlag()) {
    return LaunchKernelWithProfiling(kernel, inputs, workspace, outputs);
  }
#endif
  return DoLaunchKernel(kernel_mod, inputs, workspace, outputs);
}

bool CPUDeviceContext::LaunchKernelWithProfiling(const CNodePtr &kernel, const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) const {
  MS_EXCEPTION_IF_NULL(kernel);
  std::lock_guard<std::mutex> locker(launch_mutex_);

  auto profiler_inst = profiler::cpu::CPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);

  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  uint32_t pid = IntToUint(getpid());
  profiler_inst->OpDataProducerBegin(kernel->fullname_with_scope(), pid);
  bool ret = DoLaunchKernel(kernel_mod, inputs, workspace, outputs);
  profiler_inst->OpDataProducerEnd();

  return ret;
}

bool CPUDeviceContext::DoLaunchKernel(KernelMod *const kernel_mod, const std::vector<AddressPtr> &inputs,
                                      const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) const {
  MS_EXCEPTION_IF_NULL(kernel_mod);
  return kernel_mod->Launch(inputs, workspace, outputs, nullptr);
}

MS_REGISTER_DEVICE(kCPUDevice, CPUDeviceContext);
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
