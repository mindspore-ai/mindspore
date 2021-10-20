/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License"){}
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

#include "runtime/hardware/gpu/gpu_device_context.h"
#include <dlfcn.h>
#include <utility>
#include "pipeline/pynative/pynative_profiling.h"
#include "runtime/device/gpu/kernel_info_setter.h"
#include "runtime/device/gpu/gpu_kernel_build.h"
#include "runtime/device/gpu/gpu_device_address.h"
#include "runtime/device/gpu/gpu_memory_manager.h"
#include "runtime/device/gpu/gpu_memory_allocator.h"
#include "runtime/device/gpu/gpu_stream_assign.h"
#include "runtime/device/gpu/distribution/collective_init.h"
#include "runtime/device/gpu/gpu_device_manager.h"
#include "runtime/device/gpu/gpu_buffer_mgr.h"
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/gpu/gpu_common.h"
#include "runtime/hardware/gpu/optimizer.h"
#include "common/trans.h"
#include "utils/context/graph_kernel_flags.h"
#include "runtime/device/gpu/gpu_bucket.h"
#include "profiler/device/gpu/gpu_profiling.h"
#include "profiler/device/gpu/gpu_profiling_utils.h"
#include "backend/session/kernel_graph.h"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/running_data_recorder.h"
#endif
#include "utils/comm_manager.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif
#include "backend/optimizer/pass/optimize_updatestate.h"

namespace mindspore {
namespace device {
namespace gpu {
using KernelGraph = mindspore::session::KernelGraph;

static thread_local bool cur_thread_device_inited{false};

void GPUDeviceContext::Initialize() {
  if (initialized_ == true) {
    if (!BindDeviceToCurrentThread()) {
      MS_LOG(EXCEPTION) << "BindDeviceToCurrentThread failed.";
    }
    GPUMemoryAllocator::GetInstance().CheckMaxDeviceMemory();
    return;
  }

  // Set device id
  const void *collective_handle_ = CollectiveInitializer::instance().collective_handle();
  bool collective_inited = CollectiveInitializer::instance().collective_inited();
  if (collective_inited && collective_handle_ != nullptr) {
    DeviceContextKey old_key = device_context_key_;
    auto get_local_rank_funcptr =
      reinterpret_cast<GetLocalRankId>(dlsym(const_cast<void *>(collective_handle_), "local_rank_id"));
    MS_EXCEPTION_IF_NULL(get_local_rank_funcptr);
    device_context_key_.device_id_ = IntToUint((*get_local_rank_funcptr)());

    DeviceContextManager::GetInstance().UpdateDeviceContextKey(old_key, device_context_key_);

    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, device_context_key_.device_id_);
  }

  // Set device id and initialize device resource.
  if (!InitDevice()) {
    MS_LOG(EXCEPTION) << "GPU InitDevice failed.";
  }

  // Initialize memory pool.
  mem_manager_ = std::make_shared<GPUMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->MallocDeviceMemory();

  // Initialize NCCL.
  if (collective_inited && collective_handle_ != nullptr) {
    auto init_nccl_comm_funcptr =
      reinterpret_cast<InitNCCLComm>(dlsym(const_cast<void *>(collective_handle_), "InitNCCLComm"));
    MS_EXCEPTION_IF_NULL(init_nccl_comm_funcptr);
    (*init_nccl_comm_funcptr)();
  }

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

bool GPUDeviceContext::InitDevice() {
  if (GPUDeviceManager::GetInstance().device_count() <= 0) {
    MS_LOG(ERROR) << "No GPU device found.";
    return false;
  }

  if (!GPUDeviceManager::GetInstance().is_device_id_init()) {
    if (!GPUDeviceManager::GetInstance().set_cur_device_id(device_context_key_.device_id_)) {
      MS_LOG(ERROR) << "Failed to set current device id: " << SizeToInt(device_context_key_.device_id_);
      return false;
    }
  }

  // Initialize device resource, such as stream, cudnn and cublas handle.
  GPUDeviceManager::GetInstance().InitDevice();

  auto stream = GPUDeviceManager::GetInstance().default_stream();
  MS_ERROR_IF_NULL(stream);
  streams_.push_back(stream);

  void *communication_stream = nullptr;
  GPUDeviceManager::GetInstance().CreateStream(&communication_stream);
  MS_ERROR_IF_NULL(communication_stream);
  streams_.push_back(communication_stream);

  return true;
}

void GPUDeviceContext::Destroy() {
  // Release GPU buffer manager resource
#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  if (debugger && debugger->debugger_enabled()) {
    debugger->SetTrainingDone(true);
    debugger->SendMetadata(false);
  }
#endif

  if (GpuBufferMgr::GetInstance().IsInit()) {
    if (!GpuBufferMgr::GetInstance().IsClosed() && !GpuBufferMgr::GetInstance().CloseNotify()) {
      MS_LOG(ERROR) << "Could not close gpu data queue.";
    }
    CHECK_OP_RET_WITH_ERROR(GpuBufferMgr::GetInstance().Destroy(), "Could not destroy gpu data queue.");
  }

  // Release stream, cudnn and cublas handle, etc.
  GPUDeviceManager::GetInstance().ReleaseDevice();

  // Release device memory
  if (mem_manager_ != nullptr) {
    mem_manager_->FreeDeviceMemory();
    mem_manager_ = nullptr;
  }
}

bool GPUDeviceContext::AllocateMemory(DeviceAddress *const &address, size_t size) const {
  MS_EXCEPTION_IF_NULL(address);
  if (!BindDeviceToCurrentThread()) {
    return false;
  }
  auto device_ptr = mem_manager_->MallocMemFromMemPool(size);
  if (!device_ptr) {
    return false;
  }
  address->ptr_ = device_ptr;
  address->size_ = size;
  address->from_mem_pool_ = true;
  return true;
}

void GPUDeviceContext::FreeMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(address->ptr_);
  if (!address->from_mem_pool()) {
    return;
  }
  mem_manager_->FreeMemFromMemPool(address->ptr_);
  address->ptr_ = nullptr;
}

bool GPUDeviceContext::AllocateContinuousMemory(const std::vector<DeviceAddressPtr> &addr_list, size_t total_size,
                                                const std::vector<size_t> &size_list) const {
  if (!BindDeviceToCurrentThread()) {
    return false;
  }
  return mem_manager_->MallocContinuousMemFromMemPool(addr_list, total_size, size_list);
}

DeviceAddressPtr GPUDeviceContext::CreateDeviceAddress(void *const device_ptr, size_t device_size, const string &format,
                                                       TypeId type_id) const {
  return std::make_shared<GPUDeviceAddress>(device_ptr, device_size, format, type_id);
}

void GPUDeviceContext::OptimizeGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  // Optimization pass which is irrelevant to device type or format.
  OptimizeGraphWithoutDeviceInfo(graph);

  FormatTransformChecker::GetInstance().CheckSupportFormatTransform(graph);
  SetOperatorInfo(graph->execution_order());

  // Optimization pass which is relevant to device type or format.
  OptimizeGraphWithDeviceInfo(graph);

  // Run final optimization.
  opt::CommonFinalOptimization(graph);

  // Graph kernel fusion optimization
  if (context::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    opt::GraphKernelOptimize(graph);
    graph->SetExecOrderByDefault();
  }

  // Assign the stream and insert the send/recv node for all reduce kernel, so must be the last in the optimizer.
  device::gpu::AssignGpuStream(graph);
}

void GPUDeviceContext::OptimizeGraphWithoutDeviceInfo(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  // Operator fusion optimization.
  FuseOperators(graph);

  // Update Graph Dynamic Shape Attr.
  UpdateGraphDynamicShapeAttr(NOT_NULL(graph));
}

void GPUDeviceContext::OptimizeGraphWithDeviceInfo(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // Graph optimization relevant to device data format
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::BatchNormReluFusion>());
  pm->AddPass(std::make_shared<opt::BatchNormReluGradFusion>());
  pm->AddPass(std::make_shared<opt::BatchNormAddReluFusion>());
  pm->AddPass(std::make_shared<opt::PostBatchNormAddReluFusion>());
  pm->AddPass(std::make_shared<opt::BatchNormAddReluGradFusion>());
  pm->AddPass(std::make_shared<opt::InsertFormatTransformOp>());
  pm->AddPass(std::make_shared<opt::RemoveFormatTransformPair>());
  pm->AddPass(std::make_shared<opt::RemoveRedundantFormatTransform>());
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    // Remove node only used by UpdateState, in order to ensure the correct execution sequence in CudnnInplaceAggregate.
    pm->AddPass(std::make_shared<opt::OptimizeUpdateState>());
    pm->AddPass(std::make_shared<opt::CudnnInplaceAggregate>());
  }
  pm->AddPass(std::make_shared<opt::ReluV2Pass>());
  pm->AddPass(std::make_shared<opt::AddReluV2Fusion>());
  pm->AddPass(std::make_shared<opt::AddReluGradV2Fusion>());
  pm->AddPass(std::make_shared<opt::AllReduceFusion>());
  pm->AddPass(std::make_shared<opt::AllGatherFusion>());
  pm->AddPass(std::make_shared<opt::ConcatOutputsForAllGather>());
  pm->AddPass(std::make_shared<opt::GetitemTuple>());
  pm->AddPass(std::make_shared<opt::ReducePrecisionFusion>("reduce_precision"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
  graph->SetExecOrderByDefault();
}

void GPUDeviceContext::FuseOperators(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::MatMulBiasAddFusion>());
  pm->AddPass(std::make_shared<opt::AdamWeightDecayFusion>());
  pm->AddPass(std::make_shared<opt::AdamFusion>());
  pm->AddPass(std::make_shared<opt::ApplyMomentumWeightDecayScaleFusion>());
  pm->AddPass(std::make_shared<opt::ApplyMomentumScaleFusion>());
  pm->AddPass(std::make_shared<opt::ApplyMomentumWeightDecayFusion>());
  if (!context::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    pm->AddPass(std::make_shared<opt::CastAllFusion>("cast_all"));
  }
  pm->AddPass(std::make_shared<opt::CombineMomentumFusion>("combine_momentum"));
  pm->AddPass(std::make_shared<opt::ReplaceMomentumCastFusion>());
  pm->AddPass(std::make_shared<opt::ReplaceAddNFusion>());
  pm->AddPass(std::make_shared<opt::PrintReduceFusion>("print_reduce"));
  pm->AddPass(std::make_shared<opt::BCEWithLogitsLossFusion>());
  pm->AddPass(std::make_shared<opt::InsertCastGPU>("insert_cast_gpu"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
  graph->SetExecOrderByDefault();
}

void GPUDeviceContext::UpdateGraphDynamicShapeAttr(const NotNull<KernelGraphPtr> &graph) const {
  for (const auto &cnode : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (AnfAlgo::IsNodeDynamicShape(cnode)) {
      AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(true), cnode);
      MS_LOG(INFO) << "Set Dynamic Shape Attr to Node:" << cnode->fullname_with_scope();
    }
  }
  graph->UpdateGraphDynamicAttr();
}

namespace {
void RunOpOptimize(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::BCEWithLogitsLossFusion>());
  pm->AddPass(std::make_shared<opt::InsertCastGPU>("insert_cast_gpu"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void RunOpHardwareOptimize(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ReducePrecisionFusion>("reduce_precision"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void RunOpHideNopNode(const KernelGraphPtr &kernel_graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    opt::HideNopNode(kernel_graph.get());
  }
}

void RunOpRemoveNopNode(const KernelGraphPtr &kernel_graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    opt::RemoveNopNode(kernel_graph.get());
  }
}
}  // namespace

void GPUDeviceContext::OptimizeSingleOpGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  RunOpOptimize(graph);

  FormatTransformChecker::GetInstance().CheckSupportFormatTransform(graph);
  SetOperatorInfo(graph->execution_order());

  RunOpHardwareOptimize(graph);

  RunOpHideNopNode(graph);
  RunOpRemoveNopNode(graph);
}

void GPUDeviceContext::SetOperatorInfo(const std::vector<CNodePtr> &nodes) const {
  for (const auto &node : nodes) {
    SetKernelInfo(node);
  }
}

void GPUDeviceContext::CreateKernel(const std::vector<CNodePtr> &nodes) const { CreateGPUKernel(nodes); }

void GPUDeviceContext::UpdateDynamicShape(const CNodePtr &kernel) const {
  MS_EXCEPTION_IF_NULL(kernel);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool is_pynative_infer = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  bool is_pynative_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode;
  if (is_pynative_infer || is_pynative_mode) {
    return;
  }

  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  if (session::AnfRuntimeAlgorithm::GetKernelType(kernel) == KernelType::AKG_KERNEL) {
    MS_LOG(EXCEPTION) << "Akg kernel do not support dynamic shape: " << kernel->fullname_with_scope();
  }

  kernel::GpuKernel *gpu_kernel = dynamic_cast<kernel::GpuKernel *>(kernel_mod);
  MS_EXCEPTION_IF_NULL(gpu_kernel);
  device::DynamicKernelPtr dynamic_kernel = gpu_kernel->DynamicKernel();
  MS_EXCEPTION_IF_NULL(dynamic_kernel);

  dynamic_kernel->InferShape();
  dynamic_kernel->UpdateArgs();
}

bool GPUDeviceContext::LaunchKernel(const CNodePtr &kernel, const std::vector<AddressPtr> &inputs,
                                    const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs,
                                    bool is_dynamic_shape) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_LOG(DEBUG) << "Launch kernel: " << kernel->fullname_with_scope();
  if (!BindDeviceToCurrentThread()) {
    return false;
  }

  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  bool ret = true;
#ifndef ENABLE_SECURITY
  const auto &profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);

  if (!profiler_inst->GetEnableFlag()) {
#endif
    std::lock_guard<std::mutex> locker(launch_mutex_);
    ret = DoLaunchKernel(kernel_mod, inputs, workspace, outputs);
#ifndef ENABLE_SECURITY
  } else {
    std::lock_guard<std::mutex> locker(launch_mutex_);
    ret = LaunchKernelWithProfiling(kernel, inputs, workspace, outputs);
  }
#endif
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
    return false;
  }

  // Sync running.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if ((ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) &&
      ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) && !SyncStream()) {
    return false;
  }

  // Processing after execution of dynamic kernel to update output shape.
  if (is_dynamic_shape) {
    kernel::GpuKernel *gpu_kernel = dynamic_cast<kernel::GpuKernel *>(kernel_mod);
    MS_EXCEPTION_IF_NULL(gpu_kernel);
    gpu_kernel->PostExecute();
  }
  return ret;
}
#ifndef ENABLE_SECURITY
bool GPUDeviceContext::LaunchKernelWithProfiling(const CNodePtr &kernel, const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) const {
  MS_EXCEPTION_IF_NULL(kernel);

  auto kernel_graph = std::dynamic_pointer_cast<KernelGraph>(kernel->func_graph());
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);

  if (profiler::gpu::ProfilingUtils::IsFirstStep(kernel_graph->graph_id())) {
    profiler::gpu::ProfilingTraceInfo profiling_trace =
      profiler::gpu::ProfilingUtils::GetProfilingTraceFromEnv(NOT_NULL(kernel_graph.get()));
    profiler_inst->SetStepTraceOpName(profiling_trace);
  }

  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  profiler_inst->OpDataProducerBegin(kernel->fullname_with_scope(), streams_.front());
  bool ret = DoLaunchKernel(kernel_mod, inputs, workspace, outputs);
  profiler_inst->OpDataProducerEnd();

  auto op_launch_start_end_time = profiler_inst->GetSingleOpLaunchTime();
  std::string op_name = kernel->fullname_with_scope();
  PynativeProfiler::SetDeviceOpNameAndLaunchTimePoint(std::make_pair(op_name, op_launch_start_end_time));
  PynativeProfiler::SetDeviceOpNameAndLaunchCostTime(
    std::make_pair(op_name, op_launch_start_end_time.second - op_launch_start_end_time.first));

  if (profiler_inst->GetSyncEnableFlag()) {
    CHECK_RET_WITH_RETURN_ERROR(SyncStream(), "Profiler SyncStream failed.");
  }
  return ret;
}
#endif
bool GPUDeviceContext::DoLaunchKernel(KernelMod *kernel_mod, const std::vector<AddressPtr> &inputs,
                                      const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) const {
  MS_EXCEPTION_IF_NULL(kernel_mod);
  return kernel_mod->Launch(inputs, workspace, outputs, streams_.front());
}

bool GPUDeviceContext::SyncStream(size_t stream_id) const {
  if (stream_id >= streams_.size()) {
    MS_LOG(EXCEPTION) << "The stream_id: " << stream_id << " is greater than stream array size: " << streams_.size();
  }
  bool result = GPUDeviceManager::GetInstance().SyncStream(streams_[stream_id]);
#ifdef ENABLE_DUMP_IR
  if (!result) {
    mindspore::RDR::TriggerAll();
  }
  // clear RDR gpu memory info
  mindspore::RDR::ClearMemAddressInfo();
#endif
  return result;
}

uint32_t GPUDeviceContext::GetRankID() const {
  const void *collective_handle_ = CollectiveInitializer::instance().collective_handle();
  bool collective_inited = CollectiveInitializer::instance().collective_inited();
  uint32_t rank_id = 0;
  if (collective_inited && collective_handle_ != nullptr) {
    if (!CommManager::GetInstance().GetRankID(kNcclWorldGroup, &rank_id)) {
      MS_LOG(EXCEPTION) << "Failed to get rank id.";
    }
  }
  return rank_id;
}

std::shared_ptr<Bucket> GPUDeviceContext::CreateBucket(uint32_t bucket_id, uint32_t bucket_size) const {
  auto bucket = std::make_shared<GPUBucket>(bucket_id, bucket_size);
  MS_EXCEPTION_IF_NULL(bucket);
  // One computation stream, one communication stream.
  const size_t min_num_of_stream = 2;
  if (min_num_of_stream > streams_.size()) {
    MS_LOG(EXCEPTION) << "The total stream num: " << streams_.size() << " is less than: " << min_num_of_stream;
  }

  bucket->Init({streams_[0]}, {streams_[1]});
  return bucket;
}

bool GPUDeviceContext::BindDeviceToCurrentThread() const {
  if (cur_thread_device_inited) {
    return true;
  }

  if (!CudaDriver::SetDevice(UintToInt(device_context_key_.device_id_))) {
    MS_LOG(ERROR) << "Failed to set device id: " << device_context_key_.device_id_;
    return false;
  }

  cur_thread_device_inited = true;
  return true;
}

MS_REGISTER_DEVICE(kGPUDevice, GPUDeviceContext);
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
