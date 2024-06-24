/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/hal/hardware/gpu_device_context.h"
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#endif
#include <tuple>
#include <utility>
#include "plugin/device/gpu/hal/device/kernel_info_setter.h"
#include "plugin/device/gpu/hal/device/gpu_kernel_build.h"
#include "plugin/device/gpu/hal/device/gpu_device_address.h"
#include "plugin/device/gpu/hal/device/gpu_memory_manager.h"
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"
#include "plugin/device/gpu/hal/device/gpu_stream_assign.h"
#include "include/backend/distributed/init.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "plugin/device/gpu/hal/hardware/gpu_somas.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/hal/hardware/optimizer.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "plugin/device/gpu/hal/profiler/gpu_profiling.h"
#include "plugin/device/gpu/hal/profiler/gpu_profiling_utils.h"
#include "plugin/device/gpu/optimizer/clip_by_norm_fission.h"
#include "include/backend/kernel_graph.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/hal/device/gpu_event.h"
#include "plugin/device/gpu/hal/device/gpu_kernel_task.h"
#include "plugin/device/gpu/hal/device/gpu_hash_table_util.h"
#include "plugin/device/gpu/optimizer/reg_gpu_const_input_to_attr.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "include/common/debug/anf_ir_dump.h"
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/rdr/recorder_manager.h"
#include "debug/rdr/mem_address_recorder.h"
#endif
#include "include/common/utils/comm_manager.h"
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/debugger/debugger.h"
#endif
#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#endif
#include "backend/common/pass/optimize_updatestate.h"
#include "abstract/ops/primitive_infer_map.h"
#include "backend/common/expander/fallback/expander_fallback.h"
#include "backend/common/graph_kernel/value_graph_binder.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/gpu/hal/device/gpu_pin_mem_pool.h"
#include "plugin/device/gpu/hal/device/gpu_device_synchronizer.h"
#include "include/common/profiler.h"
#include "ops/ascend_op_name.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/pipeline/task/kernel_task.h"
#include "runtime/device/move_to.h"
#include "include/backend/mem_reuse/mem_tracker.h"

namespace mindspore {
namespace device {
namespace gpu {
namespace {
const char kModelNameGPU[] = "GPU";
const char kEventOptimizeGraph[] = "OptimizeGraph";
const char kStageOptimizeWithoutDeviceInfo[] = "OptimizeWithoutDeviceInfo";
const char kStageSetKernelInfo[] = "SetKernelInfo";
const char kStageOptimizeWithDeviceInfo[] = "OptimizeWithDeviceInfo";
std::string GetCurrentDir() {
#ifndef _WIN32
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(GetCurrentDir), &dl_info) == 0) {
    MS_LOG(WARNING) << "Get dladdr error";
    return "";
  }
  std::string cur_so_path = dl_info.dli_fname;
  return dirname(cur_so_path.data());
#else
  return "";
#endif
}

runtime::KernelTaskPtr GetTaskByTaskType(const runtime::KernelTaskType &task_type,
                                         const std::shared_ptr<runtime::KernelTaskContext> &task_context) {
  switch (task_type) {
    case runtime::KernelTaskType::kCONTIGUOUS_TASK:
      return std::make_shared<GpuContiguousKernelTask>(task_context);
      break;
    case runtime::KernelTaskType::kCOPY_TASK:
      return std::make_shared<GpuCopyWithSliceKernelTask>(task_context);
      break;
    default:
      MS_LOG(EXCEPTION) << "KernelTaskType is invalid, task_type:" << task_type;
  }
}
}  // namespace
using KernelGraph = mindspore::session::KernelGraph;

static thread_local bool cur_thread_device_inited{false};

void GPUDeviceContext::Initialize() {
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (initialized_) {
    if (!device_res_manager_->BindDeviceToCurrentThread(false)) {
      MS_LOG(EXCEPTION) << "BindDeviceToCurrentThread failed.";
    }
    GPUMemoryAllocator::GetInstance().CheckMaxDeviceMemory();
    return;
  }

  device_res_manager_->Initialize();
  MS_EXCEPTION_IF_NULL(GetKernelExecutor(false));
  GetKernelExecutor(false)->Initialize();
#ifndef ENABLE_SECURITY
  // Dump json config file if dump is enabled.
  uint32_t rank_id = 0;
  if (distributed::collective::CollectiveManager::instance()->need_init()) {
    rank_id = device_context_key().device_id_;
  }

  MS_LOG(INFO) << "Set rank id " << rank_id << " for dumping.";
  auto &json_parser = DumpJsonParser::GetInstance();
  json_parser.Parse();
  json_parser.CopyDumpJsonToDir(rank_id);
  json_parser.CopyMSCfgJsonToDir(rank_id);
#endif
  initialized_ = true;
}

void GPUDeviceResManager::Initialize() {
  // Set device id
  if (distributed::collective::CollectiveManager::instance()->initialized()) {
    DeviceContextKey old_key = device_context_->device_context_key();
    device_context_->device_context_key_.device_id_ =
      distributed::collective::CollectiveManager::instance()->local_rank_id();

    DeviceContextManager::GetInstance().UpdateDeviceContextKey(old_key, device_context_->device_context_key());

    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    ms_context->set_param_inner<uint32_t>(MS_CTX_DEVICE_ID, device_context_->device_context_key().device_id_);
  }

  MS_LOG(INFO) << "Set GPU device id index " << device_context_->device_context_key().device_id_;
  // Set device id and initialize device resource.
  if (!InitDevice()) {
    MS_LOG(EXCEPTION) << "GPU InitDevice failed.";
  }

  // Initialize memory pool.
  mem_manager_ = std::make_shared<GPUMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->Initialize();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    swap_manager_ = std::make_shared<SwapManager>(GPUDeviceManager::GetInstance().default_stream_id(),
                                                  &GPUMemoryAllocator::GetInstance(), &GPUPinMemPool::GetInstance());
  }

  // Initialize NCCL.
  if (distributed::collective::CollectiveManager::instance()->initialized()) {
#if defined(_WIN32)
    MS_LOG(EXCEPTION) << "windows not support nccl.";
#endif
  }
}

bool GPUDeviceResManager::InitDevice() {
  if (GPUDeviceManager::GetInstance().device_count() <= 0) {
    MS_LOG(ERROR) << "No GPU device found.";
    return false;
  }

  if (!GPUDeviceManager::GetInstance().is_device_id_init()) {
    if (!GPUDeviceManager::GetInstance().set_cur_device_id(device_context_->device_context_key().device_id_)) {
      MS_LOG(ERROR) << "Failed to set current device id: "
                    << SizeToInt(device_context_->device_context_key().device_id_);
      return false;
    }
  }
  // Check the Cuda capability
  const float cuda_cap = GET_CUDA_CAP;
  if (cuda_cap < SUPPORTED_CAP) {
    MS_LOG(WARNING) << "The device with Cuda compute capability " << cuda_cap
                    << " is lower than the minimum required capability " << SUPPORTED_CAP
                    << ", this may cause some unexpected problems and severely affect the results. "
                    << "Eg: the outputs are all zeros.\n"
                    << "Device with a compute capability > " << SUPPORTED_CAP << " is required, "
                    << "and it is recommended to use devices with a compute capability >= " << RECOMMEND_SM;
  }

  // Initialize device resource, such as stream, cudnn and cublas handle.
  GPUDeviceManager::GetInstance().InitDevice();
  return true;
}

void GPUDeviceResManager::Destroy() {
  (void)DestroyAllEvents();
  if (DataQueueMgr::GetInstance().IsInit()) {
    if (!DataQueueMgr::GetInstance().IsClosed() && !DataQueueMgr::GetInstance().CloseNotify()) {
      MS_LOG(ERROR) << "Could not close gpu data queue.";
    }
    DataQueueMgr::GetInstance().Release();
  }

  // Release stream, cudnn and cublas handle, etc.
  GPUDeviceManager::GetInstance().ReleaseDevice();

  // Release device memory
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
    mem_manager_ = nullptr;
  }
}

void GPUDeviceContext::Destroy() {
#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  if (debugger && debugger->debugger_enabled()) {
    debugger->SetTrainingDone(true);
    bool ret = debugger->SendMetadata(false);
    if (!ret) {
      MS_LOG(ERROR) << "Failed to SendMetadata when finalize";
    }
  }
#endif
  MS_EXCEPTION_IF_NULL(GetKernelExecutor(false));
  GetKernelExecutor(false)->Destroy();
  device_res_manager_->Destroy();
  initialized_ = false;
}

void *GPUDeviceResManager::AllocateMemory(size_t size, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (!BindDeviceToCurrentThread(false)) {
    return nullptr;
  }
  if (swap_manager_ != nullptr) {
    return swap_manager_->AllocDeviceMemory(size, stream_id);
  }
  return mem_manager_->MallocMemFromMemPool(size, false, false, stream_id);
}

void GPUDeviceResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  MS_EXCEPTION_IF_NULL(ptr);
  mem_manager_->FreeMemFromMemPool(ptr);
}

void GPUDeviceResManager::FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                                          const std::vector<size_t> &keep_addr_sizes) const {
  GPUMemoryAllocator::GetInstance().FreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

bool GPUDeviceResManager::AllocateMemory(DeviceAddress *const &address, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(address);
  auto device_name_in_address = GetDeviceNameByType(static_cast<const DeviceType>(address->GetDeviceType()));
  if (device_name_in_address != device_context_->device_context_key().device_name_) {
    MS_LOG(EXCEPTION) << "The device address type is wrong: type name in address:" << device_name_in_address
                      << ", type name in context:" << device_context_->device_context_key().device_name_;
  }

  if (address->GetPtr() != nullptr) {
    MS_LOG(ERROR) << "Memory leak detected!";
    return false;
  }

  if (!BindDeviceToCurrentThread(false)) {
    return false;
  }

  if (stream_id == UINT32_MAX) {
    stream_id = address->stream_id();
  }

  void *device_ptr;
  if (swap_manager_ != nullptr) {
    device_ptr = swap_manager_->AllocDeviceMemory(address->GetSize(), stream_id);
  } else {
    device_ptr =
      mem_manager_->MallocMemFromMemPool(address->GetSize(), address->from_persistent_mem(), false, stream_id);
  }
  if (!device_ptr) {
    return false;
  }

  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, address, device_ptr);
  return true;
}

std::vector<void *> GPUDeviceResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                                  uint32_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    std::vector<void *> ptr_list;
    return ptr_list;
  }

  // Memory allocation ensures memory alignment.
  std::vector<size_t> align_size_list;
  for (size_t size : size_list) {
    auto align_size = GPUMemoryAllocator::GetInstance().AlignMemorySize(size);
    (void)align_size_list.emplace_back(align_size);
  }
  if (swap_manager_ != nullptr) {
    return swap_manager_->AllocDeviceContinuousMem(align_size_list, stream_id);
  }
  return mem_manager_->MallocContinuousMemFromMemPool(align_size_list, stream_id);
}

// Relevant function to manage memory statistics
size_t GPUDeviceResManager::GetTotalMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalMemStatistics();
}
size_t GPUDeviceResManager::GetTotalUsedMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalUsedMemStatistics();
}
size_t GPUDeviceResManager::GetTotalIdleMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalIdleMemStatistics();
}
size_t GPUDeviceResManager::GetTotalEagerFreeMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalEagerFreeMemStatistics();
}
size_t GPUDeviceResManager::GetUsedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetUsedMemPeakStatistics();
}
size_t GPUDeviceResManager::GetReservedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetReservedMemPeakStatistics();
}
std::unordered_map<std::string, std::size_t> GPUDeviceResManager::GetBlockCountsStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetBlockCountsStatistics();
}
std::unordered_map<std::string, std::size_t> GPUDeviceResManager::GetBlockUnitSizeStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetBlockUnitSizeStatistics();
}
std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
GPUDeviceResManager::GetCommonMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetCommonMemBlocksInfoStatistics();
}
std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
GPUDeviceResManager::GetPersistentMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetPersistentMemBlocksInfoStatistics();
}
void GPUDeviceResManager::ResetMaxMemoryReserved() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetMaxMemoryReserved();
}
void GPUDeviceResManager::ResetMaxMemoryAllocated() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetMaxMemoryAllocated();
}

namespace {
// Create data in user data for device address.
void SetUserData(DeviceAddress *device_address, const UserDataPtr &user_data) {
  MS_EXCEPTION_IF_NULL(device_address);
  MS_EXCEPTION_IF_NULL(user_data);

  device_address->set_user_data(user_data);
  const auto &user_data_type = user_data->get<UserDataType>(kUserDataType);
  if (user_data_type == nullptr) {
    return;
  }
  if (*user_data_type == UserDataType::kUserTypeHashTable) {
#if CUDA_VERSION > 11000 && defined(__linux__)
    auto key_type = user_data->get<TypeId>(kHashTableKeyType);
    auto value_type = user_data->get<TypeId>(kHashTableValueType);
    MS_EXCEPTION_IF_NULL(key_type);
    MS_EXCEPTION_IF_NULL(value_type);
    const auto &iter = hashtable_func_list.find({*key_type, *value_type});
    if (iter != hashtable_func_list.end()) {
      return std::get<kSetFuncIndex>(iter->second)(user_data);
    } else {
      MS_LOG(EXCEPTION) << "Unsupported hash table type:" << *key_type << " and:" << *value_type;
    }
#else
    MS_LOG(EXCEPTION) << "Invalid platform or cuda version for gpu hash table.";
#endif
  } else {
    MS_LOG(EXCEPTION) << "Invalid user data type:" << *user_data_type;
  }
}
}  // namespace

void GPUDeviceResManager::MoveTo(const tensor::TensorPtr &src_tensor, const tensor::TensorPtr &dst_tensor,
                                 const std::string &to, bool blocking, bool *return_self) {
  device::MoveTo(src_tensor, dst_tensor, to, blocking, return_self);
}

DeviceAddressPtr GPUDeviceResManager::CreateDeviceAddress(const KernelTensorPtr &kernel_tensor) const {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (kernel_tensor->device_name().empty()) {
    kernel_tensor->set_device_name(device_context_->device_context_key().device_name_);
    kernel_tensor->set_device_id(device_context_->device_context_key().device_id_);
  }
  auto device_address = std::make_shared<GPUDeviceAddress>(kernel_tensor);

  const auto &user_data = kernel_tensor->user_data();
  if (user_data != nullptr) {
    SetUserData(device_address.get(), user_data);
  }

  device_address->set_device_synchronizer(std::make_shared<GPUDeviceSynchronizer>());
  return device_address;
}

DeviceAddressPtr GPUDeviceResManager::CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector,
                                                          const Format &format, TypeId type_id,
                                                          const std::string &device_name, uint32_t device_id,
                                                          uint32_t stream_id) const {
  return std::make_shared<GPUDeviceAddress>(ptr, size, shape_vector, format, type_id, device_name, device_id,
                                            stream_id);
}

void GPUKernelExecutor::PreprocessBeforeRun(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // somas
  if (ms_context->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL) != kOptimizeO0) {
    auto somas = std::make_shared<GPUSomas>();
    bool ret = somas->Assign(kernel_graph);
    if (ret) {
      MS_LOG(INFO) << "Somas allocate success for graph " << kernel_graph->graph_id()
                   << " somas size: " << kernel_graph->somas_whole_block_size();
    } else if (somas->IsSupportSomas(*kernel_graph)) {
      MS_LOG(WARNING) << "Somas allocate failed for graph " << kernel_graph->graph_id();
    }
  }
  MS_LOG(INFO) << "Status record: end preprocess before run graph. graph id: " << kernel_graph->graph_id();
}

void GPUKernelExecutor::OptimizeGraphWithoutDeviceInfo(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  (void)profiler::CollectHostInfo(kModelNameGPU, kEventOptimizeGraph, kStageOptimizeWithoutDeviceInfo, 1, 0, 0);
  // Operator fusion optimization.
  FuseOperators(graph);
  (void)profiler::CollectHostInfo(kModelNameGPU, kEventOptimizeGraph, kStageOptimizeWithoutDeviceInfo, 1, 0, 1);
}

void GPUKernelExecutor::OptimizeGraphWithDeviceInfo(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  (void)profiler::CollectHostInfo(kModelNameGPU, kEventOptimizeGraph, kStageOptimizeWithDeviceInfo, 1, 0, 0);
  // Graph optimization relevant to device data format
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>("insert_type_transform_op"));
  // ReplaceAddNFusion depends on the input expansion of AddN, so must be after the operator select.
  pm->AddPass(std::make_shared<opt::ReplaceAddNFusion>());
  // PrintReduceFusion depends on the input expansion of Print, so must be after the operator select.
  pm->AddPass(std::make_shared<opt::PrintReduceFusion>("print_reduce"));

  // The fusion operator generates a new primitive and can't be supported in dynamic shape scene.
  if (!graph->is_dynamic_shape()) {
    pm->AddPass(std::make_shared<opt::BatchNormReluFusion>());
    pm->AddPass(std::make_shared<opt::BatchNormSiluFusion>());
    pm->AddPass(std::make_shared<opt::BatchNormReluGradFusion>());
    pm->AddPass(std::make_shared<opt::BatchNormSiluGradFusion>());
    pm->AddPass(std::make_shared<opt::BatchNormAddReluFusion>());
    pm->AddPass(std::make_shared<opt::PostBatchNormAddReluFusion>());
    pm->AddPass(std::make_shared<opt::BatchNormAddReluGradFusion>());
    pm->AddPass(std::make_shared<opt::InsertFormatTransformOp>());
    pm->AddPass(std::make_shared<opt::RemoveFormatTransformPair>());
    pm->AddPass(std::make_shared<opt::RemoveRedundantFormatTransform>());
    if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
      // Remove node only used by UpdateState, in order to ensure the correct execution sequence in
      // CudnnInplaceAggregate.
      pm->AddPass(std::make_shared<opt::OptimizeUpdateState>());
      pm->AddPass(std::make_shared<opt::CudnnInplaceAggregate>());
    }
  }

  pm->AddPass(std::make_shared<opt::AllReduceFusion>());
  pm->AddPass(std::make_shared<opt::AdjustDependForParallelOptimizerRecomputeAllGather>());
  pm->AddPass(std::make_shared<opt::AllGatherFusion>());
  pm->AddPass(std::make_shared<opt::ConcatOutputsForAllGather>());
  pm->AddPass(std::make_shared<opt::GetitemTuple>());
  pm->AddPass(std::make_shared<opt::ReducePrecisionFusion>("reduce_precision"));
  pm->AddPass(std::make_shared<opt::InsertTensorMoveForCommunication>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
  graph->SetExecOrderByDefault();
  (void)profiler::CollectHostInfo(kModelNameGPU, kEventOptimizeGraph, kStageOptimizeWithDeviceInfo, 1, 0, 1);
}

void GPUKernelExecutor::FuseOperators(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  // In the dynamic shape scene, the infershape stage needs to call the primitive infer function.
  // When the fusion operator generates a new primitive, but there
  // is no corresponding primitive infer function, an error will occur.
  // Therefore, this kind of scene does not support dynamic shape.
  if (graph->is_dynamic_shape()) {
    MS_LOG(INFO) << "Dynamic shape skip some fusion pass";
    pm->AddPass(std::make_shared<opt::InsertCastGPU>("insert_cast_gpu"));
  } else {
    pm->AddPass(std::make_shared<opt::ClipByNormFission>());
    pm->AddPass(std::make_shared<opt::MatMulBiasAddFusion>());
    pm->AddPass(std::make_shared<opt::AdamWeightDecayFusion>());
    pm->AddPass(std::make_shared<opt::AdamFusion>());
    pm->AddPass(std::make_shared<opt::AllToAllFusion>());
    pm->AddPass(std::make_shared<opt::ApplyMomentumWeightDecayScaleFusion>());
    pm->AddPass(std::make_shared<opt::ApplyMomentumScaleFusion>());
    pm->AddPass(std::make_shared<opt::ApplyMomentumWeightDecayFusion>());
    if (!graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
      pm->AddPass(std::make_shared<opt::CastAllFusion>("cast_all"));
    }
    pm->AddPass(std::make_shared<opt::CombineOptimizerFusion>(kCombineOptimizerOpName));
    pm->AddPass(std::make_shared<opt::ReplaceMomentumCastFusion>());
    pm->AddPass(std::make_shared<opt::InsertCastGPU>("insert_cast_gpu"));
    pm->AddPass(std::make_shared<opt::NeighborExchangeV2Fusion>());
    pm->AddPass(std::make_shared<opt::NeighborExchangeV2GradFusion>());
    pm->AddPass(std::make_shared<opt::BiasDropoutAddFusion>());
  }
  pm->AddPass(std::make_shared<opt::DynamicSequenceOpsAdaptation>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
  graph->SetExecOrderByDefault();
}

namespace {
void RunOpOptimize(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
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

bool CheckSupportBackoff(const KernelGraphPtr &graph, const CNodePtr &node,
                         const std::pair<std::string, ExceptionType> &failure_info) {
  MS_EXCEPTION_IF_NULL(node);
  // The single op does not support the backoff ability.
  if (!AnfAlgo::IsNodeSupportKernelSelectBackoff(node, graph)) {
    return false;
  }
  const auto &kernel_name = common::AnfAlgo::GetCNodeName(node);
  const auto &kernel_attrs = kernel::NativeCpuKernelMod::GetCpuSupportedList(kernel_name);
  // CPU also doesn't support the kernel.
  if (kernel_attrs.empty() || kernel::IsKernelObjectTypeNotSupportedError(failure_info.first)) {
    return false;
  }
  return true;
}

void SetBackoffInfo(const CNodePtr &node, const std::pair<std::string, ExceptionType> &failure_info) {
  MS_LOG(INFO) << "GPU doesn't support the kernel " << common::AnfAlgo::GetCNodeName(node)
               << " and will try to backoff on CPU.";
  // Mark kernel selection backoff info.
  AnfAlgo::SetKernelSelectBackoffInfo(node, failure_info);
}

// Mark the kernel backoff with failure info when setting operator info fails.
void HandleKernelSelectFailure(const KernelGraphPtr &graph, const CNodePtr &node,
                               const std::pair<std::string, ExceptionType> &failure_info) {
  if (!CheckSupportBackoff(graph, node, failure_info)) {
    MS_EXCEPTION(failure_info.second) << "#umsg#Kernel select failed:#umsg#" << failure_info.first;
  }
  SetBackoffInfo(node, failure_info);
}

bool TryExpandFallback(const KernelGraphPtr &graph, const CNodePtr &node,
                       const std::pair<std::string, ExceptionType> &failure_info) {
  auto f = [ori_node = node, &failure_info, &graph](const CNodePtr &basic_op) mutable {
    MS_EXCEPTION_IF_NULL(basic_op);
    auto res = SetKernelInfoWithMsg(basic_op);
    if (res.first.empty()) {
      // select gpu kernel success.
      return true;
    }
    // select gpu kernel failed, first try to use CPU kernel for original op.
    if (ori_node != nullptr) {
      MS_LOG(DEBUG) << "The basic op " << basic_op->fullname_with_scope()
                    << " select kernel failed. Try to backoff on CPU for original op "
                    << ori_node->fullname_with_scope();
      if (CheckSupportBackoff(graph, ori_node, failure_info)) {
        // original node use cpu kernel, stop expanding.
        MS_LOG(DEBUG) << "Original op " << ori_node->fullname_with_scope() << " use CPU kernel.";
        return false;
      } else {
        MS_LOG(DEBUG) << "Failed to backoff on CPU for original op " << ori_node->fullname_with_scope()
                      << ", try to backoff on CPU for basic op " << basic_op->fullname_with_scope();
      }
      // only try once for original node.
      ori_node = nullptr;
    } else {
      MS_LOG(DEBUG) << "The basic op " << basic_op->fullname_with_scope()
                    << " select kernel failed, try to backoff on CPU";
    }
    // Original op cannot backoff on CPU, try to use CPU kernel for current op.
    if (CheckSupportBackoff(graph, basic_op, res)) {
      AnfAlgo::SetKernelSelectBackoffInfo(basic_op, res);
      MS_LOG(DEBUG) << "The basic op " << basic_op->fullname_with_scope() << " use CPU kernel.";
      return true;
    }
    return false;
  };
  return expander::TryExpandCNode(node, f);
}

// Before creating the kernel, check whether the node has completed the operator selection. If not, the operator
// selection needs to be performed to set kernel info.
void SetKernelInfoBeforeCreateKernel(const std::vector<CNodePtr> &nodes) {
  for (const auto &node : nodes) {
    auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
    // Kernel selection process.
    if (build_info == nullptr) {
      const auto &failure_info = SetKernelInfoWithMsg(node);
      if (!failure_info.first.empty()) {
        const auto &kernel_graph = AnfAlgo::FetchKernelGraph(node.get());
        HandleKernelSelectFailure(kernel_graph, node, failure_info);
      }
    } else if (!build_info->valid()) {
      // Judge whether match strictly between kernel build info and supported kernel attrs.
      const auto &kernel_attr = kernel::GetKernelAttrFromBuildInfo(build_info);
      const auto &supported_kernel_attrs =
        kernel::NativeGpuKernelModFactory::GetInstance().GetGpuSupportedList(common::AnfAlgo::GetCNodeName(node));
      const auto &match_result = kernel::MatchKernelAttrStrict(kernel_attr, supported_kernel_attrs);
      if (!match_result.first) {
        auto attr_info = kernel::FetchPrintInfoByKernelAttr(kernel_attr);
        std::string error_info =
          "Unsupported op [" + common::AnfAlgo::GetCNodeName(node) + "] on GPU, node attr: " + attr_info;
        const auto &kernel_graph = AnfAlgo::FetchKernelGraph(node.get());
        HandleKernelSelectFailure(kernel_graph, node, {error_info, NotSupportError});
      }
      build_info->set_valid(true);
    }
  }
}

// Check whether mutex exists for a stream.
std::pair<bool, std::mutex *> CheckStreamMutexExist(
  const void *stream, const mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> &mtxs_for_streams,
  std::shared_mutex *shd_mtx) {
  MS_EXCEPTION_IF_NULL(stream);
  MS_EXCEPTION_IF_NULL(shd_mtx);
  std::shared_lock<std::shared_mutex> shd_lock(*shd_mtx);
  auto iter = mtxs_for_streams.find(stream);
  if (iter != mtxs_for_streams.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    return std::make_pair(true, iter->second.get());
  }
  return std::make_pair(false, nullptr);
}

// Create a mutex for stream.
std::mutex *CreateStreamMutex(const void *stream, std::shared_mutex *shd_mtx,
                              mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> *mtxs_for_streams) {
  MS_EXCEPTION_IF_NULL(stream);
  MS_EXCEPTION_IF_NULL(shd_mtx);
  MS_EXCEPTION_IF_NULL(mtxs_for_streams);

  std::unique_lock<std::shared_mutex> unq_lock(*shd_mtx);
  auto ret_pair = mtxs_for_streams->emplace(stream, std::make_shared<std::mutex>());

  MS_EXCEPTION_IF_NULL(ret_pair.first->second);
  return ret_pair.first->second.get();
}

// The launch kernel is thread-unsafe, and the behavior of delivering the kernel launch to the same stream requires
// lock protection, need to create a separate lock for each stream.
// for GPU, The cublas handle is not thread safety specifically, it is not recommended that multiple threads access the
// same cublas handle at the same time, so need the launch mutex when multiple threads launch the cublas kernels.
std::lock_guard<std::mutex> LockLaunchKernel(const void *stream) {
  MS_EXCEPTION_IF_NULL(stream);
  // Read-write lock for accessing mtxs_for_streams map.
  // When the lock of each stream is created, mtxs_for_streams can be accessed concurrently to improve performance.
  static std::shared_mutex shd_mtx;
  static mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> mtxs_for_streams;

  std::mutex *stream_mtx;
  // Check whether mutex exists for a stream.
  std::pair<bool, std::mutex *> ret_pair = CheckStreamMutexExist(stream, mtxs_for_streams, &shd_mtx);
  if (ret_pair.first) {
    stream_mtx = ret_pair.second;
  } else {
    // Create a mutex for stream.
    stream_mtx = CreateStreamMutex(stream, &shd_mtx, &mtxs_for_streams);
  }

  MS_EXCEPTION_IF_NULL(stream_mtx);
  // Lock kernel launch for the stream.
  return std::lock_guard<std::mutex>(*stream_mtx);
}
}  // namespace

void GPUKernelExecutor::Initialize() {
  if (initialized_) {
    return;
  }
  res_manager_ = dynamic_cast<GPUDeviceResManager *>(device_context_->device_res_manager_.get());
  MS_EXCEPTION_IF_NULL(res_manager_);
  initialized_ = true;
}

void GPUKernelExecutor::Destroy() {
  if (!initialized_) {
    return;
  }
  res_manager_ = nullptr;
  initialized_ = false;
}

void GPUKernelExecutor::OptimizeGraph(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto enable_lazy_inline = ms_context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (enable_lazy_inline) {
    MS_LOG(EXCEPTION) << "GPU does not support the lazy_inline feature, "
                      << "please do not mark @lazy_inline in cell's __init__ func.";
  }
  if (kernel_graph->is_from_single_op()) {
    RunOpOptimize(kernel_graph);

    FormatTransformChecker::GetInstance().CheckSupportFormatTransform(kernel_graph);
    SetOperatorInfo(kernel_graph);

    RunOpHardwareOptimize(kernel_graph);

    RunOpHideNopNode(kernel_graph);
    RunOpRemoveNopNode(kernel_graph);
    UpdateKernelRefInfo(kernel_graph);
    AssignDefaultGpuStream(kernel_graph);
  } else {
    // Optimization pass which is irrelevant to device type or format.
    OptimizeGraphWithoutDeviceInfo(kernel_graph);

    FormatTransformChecker::GetInstance().CheckSupportFormatTransform(kernel_graph);
    SetOperatorInfo(kernel_graph);

    // SetOperatorInfo may generate new node, so need set kernel object type again.
    kernel_graph->SetKernelObjectTypesForUnrealNodes();
#ifdef ENABLE_DUMP_IR
    if (ms_context->CanDump(kIntroductory)) {
      DumpIR("hwopt_comm_after_kernel_select_" + graph->ToString() + ".ir", graph, true);
    }
#endif

    // Optimization pass which is relevant to device type or format.
    OptimizeGraphWithDeviceInfo(kernel_graph);

    // Run final optimization.
    opt::CommonFinalOptimization(kernel_graph);

    // Graph kernel fusion optimization
    if (graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
      graphkernel::GraphKernelOptimize(kernel_graph);
      kernel_graph->SetExecOrderByDefault();
    }

    // Assign the stream and insert the send/recv node for all reduce kernel, so must be the last in the optimizer.
    device::gpu::AssignGpuStream(kernel_graph);
  }
}

void GPUKernelExecutor::UpdateKernelRefInfo(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    const std::string &op_name = common::AnfAlgo::GetCNodeName(kernel);

    auto kernel_attr_list = kernel::NativeGpuKernelModFactory::GetInstance().GetGpuSupportedList(op_name);
    if (kernel_attr_list.empty()) {
      MS_LOG(DEBUG) << "kernel_attr_list is empty";
      return;
    }

    auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel->kernel_info());
    MS_EXCEPTION_IF_NULL(kernel_info);
    // For the same kernel, there are currently no multiple Ref info.
    kernel_info->set_ref_map(kernel_attr_list[0].GetAllOutInRef(), kernel_attr_list[0].GetOutInRefMap());
  }
}

void GPUKernelExecutor::SetOperatorInfo(const KernelGraphPtr &graph) const {
  (void)profiler::CollectHostInfo(kModelNameGPU, kEventOptimizeGraph, kStageSetKernelInfo, 1, 0, 0);
  auto mng = graph->manager();
  if (mng == nullptr) {
    mng = Manage(graph, true);
    graph->set_manager(mng);
  }
  bool do_expand = false;
  auto &node_list = graph->execution_order();
  for (auto &node : node_list) {
    const auto &failure_info = SetKernelInfoWithMsg(node);
    if (failure_info.first.empty()) {
      continue;
    }
    auto expand_ret = TryExpandFallback(graph, node, failure_info);
    if (expand_ret) {
      MS_LOG(INFO) << failure_info.first << " but expand success.";
      do_expand = true;
    } else {
      HandleKernelSelectFailure(graph, node, failure_info);
    }
  }
  if (do_expand) {
    graphkernel::BindValueToGraph().Run(graph);
    graph->SetExecOrderByDefault();
  }
  (void)profiler::CollectHostInfo(kModelNameGPU, kEventOptimizeGraph, kStageSetKernelInfo, 1, 0, 1);
}

kernel::KernelModPtr GPUKernelExecutor::CreateKernelMod(const std::string &op_name) const {
  return kernel::Factory<kernel::NativeGpuKernelMod>::Instance().Create(op_name);
}

void GPUKernelExecutor::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  SetKernelInfoBeforeCreateKernel(nodes);
  CreateGPUKernel(nodes);
}

bool GPUKernelExecutor::LaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod,
                                     void *stream) const {
  MS_EXCEPTION_IF_NULL(kernel);
  if (!res_manager_->BindDeviceToCurrentThread(false)) {
    return false;
  }
  bool ret = true;

#ifndef ENABLE_SECURITY
  const auto &profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);

  if (!profiler_inst->GetEnableFlag() || !profiler_inst->GetOpTimeFlag()) {
#endif
    auto lock = LockLaunchKernel(stream);
    ret = DoLaunchKernel(kernel, inputs, workspace, outputs, kernel_mod, stream);
#ifndef ENABLE_SECURITY
  } else {
    auto lock = LockLaunchKernel(stream);
    ret = LaunchKernelWithProfiling(kernel, inputs, workspace, outputs, kernel_mod, stream);
  }
#endif
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
    return false;
  }

  return ret;
}

#ifndef ENABLE_SECURITY
bool GPUKernelExecutor::LaunchKernelWithProfiling(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &workspace,
                                                  const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod,
                                                  void *stream) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(stream);

  auto kernel_graph = std::dynamic_pointer_cast<KernelGraph>(kernel->func_graph());
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);

  if (profiler::gpu::ProfilingUtils::IsFirstStep(kernel_graph->graph_id())) {
    profiler::gpu::ProfilingTraceInfo profiling_trace =
      profiler::gpu::ProfilingUtils::GetProfilingTraceFromEnv(NOT_NULL(kernel_graph.get()));
    profiler_inst->SetStepTraceOpName(profiling_trace);
  }

  profiler_inst->OpDataProducerBegin(kernel->fullname_with_scope(), GPUDeviceManager::GetInstance().default_stream());
  bool ret = DoLaunchKernel(kernel, inputs, workspace, outputs, kernel_mod, stream);
  profiler_inst->OpDataProducerEnd();
  profiler_inst->RecordFrameWorkInfo(kernel);

  auto op_launch_start_end_time = profiler_inst->GetSingleOpLaunchTime();
  MS_LOG(DEBUG) << "Launch kernel:" << kernel->fullname_with_scope() << " cost:"
                << (op_launch_start_end_time.second - op_launch_start_end_time.first) / kBasicTimeTransferUnit;

  if (profiler_inst->GetSyncEnableFlag()) {
    CHECK_RET_WITH_RETURN_ERROR(res_manager_->SyncAllStreams(), "Profiler SyncStream failed.");
  }
  return ret;
}
#endif

bool GPUKernelExecutor::DoLaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod,
                                       void *stream) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  MS_EXCEPTION_IF_NULL(stream);

  uint64_t start_time = 0;
  PROFILER_START(start_time);
  auto ret = kernel_mod->Launch(inputs, workspace, outputs, stream);
  // Sync running.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if ((ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) &&
      ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) && !res_manager_->SyncAllStreams()) {
    return false;
  }
  PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunch,
               kernel->fullname_with_scope(), false);

  return ret;
}

bool GPUDeviceResManager::CreateStream(size_t *stream_id) const {
  return GPUDeviceManager::GetInstance().CreateStream(stream_id);
}

bool GPUDeviceResManager::CreateStreamWithPriority(size_t *stream_id, int32_t priority) const {
  return GPUDeviceManager::GetInstance().CreateStreamWithPriority(stream_id, priority);
}

size_t GPUDeviceResManager::QueryStreamSize() const { return GPUDeviceManager::GetInstance().QueryStreamSize(); }

std::vector<uint32_t> GPUDeviceResManager::GetStreamIds() const {
  return GPUDeviceManager::GetInstance().GetStreamIds();
}

bool GPUDeviceResManager::single_op_multi_stream_enable() const {
  return GPUDeviceManager::GetInstance().single_op_multi_stream_enable();
}

void GPUDeviceResManager::set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {
  return GPUDeviceManager::GetInstance().set_single_op_multi_stream_enable(single_op_multi_stream_enable);
}

void *GPUDeviceResManager::GetStream(size_t stream_id) const {
  return GPUDeviceManager::GetInstance().GetStream(stream_id);
}

bool GPUDeviceResManager::DestroyStream(size_t stream_id) const {
  return GPUDeviceManager::GetInstance().DestroyStream(stream_id);
}

void GPUDeviceResManager::SetCurrentStreamId(size_t stream_id) {
  GPUDeviceManager::GetInstance().set_current_stream(stream_id);
}

size_t GPUDeviceResManager::GetCurrentStreamId() const { return GPUDeviceManager::GetInstance().current_stream(); }

bool GPUDeviceResManager::QueryStream(size_t stream_id) const {
  return GPUDeviceManager::GetInstance().QueryStream(stream_id);
}

bool GPUDeviceResManager::SyncStream(size_t stream_id) const {
  bool result = GPUDeviceManager::GetInstance().SyncStream(stream_id);
#ifdef ENABLE_DUMP_IR
  if (!result) {
    mindspore::RDR::TriggerAll();
  }
  // clear RDR gpu memory info
  mindspore::RDR::ClearMemAddressInfo();
#endif
  return result;
}

bool GPUDeviceResManager::SyncAllStreams() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Fail to bind device to current thread";
    return false;
  }
  bool result = GPUDeviceManager::GetInstance().SyncAllStreams();
#ifdef ENABLE_DUMP_IR
  if (!result) {
    mindspore::RDR::TriggerAll();
  }
  // clear RDR gpu memory info
  mindspore::RDR::ClearMemAddressInfo();
#endif
  return result;
}
bool GPUDeviceResManager::SyncNotDefaultStreams() const {
  return GPUDeviceManager::GetInstance().SyncNotDefaultStreams();
}

size_t GPUDeviceResManager::DefaultStream() const { return GPUDeviceManager::GetInstance().default_stream_id(); }

uint32_t GPUKernelExecutor::GetRankID() const {
  bool collective_inited = distributed::collective::CollectiveManager::instance()->initialized();
  uint32_t rank_id = 0;
  if (collective_inited) {
    if (!CommManager::GetInstance().GetRankID(kNcclWorldGroup, &rank_id)) {
      MS_LOG(EXCEPTION) << "Failed to get rank id.";
    }
  }
  return rank_id;
}

// cudaEventRecordDefault 0x0 | cudaEventRecordExternal 0x1 | cudaEventWaitExternal 0x1, no need to set again.
DeviceEventPtr GPUDeviceResManager::CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) {
  if (!enable_blocking && !enable_record_wait) {
    MS_LOG(INTERNAL_EXCEPTION) << "Bad parameters, enable_blocking is false and enable_record_wait is false.";
  }
  uint32_t flag = cudaEventDefault;
  flag |= cudaEventDisableTiming;
  flag |= cudaEventBlockingSync;
  return std::make_shared<GpuEvent>(flag);
}

DeviceEventPtr GPUDeviceResManager::CreateEventWithFlag(bool enable_timing, bool blocking) {
  uint32_t flag =
    (blocking ? cudaEventBlockingSync : cudaEventDefault) | (enable_timing ? cudaEventDefault : cudaEventDisableTiming);
  auto event = std::make_shared<GpuEvent>(flag);
  MS_EXCEPTION_IF_NULL(event);
  std::lock_guard<std::mutex> lock(device_events_mutex_);
  device_events_.push_back(event);
  return event;
}

bool GPUKernelExecutor::ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                                          const device::DeviceAddressPtrList &input_addr_list,
                                          const device::DeviceAddressPtrList &output_addr_list,
                                          const size_t &stream_id) const {
  auto stream = GPUDeviceManager::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream);

  auto task_context =
    std::make_shared<runtime::KernelTaskContext>(device_context_, input_addr_list, output_addr_list, stream);

  auto task = GetTaskByTaskType(task_type, task_context);
  MS_EXCEPTION_IF_NULL(task);

  uint64_t start_time = 0;
  PROFILER_START(start_time);
  auto lock = LockLaunchKernel(stream);
  auto ret = task->RunWithRet();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Exec task failed, task_type:" << task_type;
  }

  // Sync running.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if ((ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) &&
      ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) && !res_manager_->SyncAllStreams()) {
    return false;
  }

  runtime::DeviceAddressUtils::ProcessCrossStreamAddress("Contiguous", device_context_, stream_id, input_addr_list,
                                                         output_addr_list);
  PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunch, "Contiguous",
               false);

  return true;
}

bool GPUDeviceResManager::LoadCollectiveCommLib() {
#ifdef ENABLE_MPI
  std::string nvidia_comm_lib_name = GetCurrentDir() + "/gpu" + std::to_string(CUDA_VERSION / 1000) + "." +
                                     std::to_string(CUDA_VERSION / 10 % 10) + "/libnvidia_collective.so";
  auto loader = std::make_shared<CollectiveCommLibLoader>(nvidia_comm_lib_name);
  MS_EXCEPTION_IF_NULL(loader);
  if (!loader->Initialize()) {
    MS_LOG(EXCEPTION) << "Loading NCCL collective library failed.";
    return false;
  }
  void *collective_comm_lib_handle = loader->collective_comm_lib_ptr();
  MS_EXCEPTION_IF_NULL(collective_comm_lib_handle);

  auto instance_func = DlsymFuncObj(communication_lib_instance, collective_comm_lib_handle);
  collective_comm_lib_ = instance_func();
  MS_EXCEPTION_IF_NULL(collective_comm_lib_);
  return true;
#else
  return false;
#endif
}

bool GPUDeviceResManager::BindDeviceToCurrentThread(bool force_bind) const {
  if (cur_thread_device_inited && !force_bind) {
    return true;
  }

  if (!CudaDriver::SetDevice(UintToInt(device_context_->device_context_key().device_id_))) {
    MS_LOG(ERROR) << "Failed to set device id: " << device_context_->device_context_key().device_id_;
    return false;
  }

  cur_thread_device_inited = true;
  return true;
}

DeprecatedInterface *GPUDeviceContext::GetDeprecatedInterface() {
  if (deprecated_interface_ == nullptr) {
    deprecated_interface_ = std::make_unique<GPUDeprecatedInterface>();
  }
  return deprecated_interface_.get();
}

uint32_t GPUDeviceContext::GetDeviceCount() { return IntToUint(CudaDriver::device_count()); }

std::string GPUDeviceContext::GetDeviceName(uint32_t device_id) {
  return GPUdeviceInfo::GetInstance(device_id)->name();
}

std::tuple<int, int> GPUDeviceContext::GetDeviceCapability(uint32_t device_id) {
  int major_sm = GPUdeviceInfo::GetInstance(device_id)->major_sm();
  int minor_sm = GPUdeviceInfo::GetInstance(device_id)->minor_sm();
  return std::make_tuple(major_sm, minor_sm);
}

cudaDeviceProp GPUDeviceContext::GetDeviceProperties(uint32_t device_id) {
  return GPUdeviceInfo::GetInstance(device_id)->properties();
}

std::string GPUDeviceContext::GetArchList() { return STRING_COMPILE_OPT(CUDA_ARCH_LIST); }

std::shared_ptr<void> GPUDeviceResManager::AllocateHostMemory(size_t size) const {
  void *ptr;
  if (CudaDriver::AllocHostPinnedMem(size, &ptr) != size) {
    MS_LOG(ERROR) << "Failed to allow host pinned memory.";
    return nullptr;
  }

  return std::shared_ptr<void>(ptr, [](void *ptr) -> void {
    if (ptr != nullptr) {
      CudaDriver::FreeHostPinnedMem(ptr);
    }
  });
}

MS_REGISTER_DEVICE(kGPUDevice, GPUDeviceContext);
#ifdef WITH_BACKEND
MSCONTEXT_REGISTER_INIT_FUNC(kGPUDevice, [](MsContext *ctx) -> void {
  MS_EXCEPTION_IF_NULL(ctx);
  if (ctx->backend_policy() != "ms") {
    ctx->set_backend_policy("ms");
  }
});
#endif

// Register functions to _c_expression so python hal module could call GPU device interfaces.
void PybindGPUStatelessFunc(py::module *m) {
  MS_EXCEPTION_IF_NULL(m);
  (void)py::class_<cudaDeviceProp>(*m, "cudaDeviceProp", py::module_local())
    .def_readonly("name", &cudaDeviceProp::name)
    .def_readonly("major", &cudaDeviceProp::major)
    .def_readonly("minor", &cudaDeviceProp::minor)
    .def_readonly("is_multi_gpu_board", &cudaDeviceProp::isMultiGpuBoard)
    .def_readonly("is_integrated", &cudaDeviceProp::integrated)
    .def_readonly("multi_processor_count", &cudaDeviceProp::multiProcessorCount)
    .def_readonly("total_memory", &cudaDeviceProp::totalGlobalMem)
    .def_readonly("warp_size", &cudaDeviceProp::warpSize)
    .def("__repr__", [](const cudaDeviceProp &p) {
      std::ostringstream s;
      s << "cudaDeviceProp(name='" << p.name << "', major=" << p.major << ", minor=" << p.minor
        << ", is_multi_gpu_board=" << p.isMultiGpuBoard << ", is_integrated=" << p.integrated
        << ", multi_processor_count=" << p.multiProcessorCount << ", total_memory=" << p.totalGlobalMem / (1024 * 1024)
        << "MB, warp_size=" << p.warpSize << ")";
      return s.str();
    });
  (void)m->def("gpu_get_device_count", &GPUDeviceContext::GetDeviceCount, "Get GPU device count.");
  (void)m->def("gpu_get_device_name", &GPUDeviceContext::GetDeviceName, "Get GPU device name of specified device id.");
  (void)m->def("gpu_get_device_capability", &GPUDeviceContext::GetDeviceCapability,
               "Get GPU major and minor capability of specified device id.");
  (void)m->def("gpu_get_device_properties", &GPUDeviceContext::GetDeviceProperties,
               "Get GPU device properties of specified device id.");
  (void)m->def("gpu_get_arch_list", &GPUDeviceContext::GetArchList, "Get GPU arch list of this MindSpore package.");
}
REGISTER_DEV_STATELESS_FUNC_CB(kGPUDevice, PybindGPUStatelessFunc);
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
