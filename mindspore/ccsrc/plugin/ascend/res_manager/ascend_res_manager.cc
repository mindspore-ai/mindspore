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

#include "plugin/ascend/res_manager/ascend_res_manager.h"
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#endif
#include <utility>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <numeric>
#include <set>
#include <unordered_map>
#include <memory>
#include <map>
#include <string>

#include "ir/tensor_new.h"
#include "hccl/hccl.h"
#include "include/runtime/hardware_abstract/collective/collective_comm_lib_loader.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "plugin/ascend/res_manager/mbuf_manager/tensorreport_utils.h"
#include "plugin/ascend/res_manager/device_context_conf/op_debug_conf.h"
#include "plugin/ascend/res_manager/event/ascend_event.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "plugin/ascend/res_manager/capture_graph/ascend_capture_graph.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_memory_pool.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_pin_mem_pool.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_compiler_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "utils/file_utils.h"
#include "utils/distributed_meta.h"
#include "graph/def_types.h"
#include "acl/acl_rt.h"
#include "plugin/ascend/res_manager/collective/ccool_collective_comm_lib.h"
#include "plugin/ascend/res_manager/collective/multi_ascend_collective_comm_lib.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "plugin/ascend/res_manager/collective/dummy_ascend_collective_comm_lib.h"
#ifdef ENABLE_INTERNAL_KERNELS
#include "plugin/ascend/res_manager/collective/lowlatency_collective_comm_lib.h"
#endif
#include "plugin/ascend/res_manager/hal_manager/ascend_hal_manager.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "mindspore/core/include/device_address/convert_tensor_utils.h"
#include "include/backend/common/ms_device_shape_transfer.h"
#include "include/runtime/hardware_abstract/memory_manager/swap_manager.h"
#include "runtime/hardware_abstract/utils.h"
#include "include/utils/callback.h"
#include "plugin/ascend/res_manager/hal_manager/ascend_err_manager.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr uint32_t kDefaultHcclExecTimeout = 1800;
constexpr size_t kAlignBytes = 32;

// Register callbacks for collective methods.
// These code should be deleted after collective so is extracted.
std::string GetCommName(const std::string &group) {
  if (!common::GetEnv(kSimulationLevel).empty()) {
    return DummyAscendCollectiveCommLib::GetInstance().CommName(group);
  }
  return AscendCollectiveCommLib::GetInstance().CommName(group);
}
REGISTER_COMMON_CALLBACK(GetCommName);

using Callback = std::function<void(void)>;
typedef HcclResult (*HcclSetConfigFunc)(HcclConfig, HcclConfigValue);
std::mutex set_opt_mutex;

void *GetAclFunc(const std::string &lib_path, const std::string &func_name) {
  static auto ascend_path = device::ascend::GetAscendPath();
  auto load_path = ascend_path + "/lib64/" + lib_path;

  auto handler = dlopen(load_path.c_str(), RTLD_LAZY);
  if (handler == nullptr) {
    MS_LOG(INFO) << "Dlopen " << load_path << " failed!" << dlerror();
    return nullptr;
  }

  auto func = dlsym(handler, func_name.c_str());
  if (func == nullptr) {
    MS_LOG(INFO) << "Dlsym " << func_name << " from " << load_path << " failed!" << dlerror();
  }
  return func;
}

void AclrtLaunchCallback(void *user_data) {
  Callback *callback_func = reinterpret_cast<Callback *>(user_data);
  (*callback_func)();
  delete callback_func;
}

static bool initialized_ge = false;

// ge.exec.allow_hf32 default value is "10"(enable Conv, disable Matmul) set by CANN
void SetAscendHF32Config(const std::shared_ptr<MsContext> &ms_context_ptr,
                         std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  std::string allow_matmul_hf32 = ms_context_ptr->get_param<std::string>(MS_CTX_MATMUL_ALLOW_HF32);
  std::string allow_conv_hf32 = ms_context_ptr->get_param<std::string>(MS_CTX_CONV_ALLOW_HF32);
  if (allow_matmul_hf32.empty() && allow_conv_hf32.empty()) {
    MS_LOG(INFO) << "The default value of allow_matmul_hf32 and allow_conv_hf32 are set by CANN.";
  } else if (allow_matmul_hf32.empty() && !allow_conv_hf32.empty()) {
    (*ge_options)["ge.exec.allow_hf32"] = allow_conv_hf32 + std::string("0");
  } else if (!allow_matmul_hf32.empty() && allow_conv_hf32.empty()) {
    (*ge_options)["ge.exec.allow_hf32"] = std::string("1") + allow_matmul_hf32;
  } else {
    (*ge_options)["ge.exec.allow_hf32"] = allow_conv_hf32 + allow_matmul_hf32;
  }

  MS_LOG(INFO) << "allow_matmul_hf32: " << allow_matmul_hf32 << ", allow_conv_hf32: " << allow_conv_hf32;
}

void SetAscendConfig(const std::shared_ptr<MsContext> &ms_context_ptr, std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  MS_EXCEPTION_IF_NULL(ge_options);

  (*ge_options)["ge.topoSortingMode"] = "0";
  // disable RemoveSameConstPass, it will be caused the communication failed on multi-card.
  (*ge_options)["ge.disableOptimizations"] = "RemoveSameConstPass";

  (*ge_options)["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  MS_LOG(INFO) << "Set GE topo mode to memory-priority.";

  (*ge_options)["ge.exec.staticMemoryPolicy"] = "2";
  MS_LOG(INFO) << "Set staticMemoryPolicy to default mode 2.";

  if (ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE) != "") {
    (*ge_options)["ge.jit_compile"] = ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE);
    MS_LOG(INFO) << "Set jit_compile " << ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE) << ".";
  } else {
    (*ge_options)["ge.jit_compile"] = "2";
    MS_LOG(INFO) << "The default value of jit_compile is set to 2.";
  }

  SetAscendHF32Config(ms_context_ptr, ge_options);

  if (ms_context_ptr->get_param<std::string>(MS_CTX_OP_PRECISION_MODE) != "") {
    (*ge_options)["ge.exec.op_precision_mode"] = ms_context_ptr->get_param<std::string>(MS_CTX_OP_PRECISION_MODE);
    MS_LOG(INFO) << "Set op_precision_mode " << ms_context_ptr->get_param<std::string>(MS_CTX_OP_PRECISION_MODE) << ".";
  }
}

void SetHcclOptions(const std::shared_ptr<MsContext> &inst_context, std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(inst_context);
  MS_EXCEPTION_IF_NULL(ge_options);
  auto env_table_file = common::GetEnv("MINDSPORE_HCCL_CONFIG_PATH");
  if (env_table_file.empty()) {
    env_table_file = common::GetEnv("RANK_TABLE_FILE");
  }
  auto simulation_level = common::GetEnv(kSimulationLevel);
  if (!simulation_level.empty()) {
    env_table_file = "";
  }
  auto env_rank_id = common::GetEnv("RANK_ID");
  auto env_device_id = std::to_string(inst_context->get_param<uint32_t>(MS_CTX_DEVICE_ID));
  auto env_cluster_info = common::GetEnv("HELP_CLUSTER");
  auto enable_hccl = inst_context->get_param<bool>(MS_CTX_ENABLE_HCCL);
  auto escluster_config_path = common::GetEnv("ESCLUSTER_CONFIG_PATH");

  MS_LOG(INFO) << "Values for hccl options: env_table_file[" << env_table_file << "], simulation_level["
               << simulation_level << "], env_rank_id[" << env_rank_id << "], env_device_id[" << env_device_id
               << "], enable_hccl[" << enable_hccl << "], UseDynamicCluster[" << common::UseDynamicCluster() << "].";
  if (enable_hccl &&
      (!(env_table_file.empty() || env_rank_id.empty()) || !(env_cluster_info.empty() || env_rank_id.empty()) ||
       hccl::HcclAdapter::GetInstance().UseHcclCM()) &&
      !(common::UseDynamicCluster() && !env_table_file.empty())) {
    MS_LOG(INFO) << "Initialize Ge for distribute parameter";
    if (!env_table_file.empty()) {
      MS_LOG(INFO) << "Use hccl, make sure hccl lib is set in OPTION_EXEC_EXTERN_PLUGIN_PATH.";
      (*ge_options)["ge.exec.rankTableFile"] = env_table_file;
    } else if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
      hccl::HcclAdapter::AddCMEnvToHcclOption(ge_options);
    }

    (*ge_options)["ge.exec.isUseHcom"] = "1";
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    (*ge_options)["ge.exec.rankId"] = env_rank_id;
    (*ge_options)["ge.exec.podName"] = env_rank_id;
  } else if (!escluster_config_path.empty()) {
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    (*ge_options)["ge.exec.rankTableFile"] = env_table_file;
    (*ge_options)["ge.exec.rankId"] = env_rank_id;
  } else {
    // device id is still needed for non-distribute case
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    MS_LOG(INFO) << "No hccl mode. If use hccl, make sure [RANK_TABLE_FILE,RANK_ID,DEVICE_ID] all be set in ENV.";
  }
}

void GetGeGlobalOptions(std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(ge_options);
  auto ms_context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context_ptr);

  SetHcclOptions(ms_context_ptr, ge_options);
  (*ge_options)["ge.exec.jobId"] = "0";
  MS_LOG(INFO) << "Set ge.exec.jobId to default value 0";

  auto proto_lib_path = common::GetEnv("OPTION_PROTO_LIB_PATH");
  if (!proto_lib_path.empty()) {
    char real_path[PATH_MAX] = {0};
    if (realpath(proto_lib_path.c_str(), real_path)) {
      proto_lib_path = real_path;
      (*ge_options)["ge.opsProtoLibPath"] = proto_lib_path;
    }
  } else {
    MS_LOG(INFO) << "Got empty proto lib path, cannot set ge.opsProtoLibPath.";
  }

  SetAscendConfig(ms_context_ptr, ge_options);

  auto op_debug_level = common::GetEnv("MS_COMPILER_OP_LEVEL");
  if (!op_debug_level.empty()) {
    (*ge_options)["ge.opDebugLevel"] = op_debug_level;
    MS_LOG(INFO) << "Use MS_COMPILER_OP_LEVEL, op debug level:" << op_debug_level;
  }

  // Enable the global variable acc may cause accuracy problems in train+eval
  (*ge_options)["ge.exec.variable_acc"] = "0";

  // ge heterogeneous mode
  if (ms_context_ptr->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    (*ge_options)["ge.socVersion"] = "Ascend310P3";
  }

  // enable overflow detection
  (*ge_options)["ge.exec.overflow"] = "1";
  // enable deterministic
  (*ge_options)[::ge::DETERMINISTIC] = ms_context_ptr->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON" ? "1" : "0";
  MS_LOG(INFO) << "Set ge::DETERMINISTIC to " << (*ge_options)[::ge::DETERMINISTIC];
}

void SetPassthroughGeOptions(std::string option_level, OptionMap *options) {
  const auto &new_options = AnfAlgo::GetGeOptions(option_level);
  for (auto &[key, value] : new_options) {
    (*options)[key] = value;
    MS_LOG(INFO) << "Set ge " << option_level << " option: {" << key << ", " << value << "}";
  }
}

size_t GetCommonAlignSize(size_t input_size) {
  return ((input_size + kMemAlignSize + kAlignBytes - 1) / kMemAlignSize) * kMemAlignSize;
}

size_t GetCommunicationAlignSize(size_t input_size) {
  return ((input_size + kMemAlignSize - 1) / kMemAlignSize) * kMemAlignSize + kTwiceMemAlignSize;
}
}  // namespace

std::function<CollectiveCommunicationLib *(void)> gLoadCollectiveCommLibCallback;
void RegisterLoadCollectiveCallback(const std::function<CollectiveCommunicationLib *(void)> &func) {
  gLoadCollectiveCommLibCallback = func;
}

void *PinMemoryAllocator::Alloc(size_t size, uint32_t stream_id) {
  MS_EXCEPTION_IF_NULL(swap_manager_);
  auto host_ptr = swap_manager_->AllocHostMemory(size, stream_id);
  if (host_ptr == nullptr) {
    MS_LOG(ERROR) << "Allocate pin memory failed, size: " << size;
  }
  return host_ptr;
}

bool PinMemoryAllocator::Free(void *address_ptr) {
  MS_EXCEPTION_IF_NULL(swap_manager_);
  swap_manager_->FreeHostMemory(address_ptr);
  return true;
}

bool PinMemoryAllocator::IsPinned() { return true; }

void AscendResManager::Initialize() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  device_id_ = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  if (initialized_) {
    AscendHalManager::GetInstance().SetContextForce(device_id_);
    return;
  }

  // init error_manager
  if (!ErrorManagerAdapter::Init()) {
    MS_LOG(WARNING) << "Init ErrorManager failed.";
  }

  // init device
  AscendHalManager::GetInstance().InitDevice(device_id_);
  AscendStreamMng::GetInstance().CreateDefaultStream();

  (void)AscendMemAdapter::GetInstance()->Initialize();
  swap_manager_ = std::make_shared<SwapManager>(kDefaultStreamIndex, &AscendMemoryPool::GetInstance(),
                                                &AscendPinMemPool::GetInstance());
  // set timeout
  auto op_debug_conf = device::ascend::OpDebugConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_debug_conf);
  uint32_t op_execute_timeout = op_debug_conf->execute_timeout();
  std::string hccl_exec_timeout = common::GetEnv("HCCL_EXEC_TIMEOUT");
  uint32_t notify_wait_timeout;
  if (hccl_exec_timeout.empty()) {
    notify_wait_timeout = kDefaultHcclExecTimeout;
  } else {
    try {
      notify_wait_timeout = std::stoi(hccl_exec_timeout);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "Parse environment variable HCCL_EXEC_TIMEOUT failed, value" << hccl_exec_timeout
                        << ", msg: " << e.what();
    }
  }
  if (op_execute_timeout >= notify_wait_timeout) {
    MS_LOG(INFO) << "OpExecuteTimeout should be less than NotifyWaitTimeout, but got OpExecuteTimeout "
                 << op_execute_timeout << ", notify_wait_timeout " << notify_wait_timeout << "."
                 << "1. You can set OpExecuteTimeout via mindspore.set_context(op_timeout=int)."
                 << "2. You can set NotifyWaitTimeout via environment variable HCCL_EXEC_TIMEOUT. ";
  }
  // 310P does not contain the following interfaces
  if (ms_context->ascend_soc_version() != "ascend310p" && ms_context->ascend_soc_version() != "ascend310b") {
    const uint32_t reserve_time = 180;
    uint32_t op_wait_timeout = notify_wait_timeout + reserve_time;
    device::ascend::AscendHalManager::GetInstance().SetOpWaitTimeout(op_wait_timeout);
    device::ascend::AscendHalManager::GetInstance().SetOpExecuteTimeOut(op_execute_timeout);
  }

  enable_memory_tracker_ = device::tracker::MemTrackerManager::GetInstance().IsEnabled();
  pin_mem_allocator_ = std::make_shared<PinMemoryAllocator>(swap_manager_);
  shared_mem_allocator_ = SharedMemoryAllocator::getInstance();
  symmetric_memory_allocator_ = SymmetricMemoryAllocator::GetInstance();
  initialized_ = true;
}

void AscendResManager::DestroyHccl() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
    MS_LOG(INFO) << "Hccl is not enabled, no need to close.";
    return;
  }

  if (common::GetEnv(kSimulationLevel).empty() &&
      !device::ascend::AscendCollectiveCommLib::GetInstance().DestroyHcclComm()) {
    MS_LOG(WARNING) << "Hccl destroy failed.";
    return;
  }
  MS_LOG(INFO) << "Hccl destroy successful.";
  context_ptr->set_param<bool>(MS_CTX_ENABLE_HCCL, false);
}

void AscendResManager::Destroy() {
  if (!initialized_) {
    AscendHalManager::GetInstance().SetContextForce(device_id_);
    return;
  }
  // To avoid call aclrtProcessReport after process exit, we should to clear all callback threads first.
  AscendStreamMng::GetInstance().Clear();

  (void)DestroyAllEvents();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // destroy hccl things
  static auto destroy_hccl_watchdog_all_handlers_cb = GET_COMMON_CALLBACK(DestroyWatchDogAllHandlers, void);
  if (destroy_hccl_watchdog_all_handlers_cb != nullptr) {
    destroy_hccl_watchdog_all_handlers_cb();
  }

  // DestroyHccl must be called before FreeDeviceMemory, watch_hccl_dog and hccl_adapter are in this function
  (void)DestroyHccl();

  AscendStreamMng::GetInstance().DestroyAllRtEvents();
  if (!AscendStreamMng::GetInstance().DestroyAllStreams()) {
    MS_LOG(EXCEPTION) << "Fail to destroy all streams when reset device.";
  }
  // Release memory.
  AscendMemoryPool::GetInstance().ReleaseDeviceRes();
  (void)AscendMemAdapter::GetInstance()->DeInitialize();
  (void)ErrorManagerAdapter::Finalize();

  // Finalize symmetric memory manager.
  symmetric_memory_allocator_->FinalizeSymmetricMemoryManager();

  // All unmap/free operations will fail after calling aclrtResetDevice in ResetDevice,
  // so it must be called before that.
  AscendVmmAdapter::GetInstance().ClearAllMemory();
  AscendHalManager::GetInstance().ResetDevice(device_id_);

  initialized_ = false;
}
void AscendResManager::ResetDynamicMemory() { AscendMemAdapter::GetInstance()->ResetDynamicMemory(); }

bool AscendResManager::IsEnableVmm() const { return AscendVmmAdapter::IsEnabled(); }

bool AscendResManager::AllocateMemory(DeviceAddress *const &address, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(address);

  if (address->device_pointer()->ptr() != nullptr) {
    MS_LOG(ERROR) << "Memory leak detected in device address:" << address->ToString();
    return false;
  }
  AscendHalManager::GetInstance().SetContext(device_id_);
  if (stream_id == UINT32_MAX) {
    stream_id = address->stream_id();
  }

  void *device_ptr = nullptr;
  const auto &allocator = address->allocator();
  if (MS_UNLIKELY(allocator != nullptr)) {
    device_ptr = allocator->Alloc(address->GetSize(), stream_id);
  } else {
    auto align_size = GetCommonAlignSize(address->GetSize());
    device_ptr = AscendMemoryPool::GetInstance().AllocTensorMem(align_size, address->from_persistent_mem(),
                                                                address->need_recycle(), stream_id);
  }

  if (!device_ptr) {
    return false;
  }

  address->set_from_mem_pool(true);
  address->set_ptr(device_ptr);
  if (enable_memory_tracker_) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, address, device_ptr);
  }
  return true;
}
void *AscendResManager::AllocateMemory(size_t size, uint32_t stream_id) const {
  AscendHalManager::GetInstance().SetContext(device_id_);
  auto align_size = GetCommonAlignSize(size);
  return AscendMemoryPool::GetInstance().AllocTensorMem(align_size, false, false, stream_id);
}

void *AscendResManager::AllocateMemory(size_t size, bool from_persistent_mem, bool need_recycle, uint32_t stream_id) {
  AscendHalManager::GetInstance().SetContext(device_id_);
  auto align_size = GetCommonAlignSize(size);
  return AscendMemoryPool::GetInstance().AllocTensorMem(align_size, from_persistent_mem, need_recycle, stream_id);
}

size_t AscendResManager::GetMaxUsedMemorySize() const { return AscendMemoryPool::GetInstance().GetMaxUsedMemSize(); }

void AscendResManager::FreeMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  void *device_ptr = address->GetMutablePtr();
  auto allocator = address->allocator();

  if (device_ptr == nullptr) {
    return;
  }

  if (!address->from_mem_pool()) {
    MS_LOG(DEBUG) << "device address:" << address << " ptr:" << device_ptr << " not from pool";
    return;
  }

  MS_LOG(DEBUG) << "Free memory from device address:" << address << " ptr:" << device_ptr;
  if (MS_UNLIKELY(allocator != nullptr)) {
    allocator->Free(device_ptr);
  } else {
    FreeMemory(device_ptr);
  }
  address->set_ptr(nullptr);
}

bool AscendResManager::IsNotEventUsedMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  void *device_ptr = address->GetMutablePtr();
  auto allocator = address->allocator();
  if (MS_UNLIKELY(allocator != nullptr)) {
    return true;
  }
  return AscendMemoryPool::GetInstance().IsNotEventUsedTensorMem(device_ptr);
}

void AscendResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  AscendMemoryPool::GetInstance().FreeTensorMem(ptr);
}

void AscendResManager::FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                                       const std::vector<size_t> &keep_addr_sizes) const {
  AscendMemoryPool::GetInstance().FreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

void AscendResManager::DefragMemory() { AscendMemoryPool::GetInstance().DefragMemory(); }

// Relevant function to manage memory statistics
size_t AscendResManager::GetTotalMemStatistics() const { return AscendMemoryPool::GetInstance().TotalMemStatistics(); }

size_t AscendResManager::GetTotalUsedMemStatistics() const {
  return AscendMemoryPool::GetInstance().TotalUsedMemStatistics();
}

size_t AscendResManager::GetTotalIdleMemStatistics() const {
  return AscendMemoryPool::GetInstance().TotalIdleMemStatistics();
}

size_t AscendResManager::GetTotalEagerFreeMemStatistics() const {
  return AscendMemoryPool::GetInstance().TotalEagerFreeMemStatistics();
}

size_t AscendResManager::GetUsedMemPeakStatistics() const {
  return AscendMemoryPool::GetInstance().MaxMemAllocatedStatistics();
}

size_t AscendResManager::GetReservedMemPeakStatistics() const {
  return AscendMemoryPool::GetInstance().MaxMemReservedStatistics();
}

std::unordered_map<std::string, std::size_t> AscendResManager::GetBlockCountsStatistics() const {
  return AscendMemoryPool::GetInstance().BlockCountsStatistics();
}

std::unordered_map<std::string, std::size_t> AscendResManager::GetBlockUnitSizeStatistics() const {
  return AscendMemoryPool::GetInstance().BlockUnitSizeStatistics();
}

DeviceMemInfo AscendResManager::GetCommonMemBlocksInfoStatistics() const {
  return AscendMemoryPool::GetInstance().CommonMemBlocksInfoStatistics();
}

DeviceMemInfo AscendResManager::GetPersistentMemBlocksInfoStatistics() const {
  return AscendMemoryPool::GetInstance().PersistentMemBlocksInfoStatistics();
}

void AscendResManager::ResetMaxMemoryReserved() {
  auto memory_pool = GetMemoryPool();
  MS_EXCEPTION_IF_NULL(memory_pool);
  memory_pool->ResetMaxMemReserved();
}

void AscendResManager::ResetMaxMemoryAllocated() {
  auto memory_pool = GetMemoryPool();
  MS_EXCEPTION_IF_NULL(memory_pool);
  memory_pool->ResetMaxMemAllocated();
}

size_t AscendResManager::EmptyCache() {
  auto memory_pool = GetMemoryPool();
  MS_EXCEPTION_IF_NULL(memory_pool);
  return memory_pool->EmptyCache();
}

void AscendResManager::SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
  if (stream == nullptr) {
    auto ret_rt_memcpy =
      CALL_ASCEND_API(aclrtMemcpy, device_ptr, mem_size, host_ptr, mem_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret_rt_memcpy != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "SwapIn aclrtMemcpy failed.";
    }
  } else {
    auto ret_rt_memcpy =
      CALL_ASCEND_API(aclrtMemcpyAsync, device_ptr, mem_size, host_ptr, mem_size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (ret_rt_memcpy != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "SwapIn aclrtMemcpyAsync failed.";
    }
    if (CALL_ASCEND_API(aclrtSynchronizeStreamWithTimeout, stream, -1) != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "Call runtime aclrtSynchronizeStreamWithTimeout error.";
    }
  }
}

void AscendResManager::SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
  if (stream == nullptr) {
    auto ret_rt_memcpy =
      CALL_ASCEND_API(aclrtMemcpy, host_ptr, mem_size, device_ptr, mem_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret_rt_memcpy != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "SwapOut aclrtMemcpy failed.";
    }
  } else {
    auto ret_rt_memcpy =
      CALL_ASCEND_API(aclrtMemcpyAsync, host_ptr, mem_size, device_ptr, mem_size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret_rt_memcpy != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "SwapOut aclrtMemcpyAsync failed.";
    }
    if (CALL_ASCEND_API(aclrtSynchronizeStreamWithTimeout, stream, -1) != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "Call runtime aclrtSynchronizeStreamWithTimeout error.";
    }
  }
}

std::vector<void *> AscendResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                               uint32_t stream_id) const {
  AscendHalManager::GetInstance().SetContext(device_id_);

  std::vector<size_t> aligned_size_list;
  for (auto size : size_list) {
    auto align_size = GetCommonAlignSize(size);
    aligned_size_list.emplace_back(align_size);
  }
  return AscendMemoryPool::GetInstance().AllocContinuousTensorMem(aligned_size_list, stream_id);
}

bool AscendResManager::SyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                                size_t stream_id, const DeviceAddressExtPtr &src_ext,
                                const DeviceAddressExtPtr &dst_ext) const {
  MS_EXCEPTION_IF_NULL(dst_device_sync);
  MS_EXCEPTION_IF_NULL(src_device_sync);
  MS_LOG(DEBUG) << "Sync copy from device address:" << src_device_sync->ToString()
                << " src ext:" << (src_ext == nullptr ? " null" : src_ext->ToString())
                << " to:" << dst_device_sync->ToString()
                << " dst ext:" << (dst_ext == nullptr ? " null" : dst_ext->ToString()) << " stream id:" << stream_id;
  if (dst_device_sync->GetDeviceType() == DeviceType::kAscend && src_device_sync->GetDeviceType() == DeviceType::kCPU) {
    return SyncHostToDevice(dst_device_sync, src_device_sync, stream_id, src_ext, dst_ext);
  }
  if (dst_device_sync->GetDeviceType() == DeviceType::kCPU && src_device_sync->GetDeviceType() == DeviceType::kAscend) {
    return SyncDeviceToHost(dst_device_sync, src_device_sync, stream_id, src_ext, dst_ext);
  }
  return SyncDeviceToDevice(dst_device_sync, src_device_sync, stream_id, src_ext, dst_ext);
}

bool AscendResManager::AsyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                                 size_t stream_id, bool keep_src, const DeviceAddressExtPtr &src_ext,
                                 const DeviceAddressExtPtr &dst_ext) const {
  MS_EXCEPTION_IF_NULL(dst_device_sync);
  MS_EXCEPTION_IF_NULL(src_device_sync);
  MS_LOG(DEBUG) << "Sync copy from device address:" << src_device_sync->ToString()
                << " src ext:" << (src_ext == nullptr ? " null" : src_ext->ToString())
                << " to:" << dst_device_sync->ToString()
                << " dst ext:" << (dst_ext == nullptr ? " null" : dst_ext->ToString()) << " stream id:" << stream_id;
  if (dst_device_sync->GetDeviceType() == DeviceType::kAscend && src_device_sync->GetDeviceType() == DeviceType::kCPU) {
    return AsyncHostToDevice(dst_device_sync, src_device_sync, stream_id, keep_src, src_ext, dst_ext);
  }
  if (dst_device_sync->GetDeviceType() == DeviceType::kCPU && src_device_sync->GetDeviceType() == DeviceType::kAscend) {
    return AsyncDeviceToHost(dst_device_sync, src_device_sync, stream_id, src_ext, dst_ext);
  }
  return AsyncDeviceToDevice(dst_device_sync, src_device_sync, stream_id, src_ext, dst_ext);
}

namespace {
bool SyncStreamForCopy(const AscendResManager *const res_manager, size_t stream_id) {
  MS_EXCEPTION_IF_NULL(res_manager);
  bool ret = res_manager->SyncStream(stream_id);
  if (!ret) {
    auto sync_copy_fail_cb = GET_COMMON_CALLBACK(SyncCopyFailCallback, void);
    if (sync_copy_fail_cb != nullptr) {
      sync_copy_fail_cb();
    }
    MS_LOG(EXCEPTION) << "Sync stream error!";
  }
  MS_LOG(DEBUG) << "SyncStream Finish!";
  return true;
}
}  // namespace

DynamicMemPool *AscendResManager::GetMemoryPool() {
  if (MS_UNLIKELY(memory_pool_ == nullptr)) {
    memory_pool_ = &AscendMemoryPool::GetInstance();
  }
  return memory_pool_;
}

uint8_t *AscendResManager::MallocDynamicMem(size_t size, bool communication_mem) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }
  MS_LOG(INFO) << "Malloc Memory for Dynamic: size[" << align_size << "] communication_mem: " << communication_mem;

  uint8_t *alloc_address = AscendMemAdapter::GetInstance()->MallocDynamicDevMem(align_size);
  MS_EXCEPTION_IF_NULL(alloc_address);
  // create protect area [kMemAlignSize -- data -- kMemAlignSize] for communication node memory
  return communication_mem ? alloc_address + kMemAlignSize : alloc_address;
}

bool AscendResManager::SyncDeviceToHost(const DeviceAddressPtr &dst_device_sync,
                                        const DeviceAddressPtr &src_device_sync, size_t stream_id,
                                        const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) const {
  if (!AsyncDeviceToHost(dst_device_sync, src_device_sync, stream_id, src_ext, dst_ext)) {
    return false;
  }
  return SyncStreamForCopy(this, stream_id);
}

bool AscendResManager::SyncHostToDevice(const DeviceAddressPtr &dst_device_sync,
                                        const DeviceAddressPtr &src_device_sync, size_t stream_id,
                                        const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) const {
  if (!AsyncHostToDevice(dst_device_sync, src_device_sync, stream_id, false, src_ext, dst_ext)) {
    return false;
  }
  return SyncStreamForCopy(this, stream_id);
}

bool AscendResManager::SyncDeviceToDevice(const DeviceAddressPtr &dst_device_sync,
                                          const DeviceAddressPtr &src_device_sync, size_t stream_id,
                                          const DeviceAddressExtPtr &src_ext,
                                          const DeviceAddressExtPtr &dst_ext) const {
  if (!AsyncDeviceToDevice(dst_device_sync, src_device_sync, stream_id, src_ext, dst_ext)) {
    return false;
  }
  return SyncStreamForCopy(this, stream_id);
}

namespace {
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

std::lock_guard<std::mutex> LockRuntime(const void *stream) {
  MS_EXCEPTION_IF_NULL(stream);
  // Read-write lock for accessing mtxs_for_streams map.
  // When the lock of each stream is created, mtxs_for_streams can be accessed concurrently to improve performance.
  static std::shared_mutex shd_mtx;
  static mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> mtxs_for_streams;

  std::mutex *stream_mtx = nullptr;
  // Check whether mutex exists for a stream.
  std::pair<bool, std::mutex *> ret_pair = CheckStreamMutexExist(stream, mtxs_for_streams, &shd_mtx);
  if (ret_pair.first) {
    stream_mtx = ret_pair.second;
  } else {
    // Create a mutex for stream.
    stream_mtx = CreateStreamMutex(stream, &shd_mtx, &mtxs_for_streams);
  }

  MS_EXCEPTION_IF_NULL(stream_mtx);
  return std::lock_guard<std::mutex>(*stream_mtx);
}

const std::set<Format> op_need_trans_format = {
  Format::NHWC,       Format::HWCN,        Format::NC1HWC0,       Format::FRACTAL_Z, Format::C1HWNCoC0,
  Format::FRACTAL_NZ, Format::NC1HWC0_C04, Format::FRACTAL_Z_C04, Format::NDC1HWC0,  Format::FRACTAL_Z_3D};

ShapeVector GetDeviceShape(ShapeVector *host_shape, const DeviceAddress *src_device_address,
                           const DeviceAddressExtPtr &src_ext) {
  MS_EXCEPTION_IF_NULL(host_shape);
  ShapeVector device_shape;
  auto node_index = src_device_address->GetNodeIndex();
  if (src_ext->format_ == Format::FRACTAL_NZ || src_ext->format_ == Format::NCDHW) {
    device_shape = trans::TransShapeToDevice(*host_shape, kernel::GetFormatFromEnumToStr(src_ext->format_),
                                             node_index.first, node_index.second, src_ext->dtype_id_);
  } else {
    if (!src_ext->shape_vector_.empty()) {
      host_shape->clear();
      *host_shape = src_ext->shape_vector_;
    }
    *host_shape = trans::PaddingShape(*host_shape, kernel::GetFormatFromEnumToStr(src_ext->format_));
    device_shape = trans::TransShapeToDevice(*host_shape, kernel::GetFormatFromEnumToStr(src_ext->format_),
                                             node_index.first, node_index.second, src_ext->dtype_id_);
  }
  return device_shape;
}
aclrtMemcpyKind CopyTypeToAclType(CopyType copy_type) {
  switch (copy_type) {
    case CopyType::kH2D:
      return aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE;
    case CopyType::kD2H:
      return aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST;
    case CopyType::kD2D:
      return aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE;
    default:
      MS_LOG(EXCEPTION) << "Invalid copy type:" << copy_type;
  }
}
}  // namespace

bool AscendResManager::Copy(void *dst, const void *src, uint64_t size, CopyType kind, size_t stream_id) const {
  BindDeviceToCurrentThread(true);
  if (!BaseCopy(dst, src, size, CopyTypeToAclType(kind), stream_id)) {
    MS_LOG(ERROR) << "Failed to copy from:" << dst << " to:" << src << " size:" << size << " kind:" << kind;
    return false;
  }
  return SyncStreamForCopy(this, stream_id);
}

bool AscendResManager::CopyDirectly(void *dst, size_t dst_size, const void *src, size_t src_size, CopyType kind) const {
  BindDeviceToCurrentThread(false);
  auto ret = CALL_ASCEND_API(aclrtMemcpy, dst, dst_size, src, dst_size, CopyTypeToAclType(kind));
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(WARNING) << "AclrtMemcpy failed, error code: " << ret;
    return false;
  }
  return true;
}

bool AscendResManager::BaseCopy(void *dst, const void *src, uint64_t size, aclrtMemcpyKind kind, size_t stream_id,
                                const DeviceAddressPtr src_device_sync) const {
  MS_LOG(DEBUG) << "Copy from:" << src << " to:" << dst << " size:" << size << " kind:" << kind
                << " stream id:" << stream_id << " src device sync:" << src_device_sync;
  if (size == 0 || common::IsCompileSimulation()) {
    return true;
  }
  if (dst == nullptr || src == nullptr) {
    MS_LOG(ERROR) << "Src ptr:" << src << " or dst ptr:" << dst
                  << " is null, please check the address is set correctly.";
    return false;
  }
  const auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get stream by id:" << stream_id;
  }
  LockRuntime(stream);
  auto ret_rt_memcpy = CALL_ASCEND_API(aclrtMemcpyAsync, dst, size, src, size, kind, stream);
  if (ret_rt_memcpy != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call runtime rtMemcpyAsync error, src ptr:" << src << " dst ptr:" << dst << " size:" << size
                  << " kind:" << kind << " stream id:" << stream_id << " ret:" << ret_rt_memcpy;
    return false;
  }

  // Check keep host address for host to device copy.
  if (src_device_sync == nullptr) {
    return true;
  }
  std::function<void(void)> callback_func = [src_device_sync, stream_id]() {
    // Clear tensor_data automatically.
    MS_LOG(DEBUG) << "Callback_func exec, device sync:" << src_device_sync
                  << " use count:" << src_device_sync.use_count() << " stream id:" << stream_id;
  };

  if (!LaunchCallback(callback_func, stream_id)) {
    MS_LOG(EXCEPTION) << "LaunchCallback failed, stream id:" << stream_id;
  }
  return true;
}

bool AscendResManager::CopyDeviceToHostForDiffFormat(const DeviceAddress *dst_device_address,
                                                     const DeviceAddress *src_device_address, size_t stream_id,
                                                     const DeviceAddressExtPtr &src_ext,
                                                     const DeviceAddressExtPtr &dst_ext) const {
  MS_LOG(DEBUG) << "Copy device to host for different format, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  const auto &src_format = src_ext->format_;
  if (op_need_trans_format.find(src_format) == op_need_trans_format.end()) {
    MS_LOG(ERROR) << "Can not find format transfer function for format:" << src_format
                  << " in device address:" << src_device_address->ToString();
    return false;
  }

  // Sync device to host.
  auto host_tmp = std::vector<uint8_t>(src_device_address->GetSize());
  if (!BaseCopy(host_tmp.data(), src_device_address->GetDevicePtr(), src_device_address->GetSize(),
                ACL_MEMCPY_DEVICE_TO_HOST, stream_id)) {
    MS_LOG(ERROR) << "Failed async copy for format transform, src device address:" << src_device_address->ToString();
    return false;
  }
  if (!SyncStreamForCopy(this, stream_id)) {
    MS_LOG(ERROR) << "Failed sync stream : " << stream_id;
    return false;
  }
  // Trans shape.
  ShapeVector host_shape = dst_ext->shape_vector_;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  auto device_shape = GetDeviceShape(&host_shape, src_device_address, src_ext);
  MS_LOG(DEBUG) << "Host shape:" << host_shape << " device shape:" << device_shape << " format:" << src_ext->format_;
  auto node_index = src_device_address->GetNodeIndex();
  if (src_ext->dtype_id_ == dst_ext->dtype_id_) {
    const trans::FormatArgs format_args{host_tmp.data(),   src_device_address->GetSize(),
                                        kOpFormat_NCHW,    kernel::GetFormatFromEnumToStr(src_ext->format_),
                                        host_shape,        device_shape,
                                        src_ext->dtype_id_};
    if (!trans::TransFormatFromDeviceToHost(format_args, dst_device_address->GetDevicePtr(), node_index.first,
                                            node_index.second)) {
      MS_LOG(ERROR) << "Trans format failed for dst device tensor:" << dst_device_address->ToString();
      return false;
    }
    return true;
  }
  const trans::FormatArgs format_args{host_tmp.data(),   src_device_address->GetSize(),
                                      kOpFormat_NCHW,    kernel::GetFormatFromEnumToStr(src_format),
                                      host_shape,        device_shape,
                                      src_ext->dtype_id_};
  auto trans_format_host = std::vector<uint8_t>(src_device_address->GetSize());
  if (!trans::TransFormatFromDeviceToHost(format_args, trans_format_host.data(), node_index.first, node_index.second)) {
    MS_LOG(ERROR) << "Trans format failed for dst device tensor:" << dst_device_address->ToString();
    return false;
  }
  auto shape_size = abstract::ShapeSize(host_shape);
  const trans::TypeIdArgs type_args{trans_format_host.data(), shape_size, src_ext->dtype_id_, dst_ext->dtype_id_,
                                    dst_device_address->GetSize()};
  if (!trans::TransDataType(type_args, dst_device_address->GetDevicePtr())) {
    MS_LOG(ERROR) << "Trans data type failed for dst device tensor:" << dst_device_address->ToString();
    return false;
  }
  return true;
}

bool AscendResManager::CopyDeviceToHostForDiffType(const DeviceAddress *dst_device_address,
                                                   const DeviceAddress *src_device_address, size_t stream_id,
                                                   const DeviceAddressExtPtr &src_ext,
                                                   const DeviceAddressExtPtr &dst_ext) const {
  MS_LOG(DEBUG) << "Copy device to host for different type, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  // Sync device to host.
  auto host_tmp = std::vector<uint8_t>(src_device_address->GetSize());
  if (!BaseCopy(host_tmp.data(), src_device_address->GetDevicePtr(), src_device_address->GetSize(),
                ACL_MEMCPY_DEVICE_TO_HOST, stream_id)) {
    MS_LOG(ERROR) << "Failed async copy for type transform, src device address:" << src_device_address->ToString();
    return false;
  }
  if (!SyncStreamForCopy(this, stream_id)) {
    MS_LOG(ERROR) << "Failed sync stream : " << stream_id;
    return false;
  }
  if (src_ext->dtype_id_ == kNumberTypeFloat32 && dst_ext->dtype_id_ == kNumberTypeFloat64) {
    if (src_device_address->GetSize() / sizeof(float) != dst_device_address->GetSize() / sizeof(double)) {
      MS_LOG(ERROR) << "Invalid src_size for device address" << src_device_address->ToString()
                    << ", dst_size for device address" << dst_device_address->ToString();
      return false;
    }
    FloatToDouble(dst_device_address->GetDevicePtr(), host_tmp.data(), src_device_address->GetSize() / sizeof(float));
    return true;
  }
  auto host_shape = dst_ext->shape_vector_;
  auto shape_size = abstract::ShapeSize(host_shape);
  const trans::TypeIdArgs type_args{host_tmp.data(), shape_size, src_ext->dtype_id_, dst_ext->dtype_id_,
                                    src_device_address->GetSize()};

  if (!trans::TransDataType(type_args, dst_device_address->GetDevicePtr())) {
    MS_LOG(ERROR) << "Trans data type failed for dst device address:" << dst_device_address->ToString();
    return false;
  }
  return true;
}

bool AscendResManager::AsyncDeviceToHost(const DeviceAddressPtr &dst_device_sync,
                                         const DeviceAddressPtr &src_device_sync, size_t stream_id,
                                         const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) const {
  const auto &dst_device_address = dynamic_cast<const DeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const DeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  auto is_contigous = [](const TensorStorageInfoPtr &info) {
    return SizeOf(info->shape) == SizeOf(info->ori_shape) && info->is_contiguous;
  };
  if ((src_device_address->GetTensorStorageInfo() != nullptr &&
       !is_contigous(src_device_address->GetTensorStorageInfo())) ||
      (dst_device_address->GetTensorStorageInfo() != nullptr &&
       !is_contigous(dst_device_address->GetTensorStorageInfo()))) {
    MS_LOG(WARNING) << "Invalid sync device to host for tensor storage info in device address:"
                    << src_device_address->ToString() << " and:" << dst_device_address->ToString();
  }
  BindDeviceToCurrentThread(false);
  if (src_ext == nullptr || dst_ext == nullptr) {
    return BaseCopy(dst_device_address->GetDevicePtr(), src_device_address->GetDevicePtr(),
                    dst_device_address->GetSize(), ACL_MEMCPY_DEVICE_TO_HOST, stream_id);
  }

  // Check format.
  static const std::set<Format> basic_format = {Format::NCHW, Format::DEFAULT_FORMAT, Format::NCDHW, Format::ND};
  if (basic_format.find(src_ext->format_) == basic_format.end()) {
    MS_LOG(DEBUG) << "Can not copy from device to host directly, format is different, src:"
                  << src_device_address->ToString() << " metadata:" << src_ext->ToString()
                  << " dst:" << dst_device_address->ToString() << " metadata:" << dst_ext->ToString();
    return CopyDeviceToHostForDiffFormat(dst_device_address, src_device_address, stream_id, src_ext, dst_ext);
  }

  // Check type.
  if (src_ext->dtype_id_ != dst_ext->dtype_id_) {
    MS_LOG(DEBUG) << "Can not copy from device to host directly, type is different, src:"
                  << src_device_address->ToString() << " metadata:" << src_ext->ToString()
                  << " dst:" << dst_device_address->ToString() << " metadata:" << dst_ext->ToString();
    return CopyDeviceToHostForDiffType(dst_device_address, src_device_address, stream_id, src_ext, dst_ext);
  }

  MS_LOG(DEBUG) << "Copy device to host, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  return BaseCopy(dst_device_address->GetDevicePtr(), src_device_address->GetDevicePtr(), dst_device_address->GetSize(),
                  ACL_MEMCPY_DEVICE_TO_HOST, stream_id);
}

bool AscendResManager::CopyHostToDevice(const DeviceAddress *dst_device_address,
                                        const DeviceAddress *src_device_address, const DeviceAddressExtPtr &src_ext,
                                        const DeviceAddressExtPtr &dst_ext, const void *src, uint64_t size,
                                        aclrtMemcpyKind kind, size_t stream_id,
                                        const DeviceAddressPtr src_device_sync) const {
  if (dst_ext != nullptr && dst_ext->dtype_id_ != kObjectTypeString) {
    return BaseCopy(dst_device_address->GetDevicePtr(), src, size, kind, stream_id, src_device_sync);
  }
  // NOTE: For string type, ge::StringHead.len does not include '\0', since kernel_tensor allocated size including
  // '\0', see method `CreateDeviceAddressForScalarAndString` defined in `device_address_utils.cc`, and method
  // `PrepareDataForStringValue` defined in `data_prepare_actor.cc`, so here pass `size - 1` to `head.len`.
  // NOTE: method `CopyHostToDevice` can be triggered from the two scenarios as below:
  // 1. method `CopyNoneTensorDataToDevice` in `device_address_utils.cc` passes a kernel tensor, the parameter
  // `size` include `ge::StringHead`
  // 2. method `PrepareDataForStringValue` in `data_prepare_actor.cc` passes a raw string, the parameter `size` does
  // not include `ge::StringHead`
  if (src_device_address->GetSize() == dst_device_address->GetSize() && size >= sizeof(ge::StringHead)) {
    size -= sizeof(ge::StringHead);
    MS_LOG(DEBUG) << "Skip string head size:" << sizeof(ge::StringHead)
                  << " for src device address:" << src_device_address->ToString();
  }
  ge::StringHead head{.addr = sizeof(ge::StringHead), .len = static_cast<int64_t>(size) - 1};
  // sync string head info from device to host
  if (!BaseCopy(dst_device_address->GetDevicePtr(), &head, sizeof(ge::StringHead), ACL_MEMCPY_HOST_TO_DEVICE,
                stream_id)) {
    MS_LOG(ERROR) << "Copy string head failed for device address:" << dst_device_address->ToString();
    return false;
  }
  SyncStreamForCopy(this, stream_id);
  // sync string body (real contents) from device to host
  if (!BaseCopy(static_cast<void *>(static_cast<char *>(dst_device_address->GetDevicePtr()) + sizeof(ge::StringHead)),
                src, size, ACL_MEMCPY_HOST_TO_DEVICE, stream_id, src_device_sync)) {
    MS_LOG(ERROR) << "Copy string failed from device address:" << src_device_address->ToString()
                  << " to:" << dst_device_address->ToString();
    return false;
  }
  MS_LOG(DEBUG) << "Copy string info to device, ge::StringHead.len=" << head.len
                << ", text=" << std::string(static_cast<const char *>(src), head.len)
                << ", device_addr=" << dst_device_address->GetDevicePtr();
  return true;
}

bool AscendResManager::CopyHostToDeviceForDiffFormat(const DeviceAddress *dst_device_address,
                                                     const DeviceAddress *src_device_address, size_t stream_id,
                                                     const DeviceAddressExtPtr &src_ext,
                                                     const DeviceAddressExtPtr &dst_ext) const {
  MS_LOG(DEBUG) << "Copy host to device for different format, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  const auto &dst_format = dst_ext->format_;
  if (op_need_trans_format.find(dst_format) == op_need_trans_format.end()) {
    MS_LOG(ERROR) << "Can not find format transfer function for format:" << dst_format
                  << " dst device address:" << dst_device_address->ToString();
    return false;
  }

  ShapeVector host_shape = src_ext->shape_vector_;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  auto node_index = dst_device_address->GetNodeIndex();
  std::vector<int64_t> device_shape;
  if (dst_format != Format::FRACTAL_NZ) {
    host_shape = trans::PaddingShape(host_shape, kernel::GetFormatFromEnumToStr(dst_format));
    MS_LOG(DEBUG) << "Padding shape from:" << src_ext->shape_vector_ << " to:" << host_shape
                  << " for device address:" << src_device_address->ToString();
  }
  device_shape = trans::TransShapeToDevice(host_shape, kernel::GetFormatFromEnumToStr(dst_format), node_index.first,
                                           node_index.second, dst_ext->dtype_id_);
  MS_LOG(DEBUG) << "Host shape:" << host_shape << " device shape:" << device_shape
                << " for device address:" << dst_device_address->ToString();
  // Trans type.
  std::vector<uint8_t> tmp_host_for_type_trans;
  void *tmp_host_ptr = src_device_address->GetDevicePtr();
  if (src_ext->dtype_id_ != dst_ext->dtype_id_) {
    MS_LOG(DEBUG) << "Convert type src:" << src_device_address->ToString() << " metadata:" << src_ext->ToString()
                  << " dst:" << dst_device_address->ToString() << " metadata:" << dst_ext->ToString();
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{src_device_address->GetMutablePtr(), shape_size, src_ext->dtype_id_,
                                      dst_ext->dtype_id_, src_device_address->GetSize()};
    tmp_host_for_type_trans = std::vector<uint8_t>(dst_device_address->GetSize());
    tmp_host_ptr = tmp_host_for_type_trans.data();
    auto ret = trans::TransDataType(type_args, tmp_host_ptr);
    if (!ret) {
      MS_LOG(ERROR) << "Trans data type failed for dst device address:" << dst_device_address->ToString();
      return false;
    }
  }

  // Trans format.
  const trans::FormatArgs format_args{tmp_host_ptr,      dst_device_address->GetSize(),
                                      kOpFormat_NCHW,    kernel::GetFormatFromEnumToStr(dst_format),
                                      host_shape,        device_shape,
                                      dst_ext->dtype_id_};
  auto host_tmp = std::vector<uint8_t>(dst_device_address->GetSize());
  if (!trans::TransFormat(format_args, host_tmp.data(), node_index.first, node_index.second)) {
    MS_LOG(ERROR) << "Trans format failed.";
    return false;
  }

  bool ret = CopyHostToDevice(dst_device_address, src_device_address, src_ext, dst_ext, host_tmp.data(),
                              dst_device_address->GetSize(), ACL_MEMCPY_HOST_TO_DEVICE, stream_id);
  if (!ret) {
    MS_LOG(ERROR) << "Failed async copy";
    return false;
  }
  ret = SyncStreamForCopy(this, stream_id);
  if (!ret) {
    MS_LOG(ERROR) << "Failed sync stream";
    return false;
  }
  return true;
}

bool AscendResManager::CopyHostToDeviceForDiffType(const DeviceAddress *dst_device_address,
                                                   const DeviceAddress *src_device_address, size_t stream_id,
                                                   const DeviceAddressExtPtr &src_ext,
                                                   const DeviceAddressExtPtr &dst_ext) const {
  MS_LOG(DEBUG) << "Copy host to device for different type, src device address:" << src_device_address->ToString()
                << " metadata:" << src_ext->ToString() << " dst device address:" << dst_device_address->ToString()
                << " metadata:" << dst_ext->ToString() << " stream id:" << stream_id;
  std::vector<uint8_t> host_tmp = std::vector<uint8_t>(dst_device_address->GetSize());
  if (dst_ext->dtype_id_ == kNumberTypeFloat32 && src_ext->dtype_id_ == kNumberTypeFloat64) {
    if (src_device_address->GetSize() / sizeof(double) != dst_device_address->GetSize() / sizeof(float)) {
      MS_INTERNAL_EXCEPTION(ArgumentError) << "Invalid src_size for device address" << src_device_address->ToString()
                                           << ", dst_size for device address" << dst_device_address->ToString();
    }
    DoubleToFloat(host_tmp.data(), src_device_address->GetDevicePtr(), dst_device_address->GetSize() / sizeof(float));
  } else {
    ShapeVector host_shape = src_ext->shape_vector_;
    if (host_shape.empty()) {
      (void)host_shape.emplace_back(1);
    }
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{src_device_address->GetDevicePtr(), shape_size, src_ext->dtype_id_,
                                      dst_ext->dtype_id_, src_device_address->GetSize()};
    if (!trans::TransDataType(type_args, host_tmp.data())) {
      MS_LOG(ERROR) << "Trans data type failed for device address:" << dst_device_address->ToString();
      return false;
    }
  }
  // Sync device to host.
  if (!CopyHostToDevice(dst_device_address, src_device_address, src_ext, dst_ext, host_tmp.data(),
                        dst_device_address->GetSize(), ACL_MEMCPY_HOST_TO_DEVICE, stream_id)) {
    MS_LOG(ERROR) << "Failed async copy for src device address:" << src_device_address->ToString()
                  << " dst device address:" << dst_device_address->ToString();
    return false;
  }
  if (!SyncStreamForCopy(this, stream_id)) {
    MS_LOG(ERROR) << "Failed sync stream : " << stream_id;
    return false;
  }
  return true;
}

bool AscendResManager::AsyncHostToDevice(const DeviceAddressPtr &dst_device_sync,
                                         const DeviceAddressPtr &src_device_sync, size_t stream_id, bool keep_src,
                                         const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) const {
  const auto &dst_device_address = dynamic_cast<const DeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const DeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);

  auto is_contigous = [](const TensorStorageInfoPtr &info) {
    return SizeOf(info->shape) == SizeOf(info->ori_shape) && info->is_contiguous;
  };
  if ((src_device_address->GetTensorStorageInfo() != nullptr &&
       !is_contigous(src_device_address->GetTensorStorageInfo())) ||
      (dst_device_address->GetTensorStorageInfo() != nullptr &&
       !is_contigous(dst_device_address->GetTensorStorageInfo()))) {
    MS_LOG(EXCEPTION) << "Invalid sync host to device for tensor storage info in device address:"
                      << src_device_address->ToString() << " and:" << dst_device_address->ToString();
  }
  BindDeviceToCurrentThread(false);
  if (src_ext == nullptr || dst_ext == nullptr) {
    return BaseCopy(dst_device_address->GetDevicePtr(), src_device_address->GetDevicePtr(),
                    src_device_address->GetSize(), ACL_MEMCPY_HOST_TO_DEVICE, stream_id, src_device_sync);
  }

  // Check format.
  static const std::set<Format> basic_format = {Format::NCHW, Format::DEFAULT_FORMAT, Format::NCDHW, Format::ND};
  if (basic_format.find(dst_ext->format_) == basic_format.end() && src_ext->format_ != dst_ext->format_) {
    MS_LOG(DEBUG) << "Can not copy from host to device directly, format is different, src:"
                  << src_device_address->ToString() << " metadata:" << src_ext->ToString()
                  << " dst:" << dst_device_address->ToString() << " metadata:" << dst_ext->ToString();
    return CopyHostToDeviceForDiffFormat(dst_device_address, src_device_address, stream_id, src_ext, dst_ext);
  }

  // Check type.
  if (src_ext->dtype_id_ != dst_ext->dtype_id_) {
    MS_LOG(DEBUG) << "Can not copy from host to device directly, type is different, src:"
                  << src_device_address->ToString() << " metadata:" << src_ext->ToString()
                  << " dst:" << dst_device_address->ToString() << " metadata:" << dst_ext->ToString();
    return CopyHostToDeviceForDiffType(dst_device_address, src_device_address, stream_id, src_ext, dst_ext);
  }
  if (src_device_address->GetSize() != dst_device_address->GetSize() && dst_ext->dtype_id_ != kObjectTypeString) {
    MS_LOG(WARNING) << "Invalid size for host to device copy, host device address:" << src_device_address->ToString()
                    << " device address:" << dst_device_address->ToString();
  }
  MS_LOG(DEBUG) << "Copy host to device, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  return CopyHostToDevice(dst_device_address, src_device_address, src_ext, dst_ext, src_device_address->GetDevicePtr(),
                          src_device_address->GetSize(), ACL_MEMCPY_HOST_TO_DEVICE, stream_id,
                          keep_src ? src_device_sync : nullptr);
}

bool AscendResManager::SyncDeviceToDeviceWithDiffFormatType(const DeviceAddressPtr &dst_device_sync,
                                                            const DeviceAddressPtr &src_device_sync, size_t stream_id,
                                                            const DeviceAddressExtPtr &src_ext,
                                                            const DeviceAddressExtPtr &dst_ext) const {
  const auto &dst_device_address = dynamic_cast<const DeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const DeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  MS_LOG(DEBUG) << "Copy device to device for different format, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  auto host_shape = src_ext->shape_vector_;
  if (host_shape.empty()) {
    MS_LOG(WARNING) << "Host shape of source device address is empty, emplace back shape [1],  device address: "
                    << src_device_address->ToString();
    (void)host_shape.emplace_back(1);
  }
  auto host_tensor = std::make_shared<tensor::Tensor>(src_ext->dtype_id_, host_shape);
  MS_EXCEPTION_IF_NULL(host_tensor);
  MS_EXCEPTION_IF_NULL(host_tensor->device_address());
  const auto &host_device_address = dynamic_cast<const DeviceAddress *>(host_tensor->device_address().get());
  MS_EXCEPTION_IF_NULL(host_device_address);
  std::vector<uint8_t> host_tmp;
  if (host_device_address->GetDevicePtr() == nullptr) {
    host_tmp = std::vector<uint8_t>(src_device_address->GetSize());
    host_device_address->SetDevicePtr(host_tmp.data());
  }
  if (!SyncDeviceToHost(host_tensor->device_address(), src_device_sync, stream_id, src_ext, dst_ext)) {
    MS_LOG(ERROR)
      << "Sync device to device failed at the stage of sync device to intermediate Tensor, src device address:"
      << src_device_address->ToString();
    return false;
  }
  if (!SyncHostToDevice(dst_device_sync, host_tensor->device_address(), stream_id, src_ext, dst_ext)) {
    MS_LOG(ERROR)
      << "Sync device to device failed at the stage of sync intermediate tensor to device, dst device address:"
      << dst_device_address->ToString();
    return false;
  }
  return true;
}

bool AscendResManager::AsyncDeviceToDevice(const DeviceAddressPtr &dst_device_sync,
                                           const DeviceAddressPtr &src_device_sync, size_t stream_id,
                                           const DeviceAddressExtPtr &src_ext,
                                           const DeviceAddressExtPtr &dst_ext) const {
  const auto &dst_device_address = dynamic_cast<const DeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const DeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  if (dst_device_address->GetDevicePtr() == src_device_address->GetDevicePtr()) {
    MS_LOG(DEBUG) << "Same addr, no need memcpy data.";
    return true;
  }
  BindDeviceToCurrentThread(true);
  if (src_ext == nullptr || dst_ext == nullptr) {
    return BaseCopy(dst_device_address->GetDevicePtr(), src_device_address->GetDevicePtr(),
                    src_device_address->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream_id, nullptr);
  }

  if (dst_ext->format_ != src_ext->format_ || dst_ext->dtype_id_ != src_ext->dtype_id_) {
    MS_LOG(DEBUG) << "Can not copy from device to device directly, format or type is different, src:"
                  << src_device_address->ToString() << " metadata:" << src_ext->ToString()
                  << " dst:" << dst_device_address->ToString() << " metadata:" << dst_ext->ToString();
    return SyncDeviceToDeviceWithDiffFormatType(dst_device_sync, src_device_sync, stream_id, src_ext, dst_ext);
  }
  MS_LOG(DEBUG) << "Copy device to device, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  if (dst_ext->dtype_id_ > kMonadTypeBegin && dst_ext->dtype_id_ < kMonadTypeEnd) {
    return true;
  }
  if (dst_device_address->GetSize() < src_device_address->GetSize()) {
    MS_LOG(ERROR) << "Src size is greater than det size, src size is: " << src_device_address->GetSize()
                  << ", dst size is: " << dst_device_address->GetSize();
    return false;
  }
  bool ret = BaseCopy(dst_device_address->GetDevicePtr(), src_device_address->GetDevicePtr(),
                      src_device_address->GetSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream_id, nullptr);
  if (!ret) {
    MS_LOG(ERROR) << "Async device to device failed.";
    return false;
  }
  return true;
}

bool AscendResManager::LoadCollectiveCommLib() {
  // If this is simulation, load dummy collective communication library.
  if (!common::GetEnv(kSimulationLevel).empty()) {
    collective_comm_lib_ = &DummyAscendCollectiveCommLib::GetInstance();
    return true;
  }
  if (DistributedMeta::GetInstance()->enable_cross_cluster()) {
    collective_comm_lib_ = &CcoolCollectiveCommLib::GetInstance();
    MS_EXCEPTION_IF_NULL(collective_comm_lib_);
    MS_LOG(INFO) << "Loading CCOOL collective library successfully.";
    return true;
  }
  // Load Multi ascend collective communication lib using dynamic library.
  collective_comm_lib_ = &MultiAscendCollectiveCommLib::GetInstance();
  MS_EXCEPTION_IF_NULL(collective_comm_lib_);
  MS_LOG(INFO) << "Loading MACCL collective library successfully.";
  return true;
}

void AscendResManager::SetAclDeterministic() {
  std::lock_guard<std::mutex> lock(set_opt_mutex);
  if (UseSimulationApi()) {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool is_deterministic = ms_context->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON" ? true : false;
  MS_LOG(INFO) << "Set acl deterministic value: " << (is_deterministic ? "1" : "0");
  GilReleaseWithCheck gil_release;
  auto ret = CALL_ASCEND_API(aclSetCompileopt, aclCompileOpt::ACL_OP_DETERMINISTIC, is_deterministic ? "1" : "0");
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set deterministic mode failed! mode is " << is_deterministic << " and error flag is "
                      << ret;
  }
}

void AscendResManager::SetDeterministic() const {
  std::lock_guard<std::mutex> lock(set_opt_mutex);
  if (UseSimulationApi()) {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool is_deterministic = ms_context->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON" ? true : false;
  MS_LOG(INFO) << "Set kernel deterministic value: " << (is_deterministic ? "1" : "0");
  // Set acl sys
  auto ret = CALL_ASCEND_API(aclrtCtxSetSysParamOpt, aclSysParamOpt::ACL_OPT_DETERMINISTIC, is_deterministic ? 1 : 0);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl sys set deterministic mode failed! mode is " << is_deterministic << " and error flag is "
                      << ret;
  }
  // Set hccl
  const std::string hccl_lib = "libhccl.so";
  const std::string hccl_set_config_name = "HcclSetConfig";
  auto hccl_set_config = GetAclFunc(hccl_lib, hccl_set_config_name);
  if (hccl_set_config == nullptr) {
    MS_LOG(EXCEPTION) << "Get 'HcclSetConfig' from " << hccl_lib << " failed!";
  }
  auto hccl_set_config_func = reinterpret_cast<HcclSetConfigFunc>(hccl_set_config);
  HcclConfigValue config = {is_deterministic ? 1 : 0};
  auto hccl_ret = hccl_set_config_func(HcclConfig::HCCL_DETERMINISTIC, config);
  if (hccl_ret != HCCL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Hccl set deterministic mode failed! mode is " << is_deterministic << " and error flag is "
                      << ret;
  }
}

void AscendResManager::SetDebugKernel() const {
  auto op_debug_conf = OpDebugConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_debug_conf);
  auto op_debug_option = op_debug_conf->debug_option();
  if (op_debug_option == "oom") {
    auto ret = CALL_ASCEND_API(aclrtCtxSetSysParamOpt, aclSysParamOpt::ACL_OPT_ENABLE_DEBUG_KERNEL, 1);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Acl enable debug kernel failed! Error flag is " << ret;
    }
  }
}

bool AscendResManager::BindDeviceToCurrentThread(bool force_bind) const {
  static thread_local std::once_flag is_set;
  std::call_once(is_set, [this]() {
    auto ret = CALL_ASCEND_API(aclrtSetDevice, static_cast<int32_t>(device_id_));
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Device " << device_id_ << " call aclrtSetDevice failed, ret:" << static_cast<int>(ret);
    }
    SetDeterministic();
    SetDebugKernel();
  });

  if (force_bind) {
    AscendHalManager::GetInstance().SetContextForce(device_id_);
  } else {
    AscendHalManager::GetInstance().SetContext(device_id_);
  }

  return true;
}

bool AscendResManager::CreateStream(size_t *stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().CreateStream(stream_id);
  return true;
}

bool AscendResManager::CreateStreamWithPriority(size_t *stream_id, int32_t priority) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().CreateStreamWithFlags(stream_id, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC,
                                                       IntToUint(priority));
  return true;
}

bool AscendResManager::DestroyStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().DestroyStream(stream_id);
  return true;
}

size_t AscendResManager::QueryStreamSize() const { return AscendStreamMng::GetInstance().QueryStreamSize(); }

std::vector<uint32_t> AscendResManager::GetStreamIds() const { return AscendStreamMng::GetInstance().GetStreamIds(); }

bool AscendResManager::single_op_multi_stream_enable() const {
  return AscendStreamMng::GetInstance().single_op_multi_stream_enable();
}

void AscendResManager::set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {
  return AscendStreamMng::GetInstance().set_single_op_multi_stream_enable(single_op_multi_stream_enable);
}

void *AscendResManager::GetStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return nullptr;
  }
  return AscendStreamMng::GetInstance().GetStream(stream_id);
}

void AscendResManager::SetCurrentStreamId(size_t stream_id) {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return;
  }
  AscendStreamMng::GetInstance().set_current_stream(stream_id);
}

size_t AscendResManager::GetCurrentStreamId() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return SIZE_MAX;
  }
  return AscendStreamMng::GetInstance().current_stream();
}

bool AscendResManager::QueryStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().QueryStream(stream_id);
}

bool AscendResManager::SyncStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().SyncStream(stream_id);
}

bool AscendResManager::SyncAllStreams(bool sync_device) const {
  AscendHalManager::GetInstance().SetContext(device_id_);
  return AscendStreamMng::GetInstance().SyncAllStreams(sync_device);
}

bool AscendResManager::SyncNotDefaultStreams() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().SyncNotDefaultStreams();
}

size_t AscendResManager::DefaultStream() const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return SIZE_MAX;
  }
  return AscendStreamMng::GetInstance().default_stream_id();
}

std::pair<std::vector<size_t>, std::vector<size_t>> AscendResManager::AllocDeviceMemoryForTensorList(
  const std::vector<tensor::TensorPtr> &tensor_list, bool enable_mem_align) {
  MS_LOG(INFO) << "Start AllocDeviceMemoryForTensorList";
  std::vector<size_t> before_padding_sizes = GetUniqueTensorListSize(tensor_list);
  if (enable_mem_align == false) {
    size_t total_size = std::accumulate(before_padding_sizes.begin(), before_padding_sizes.end(), IntToSize(0));
    auto stream_id = DefaultStream();
    auto total_align_size = GetCommonAlignSize(total_size);
    auto device_ptr = AscendMemoryPool::GetInstance().AllocTensorMem(total_align_size, false, false, stream_id);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, "PyNative", total_align_size, device_ptr,
                                                   memory::mem_pool::MemType::kContinuousMemory);
    if (!device_ptr) {
      MS_LOG(EXCEPTION) << "Alloc device memory failed!";
    }
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

    // create device for all tensor in tensor list
    char *ptr = reinterpret_cast<char *>(device_ptr);
    for (size_t i = 0; i < tensor_list.size(); ++i) {
      const auto &tensor = tensor_list[i];
      auto device_address = std::make_shared<device::DeviceAddress>(
        reinterpret_cast<void *>(ptr), before_padding_sizes[i], device::DeviceType::kAscend, stream_id);
      MS_LOG(DEBUG) << "Create DeviceAddress, ptr:" << reinterpret_cast<void *>(ptr)
                    << ", size:" << before_padding_sizes[i] << ", shape:" << tensor->shape()
                    << ", data_type:" << TypeIdToString(tensor->data_type());
      MS_EXCEPTION_IF_NULL(device_address);
      MS_EXCEPTION_IF_NULL(tensor->device_address());
      device::DeviceContextKey host_key = {GetDeviceTypeByName(device_name), device_address->device_id()};
      device::DeviceContext *host_context =
        device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
      MS_EXCEPTION_IF_NULL(host_context);
      MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
      host_context->device_res_manager_->SyncAllStreams();
      SyncCopy(device_address, tensor->device_address(), device_address->stream_id());
      tensor->set_device_address(device_address);
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(MarkTensorAsOutput, "PyNative", device_name, device_ptr,
                                                     tensor->data_type(), tensor->shape(), tensor->storage_info());
      ptr += before_padding_sizes[i];
    }
    std::vector<size_t> after_padding_sizes(before_padding_sizes.size());
    std::copy(before_padding_sizes.begin(), before_padding_sizes.end(), after_padding_sizes.begin());
    after_padding_sizes.back() = total_align_size - total_size + before_padding_sizes.back();
    return std::make_pair(before_padding_sizes, after_padding_sizes);
  }

  std::vector<size_t> after_padding_sizes;
  for (auto &size : before_padding_sizes) {
    auto align_size = GetCommonAlignSize(size);
    after_padding_sizes.emplace_back(align_size);
  }
  auto stream_id = DefaultStream();
  auto device_ptr_list = AllocateContinuousMemory(before_padding_sizes, stream_id);
  for (size_t i = 0; i < after_padding_sizes.size(); ++i) {
    auto acl_ret = CALL_ASCEND_API(aclrtMemset, device_ptr_list[i], after_padding_sizes[i], 0, after_padding_sizes[i]);
    if (acl_ret != ACL_RT_SUCCESS) {
      MS_LOG(EXCEPTION) << "Clear overflow memory failed, aclrtMemset size = " << after_padding_sizes[i]
                        << ", ret = " << acl_ret;
    }
    MS_LOG(DEBUG) << "Clear ptr:" << device_ptr_list[i] << ", size:" << after_padding_sizes[i];
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  // create device for all tensor in tensor list
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    const auto &tensor = tensor_list[i];
    const auto &ptr = device_ptr_list[i];
    auto device_address =
      std::make_shared<device::DeviceAddress>(ptr, before_padding_sizes[i], device::DeviceType::kAscend, stream_id);
    MS_LOG(DEBUG) << "Create DeviceAddress, ptr:" << ptr << ", size:" << before_padding_sizes[i]
                  << ", shape:" << tensor->shape() << ", data_type:" << TypeIdToString(tensor->data_type());
    MS_EXCEPTION_IF_NULL(device_address);
    MS_EXCEPTION_IF_NULL(tensor->device_address());
    device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    host_context->device_res_manager_->SyncAllStreams();
    SyncCopy(device_address, tensor->device_address(), device_address->stream_id());
    tensor->set_device_address(device_address);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, "PyNative", before_padding_sizes[i], ptr,
                                                   memory::mem_pool::MemType::kContinuousMemory);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(MarkTensorAsOutput, "PyNative",
                                                   device::GetDeviceNameByType(device_address->GetDeviceType()), ptr,
                                                   tensor->data_type(), tensor->shape(), tensor->storage_info());
  }
  return std::make_pair(before_padding_sizes, after_padding_sizes);
}

int AscendResManager::ResetParams(const std::vector<tensor::TensorPtr> &params) const {
  constexpr size_t kDefaultStreamId = 0;
  auto stream_id = kDefaultStreamId;
  auto stream_ptr = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(INFO) << "Size of params is " << params.size();

  for (size_t index = 0; index < params.size(); ++index) {
    auto &tensor = params[index];
    if (tensor->device_address() == nullptr || tensor->device_address()->GetMutablePtr() == nullptr) {
      MS_LOG(INFO) << "Parameter " << index << "/" << params.size() << " size=" << tensor->Size()
                   << " tensor device address is nullptr, skip resetting.";
      continue;
    }
    MS_LOG(INFO) << "Parameter " << index << "/" << params.size() << " size=" << tensor->Size()
                 << " ptr: " << tensor->device_address()->GetMutablePtr()
                 << ", type: " << tensor->device_address()->GetDeviceType();
    auto ret = ACL_SUCCESS;
    bool has_pinned_allocator =
      tensor->device_address()->allocator() != nullptr && tensor->device_address()->allocator()->IsPinned();
    if (tensor->device_address()->GetDeviceType() == device::DeviceType::kCPU && !has_pinned_allocator) {
      ret = CALL_ASCEND_API(aclrtMemset, tensor->device_address()->GetMutablePtr(), tensor->Size(), 0, tensor->Size());
    } else {
      ret = CALL_ASCEND_API(aclrtMemsetAsync, tensor->device_address()->GetMutablePtr(), tensor->Size(), 0,
                            tensor->Size(), stream_ptr);
    }
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Call aclrtMemsetAsync failed with return value " << ret << ".";
      return ret;
    }
  }
  (void)SyncStream(stream_id);

  return 0;
}

tensor::TensorPtr AscendResManager::GetSliceByTensorListIndexHandle(const std::vector<tensor::TensorPtr> &tensor_list,
                                                                    const std::vector<size_t> &before_padding_size,
                                                                    const std::vector<size_t> &after_padding_size,
                                                                    size_t start, size_t end) {
  if (start >= tensor_list.size() || end > tensor_list.size()) {
    MS_EXCEPTION(ValueError) << "start:" << start << ", end:" << end << ", but tensor_list size:" << tensor_list.size();
  }
  size_t size = std::accumulate(after_padding_size.begin() + start, after_padding_size.begin() + end - 1,
                                before_padding_size[end - 1]);
  ShapeVector shape = {SizeToLong(size / UnitSizeInBytes(tensor_list[start]->data_type()))};
  auto tensor = tensor::from_spec(tensor_list[start]->data_type(), shape, device::DeviceType::kNone);
  MS_EXCEPTION_IF_NULL(tensor_list[start]->device_address());
  auto ptr = tensor_list[start]->device_address()->GetMutablePtr();

  auto stream_id = DefaultStream();
  auto device_address = std::make_shared<device::DeviceAddress>(ptr, size, device::DeviceType::kAscend, stream_id);
  tensor->set_format(Format::ND);
  tensor->set_device_address(device_address);
  return tensor;
}

TensorPtr AscendResManager::GetSliceByPaddingShapeHandle(const tensor::TensorPtr &first_tensor, size_t start,
                                                         size_t end) {
  auto type_id = first_tensor->data_type();
  auto type_size = UnitSizeInBytes(type_id);
  size_t tensor_size = (end - start) * type_size;
  ShapeVector shape = {static_cast<int64_t>(end - start)};
  auto tensor = tensor::from_spec(type_id, shape, device::DeviceType::kNone);
  MS_EXCEPTION_IF_NULL(first_tensor->device_address());
  auto ptr = first_tensor->device_address()->GetMutablePtr();
  auto offset_size = start * type_size;

  auto stream_id = DefaultStream();
  auto device_address = std::make_shared<device::DeviceAddress>(reinterpret_cast<uint8_t *>(ptr) + offset_size,
                                                                tensor_size, device::DeviceType::kAscend, stream_id);
  MS_LOG(DEBUG) << "Create DeviceAddress, offset size to ptr0:" << offset_size << ", tensor_size:" << tensor_size
                << ", shape:" << shape << ", data_type:" << TypeIdToString(type_id);
  tensor->set_format(Format::ND);
  tensor->set_device_address(device_address);
  return tensor;
}

// ACL_EVENT_TIME_LINE: indicates that the number of created events is not limited, and the created events can be used
//  to compute the elapsed time between events, which may cause lost some performance.
// ACL_EVENT_SYNC: indicates that the number of created events is limited, and the created events can be used for
//  synchronization between multiple streams.
// ACL_EVENT_CAPTURE_STREAM_PROGRESS: indicates that the number of created events is not limited and high performance,
//  and the created events can not be used for timing and synchronization.
DeviceEventPtr AscendResManager::CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) {
  if (!enable_blocking && !enable_record_wait) {
    MS_LOG(INTERNAL_EXCEPTION) << "Bad parameters, enable_blocking is false and enable_record_wait is false.";
  }

  uint32_t flag = 0;
  if (enable_blocking) {
    flag |= ACL_EVENT_SYNC;
  }
  if (enable_record_wait) {
    flag |= ACL_EVENT_CAPTURE_STREAM_PROGRESS;
  }
  return std::make_shared<AscendEvent>(flag);
}

CaptureGraphPtr AscendResManager::CreateCaptureGraph() { return std::make_shared<AscendCaptureGraph>(); }

DeviceEventPtr AscendResManager::CreateEventWithFlag(bool enable_timing, bool blocking, bool use_extensional_api) {
  auto flag = enable_timing ? (ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC) : ACL_EVENT_SYNC;
  auto event = std::make_shared<AscendEvent>(flag, use_extensional_api);
  MS_EXCEPTION_IF_NULL(event);
  std::lock_guard<std::mutex> lock(device_events_mutex_);
  device_events_.push_back(event);
  return event;
}

bool AscendResManager::DestroyEvent(const DeviceEventPtr &event) {
  MS_EXCEPTION_IF_NULL(event);
  if (!event->DestroyEvent()) {
    MS_LOG(ERROR) << "Destroy Event failed.";
    return false;
  }
  std::lock_guard<std::mutex> lock(device_events_mutex_);
  const auto &iter = std::find(device_events_.begin(), device_events_.end(), event);
  if (iter == device_events_.end()) {
    MS_LOG(WARNING) << "Can't find specified device event.";
    return false;
  }
  (void)device_events_.erase(iter);
  return true;
}

bool AscendResManager::DestroyAllEvents() {
  DeviceEventPtrList device_events_inner;
  {
    std::lock_guard<std::mutex> lock(device_events_mutex_);
    device_events_inner = device_events_;
    device_events_.clear();
  }
  (void)std::for_each(device_events_inner.begin(), device_events_inner.end(), [this](const auto &event) {
    MS_EXCEPTION_IF_NULL(event);
    if (!event->DestroyEvent()) {
      MS_LOG(ERROR) << "Destroy Event failed.";
    }
  });
  device_events_.clear();
  return true;
}

void AscendResManager::GetDeviceLimit(int32_t device_id, uint32_t *cube_num, uint32_t *vector_num) {
  MS_EXCEPTION_IF_NULL(cube_num);
  MS_EXCEPTION_IF_NULL(vector_num);
  int32_t dev = device_id;
  if (device_id == -1) {
    dev = device_id_;
  }
  auto ret = CALL_ASCEND_API(aclrtGetDeviceResLimit, dev, aclrtDevResLimitType::ACL_RT_DEV_RES_CUBE_CORE, cube_num);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtGetDeviceResLimit failed! Error flag is " << ret;
  }
  ret = CALL_ASCEND_API(aclrtGetDeviceResLimit, dev, aclrtDevResLimitType::ACL_RT_DEV_RES_VECTOR_CORE, vector_num);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtGetDeviceResLimit failed! Error flag is " << ret;
  }
}

void AscendResManager::SetDeviceLimit(int32_t device_id, int32_t cube_num, int32_t vector_num) {
  enable_res_limit_ = true;
  int32_t dev = device_id;
  if (device_id != -1) {
    dev = device_id_;
  }
  runtime::RuntimeConf::GetInstance()->EnableResLimit();
  if (cube_num > 0) {
    auto ret = CALL_ASCEND_API(aclrtSetDeviceResLimit, dev, aclrtDevResLimitType::ACL_RT_DEV_RES_CUBE_CORE,
                               static_cast<uint32_t>(cube_num));
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Call aclrtSetDeviceResLimit failed! Error flag is " << ret;
    }
  }
  if (vector_num > 0) {
    auto ret = CALL_ASCEND_API(aclrtSetDeviceResLimit, dev, aclrtDevResLimitType::ACL_RT_DEV_RES_VECTOR_CORE,
                               static_cast<uint32_t>(vector_num));
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Call aclrtSetDeviceResLimit failed! Error flag is " << ret;
    }
  }
}

void AscendResManager::GetStreamLimit(size_t stream_id, uint32_t *cube_num, uint32_t *vector_num) {
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(cube_num);
  MS_EXCEPTION_IF_NULL(vector_num);
  auto ret = CALL_ASCEND_API(aclrtGetStreamResLimit, stream, aclrtDevResLimitType::ACL_RT_DEV_RES_CUBE_CORE, cube_num);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtGetStreamResLimit failed! Error flag is " << ret;
  }
  ret = CALL_ASCEND_API(aclrtGetStreamResLimit, stream, aclrtDevResLimitType::ACL_RT_DEV_RES_VECTOR_CORE, vector_num);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtGetStreamResLimit failed! Error flag is " << ret;
  }
}
void AscendResManager::SetStreamLimit(size_t stream_id, int32_t cube_num, int32_t vector_num) {
  enable_res_limit_ = true;
  runtime::RuntimeConf::GetInstance()->EnableResLimit();
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (cube_num > 0) {
    auto ret = CALL_ASCEND_API(aclrtSetStreamResLimit, stream, aclrtDevResLimitType::ACL_RT_DEV_RES_CUBE_CORE,
                               static_cast<uint32_t>(cube_num));
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Call aclrtSetStreamResLimit failed! Error flag is " << ret;
    }
  }
  if (vector_num > 0) {
    auto ret = CALL_ASCEND_API(aclrtSetStreamResLimit, stream, aclrtDevResLimitType::ACL_RT_DEV_RES_VECTOR_CORE,
                               static_cast<uint32_t>(vector_num));
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Call aclrtSetStreamResLimit failed! Error flag is " << ret;
    }
  }
}

void AscendResManager::ResetStreamLimit(size_t stream_id) {
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  auto ret = CALL_ASCEND_API(aclrtResetStreamResLimit, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtResetStreamResLimit failed! Error flag is " << ret;
  }
}

void AscendResManager::UseStreamResInCurrentThread(size_t stream_id) {
  if (!enable_res_limit_) {
    return;
  }
  if (prev_set_stream_id_ == stream_id) {
    return;
  }
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  auto ret = CALL_ASCEND_API(aclrtUseStreamResInCurrentThread, stream);
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call aclrtUseStreamResInCurrentThread failed! Error flag is " << ret;
  }
  prev_set_stream_id_ = stream_id;
}

void AscendResManager::SetEnableStreamLimit() {
  MS_LOG(DEBUG) << "Enter SetEnableStreamLimit";
  enable_res_limit_ = true;
}

bool AscendResManager::GetMemUceInfo(int32_t device_id) {
  aclrtMemUceInfo info[MAX_MEM_UCE_INFO_ARRAY_SIZE];
  size_t retSize = 0;
  auto ret = CALL_ASCEND_API(aclrtGetMemUceInfo, device_id, info, sizeof(info) / sizeof(aclrtMemUceInfo), &retSize);
  if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "Call aclrtGetMemUceInfo failed, ret code: " << ret;
    return false;
  }
  if (retSize == 0) {
    MS_LOG(WARNING) << "aclrtGetMemUceInfo get UCE size is 0.";
  }

  MS_LOG(INFO) << "aclrtGetMemUceInfo get UCE Error, retSize is " << retSize;

  MemUceInfo mem_uce_info;
  mem_uce_info.device_id = device_id;
  mem_uce_info.info.assign(info, info + retSize);
  mem_uce_info.retSize = retSize;

  std::lock_guard<std::mutex> lock(mem_uce_info_mutex_);
  mem_uce_info_ = mem_uce_info;

  return true;
}

std::vector<std::pair<device::DeviceMemPtr, size_t>> AscendResManager::GetMemUceAddr() {
  std::vector<std::pair<device::DeviceMemPtr, size_t>> mem_uce_addr;
  for (size_t i = 0; i < mem_uce_info_.info.size(); ++i) {
    std::pair<device::DeviceMemPtr, size_t> mem(mem_uce_info_.info[i].addr, mem_uce_info_.info[i].len);
    mem_uce_addr.emplace_back(mem);
  }
  MS_LOG(INFO) << "Get mem uce addr, size: " << mem_uce_addr.size();
  return mem_uce_addr;
}

void AscendResManager::UceMemRepair(int32_t device_id) {
  if (device_id != mem_uce_info_.device_id) {
    MS_LOG(EXCEPTION) << "Uce mem repair device id is not correct, device id is " << mem_uce_info_.device_id
                      << ", but got " << device_id << ".";
  }
  aclrtMemUceInfo *info = mem_uce_info_.info.data();
  auto ret = CALL_ASCEND_API(aclrtMemUceRepair, mem_uce_info_.device_id, info, mem_uce_info_.retSize);
  if (ret != ACL_SUCCESS) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtMemUceRepair failed, ret code: " << ret;
  }
  // Clear mem_uce_info.
  mem_uce_info_.device_id = 0;
  mem_uce_info_.info.clear();
  mem_uce_info_.retSize = 0;
}

void AscendResManager::StopDevice(int32_t device_id) {
  // Wait 1 s to avoid stop device and suspension occur at the same time.
  const int64_t kTimeToWait = 1;
  std::this_thread::sleep_for(std::chrono::seconds(kTimeToWait));
  MS_LOG(INFO) << "Device id [" << device_id << "] stop device.";
  uint32_t timeout = 0;
  auto ret = CALL_ASCEND_API(aclrtDeviceTaskAbort, device_id, timeout);
  if (ret != ACL_SUCCESS) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtDeviceTaskAbort failed, ret code: " << ret;
  }
}

void *AscendResManager::GetCopyDataStream() const {
  auto copy_out_data_stream = AscendStreamMng::GetInstance().GetCopyOutStream();
  if (copy_out_data_stream == nullptr) {
    size_t copy_stream_id;
    AscendStreamMng::GetInstance().CreateStream(&copy_stream_id);
    MS_LOG(INFO) << "Create ascend copy data stream, stream id: " << copy_stream_id;
    copy_out_data_stream = AscendStreamMng::GetInstance().GetStream(copy_stream_id);
    AscendStreamMng::GetInstance().SetCopyOutStream(copy_out_data_stream);
  }
  return copy_out_data_stream;
}

bool AscendResManager::RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                                   const DeviceEventPtr &input_event) {
  if (GetMemoryPool() == nullptr) {
    MS_LOG(WARNING) << "memory pool is nullptr.";
    return false;
  }
  return GetMemoryPool()->RecordEvent(task_id_on_stream, user_stream_id, memory_stream_addresses, input_event);
}

bool AscendResManager::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) {
  if (GetMemoryPool() == nullptr) {
    MS_LOG(WARNING) << "memory pool is nullptr.";
    return false;
  }
  return GetMemoryPool()->WaitEvent(task_id_on_stream, user_stream_id, memory_stream_id);
}

bool AscendResManager::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id) {
  if (GetMemoryPool() == nullptr) {
    MS_LOG(WARNING) << "memory pool is nullptr.";
    return false;
  }
  return GetMemoryPool()->WaitEvent(task_id_on_stream, user_stream_id);
}

bool AscendResManager::SyncAllEvents() {
  if (GetMemoryPool() == nullptr) {
    MS_LOG(WARNING) << "memory pool is nullptr.";
    return false;
  }
  return GetMemoryPool()->SyncAllEvents();
}

bool AscendResManager::LaunchCallback(std::function<void(void)> callback_func, size_t stream_id, bool is_block) const {
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().default_stream();
  }
  MS_EXCEPTION_IF_NULL(stream);
  auto block_type =
    is_block ? aclrtCallbackBlockType::ACL_CALLBACK_BLOCK : aclrtCallbackBlockType::ACL_CALLBACK_NO_BLOCK;
  auto callback_func_ptr = new Callback(callback_func);
  aclError ret = CALL_ASCEND_API(aclrtLaunchCallback, AclrtLaunchCallback, callback_func_ptr, block_type, stream);
  MS_LOG(DEBUG) << "Launch callback for stream_id : " << stream_id << ", ret : " << ret << ".";
  if (ret) {
    delete callback_func_ptr;
    MS_LOG(ERROR) << "Launch callback for stream_id : " << stream_id << " failed, ret : " << ret << ".";
    if (SyncStream(stream_id)) {
      callback_func();
      return true;
    }

    ResetStreamAndCtx();
    return false;
  }
  return true;
}

void AscendResManager::InitializeForGe() const {
  if (initialized_ge) {
    return;
  }

  MS_LOG(INFO) << "Start initializing for ge.";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT)) {
    return;
  }

  if (static_cast<bool>(ms_context->get_param<uint32_t>(MS_CTX_GE_REF))) {
    ms_context->increase_param<uint32_t>(MS_CTX_GE_REF);
    return;
  }
  std::map<std::string, std::string> ge_options;
  GetGeGlobalOptions(&ge_options);
  SetPassthroughGeOptions("global", &ge_options);
  {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    if (::ge::GEInitialize(ge_options) != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "Initialize GE failed!";
    }
  }
  initialized_ge = true;
  MS_LOG(INFO) << "End initializing for ge.";
}

void AscendResManager::ResetStreamAndCtx() const {
  AscendStreamMng::GetInstance().DestroyAllStreams();
  AscendHalManager::GetInstance().ResetContext(device_id_);
  AscendStreamMng::GetInstance().CreateDefaultStream();
}

size_t AscendResManager::GetCommunicationStreamID() const {
  return AscendStreamMng::GetInstance().communication_stream_id();
}

size_t AscendResManager::GetCommunicationStreamIDByGroup(const std::string &group) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(EXCEPTION) << "Bind context to current thread failed";
    return 0;
  }
  static std::map<std::string, size_t> group_comm_stream;
  auto res = group_comm_stream.find(group);
  if (res != group_comm_stream.end()) {
    return res->second;
  }
  size_t group_stream_id;
  AscendStreamMng::GetInstance().CreateStream(&group_stream_id);
  group_comm_stream.insert(std::pair(group, group_stream_id));
  MS_LOG(DEBUG) << "Create new stream " << group_stream_id << " for hccl group " << group;
  return group_stream_id;
}

MS_REGISTER_HAL_COPY_FUNC(
  DeviceType::kAscend,
  ([](const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id,
      const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::DeviceContextKey host_key = {DeviceType::kAscend, device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    return host_context->device_res_manager_->SyncCopy(dst_device_sync, src_device_sync, stream_id, src_ext, dst_ext);
  }),
  ([](const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id, bool keep_src,
      const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::DeviceContextKey host_key = {DeviceType::kAscend, device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    return host_context->device_res_manager_->AsyncCopy(dst_device_sync, src_device_sync, stream_id, keep_src, src_ext,
                                                        dst_ext);
  }),
  ([](void *dst, const void *src, uint64_t size, size_t stream_id) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::DeviceContextKey host_key = {DeviceType::kAscend, device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    if (stream_id != kDefaultStreamIndex) {
      if (!AscendStreamMng::GetInstance().SyncStream(kDefaultStreamIndex)) {
        MS_LOG(ERROR) << "Sync stream failed, stream id: " << kDefaultStreamIndex;
        return false;
      }
    }
    return host_context->device_res_manager_->Copy(dst, src, size, device::CopyType::kD2H, stream_id);
  }));

REGISTER_DEVICE_PTR_DELETER_MAKER(device::DeviceType::kAscend, ([](void *ptr, bool from_mem_pool) {
                                    if (ptr != nullptr && from_mem_pool) {
                                      AscendMemoryPool::GetInstance().FreeTensorMem(ptr);
                                    }
                                  }));
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
