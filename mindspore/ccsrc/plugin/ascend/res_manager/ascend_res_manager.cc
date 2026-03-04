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

#include "ir/tensor_new.h"
#include "hccl/hccl.h"
#include "runtime/hardware_abstract/collective/collective_comm_lib_loader.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_memory_manager.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "plugin/ascend/res_manager/mbuf_manager/tensorreport_utils.h"
#include "plugin/ascend/res_manager/device_context_conf/op_debug_conf.h"
#include "plugin/ascend/res_manager/event/ascend_event.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "plugin/ascend/res_manager/capture_graph/ascend_capture_graph.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_pin_mem_pool.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_compiler_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "tools/error_handler/error_config.h"
#include "utils/file_utils.h"
#include "utils/distributed_meta.h"
#include "graph/def_types.h"
#include "acl/acl_rt.h"
#include "plugin/ascend/res_manager/collective/ccool_collective_comm_lib.h"
#include "plugin/ascend/res_manager/collective/multi_ascend_collective_comm_lib.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "plugin/ascend/res_manager/collective/dummy_ascend_collective_comm_lib.h"
#include "plugin/ascend/res_manager/error_manager/collective_comm_monitor.h"
#ifdef ENABLE_INTERNAL_KERNELS
#include "plugin/ascend/res_manager/collective/lowlatency_collective_comm_lib.h"
#endif
#include "plugin/ascend/res_manager/hal_manager/ascend_hal_manager.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "mindspore/core/include/device_address/convert_tensor_utils.h"
#include "include/backend/common/ms_device_shape_transfer.h"
#include "runtime/hardware_abstract/memory_manager/swap_manager.h"
#include "runtime/hardware_abstract/utils.h"
#include "include/common/callback.h"
#include "plugin/ascend/res_manager/hal_manager/ascend_err_manager.h"
#include "runtime/hardware_abstract/device_context/device_context.h"
#include "runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr uint32_t kDefaultHcclExecTimeout = 1800;

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
std::string GetAscendPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(aclrtMalloc), &info) == 0) {
    MS_LOG(ERROR) << "Get dladdr failed.";
    return "";
  }
  auto path_tmp = std::string(info.dli_fname);
  const std::string kCann = "cann";
  const std::string kLatest = "latest";
  auto posCann = path_tmp.rfind(kCann);
  auto posLatest = path_tmp.rfind(kLatest);
  if (posCann != std::string::npos) {
    return path_tmp.substr(0, posCann) + kCann + "/";
  } else if (posLatest != std::string::npos) {
    return path_tmp.substr(0, posLatest) + kLatest + "/";
  } else {
      MS_EXCEPTION(ValueError)
      << "Get ascend path from aclrtMalloc path " << path_tmp
      << " failed, please check whether CANN packages are \n"
         "installed correctly, and environment variables are set by source ${LOCAL_ASCEND}/cann/set_env.sh.";
  }
  return "";
}

std::string GetCurrentDir() {
#ifndef _WIN32
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(GetCurrentDir), &dl_info) == 0) {
    MS_LOG(WARNING) << "Get dladdr error";
    return "";
  }
  std::string curr_so_path = dl_info.dli_fname;
  return dirname(curr_so_path.data());
#else
  HMODULE hModule = nullptr;
  if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                        (LPCSTR)GetCurrentDir, &hModule) == 0) {
    MS_LOG(WARNING) << "Get GetModuleHandleEx failed.";
    return "";
  }
  char szPath[MAX_PATH];
  if (GetModuleFileName(hModule, szPath, sizeof(szPath)) == 0) {
    MS_LOG(WARNING) << "Get GetModuleFileName failed.";
    return "";
  }
  std::string cur_so_path = std::string(szPath);
  auto pos = cur_so_path.find_last_of("\\");
  if (cur_so_path.empty() || pos == std::string::npos) {
    MS_LOG(ERROR) << "Current so path empty or the path [" << cur_so_path << "] is invalid.";
    return "";
  }
  return cur_so_path.substr(0, pos);
#endif
}

void *GetAclFunc(const std::string &lib_path, const std::string &func_name) {
  static auto ascend_path = GetAscendPath();
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

Format GetFormat(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto format = Format::DEFAULT_FORMAT;
  if (tensor->device_address() != nullptr) {
    const auto temp_device_address = tensor->device_address();
    auto const device_address = std::dynamic_pointer_cast<const DeviceAddress>(temp_device_address);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetDeviceType() != device::DeviceType::kCPU) {
      auto const src_device_address = device_address;
      format = FromStrToEnum(src_device_address->format());
    } else {
      auto cpu_tensor = tensor->cpu();
      tensor->set_device_address(cpu_tensor->device_address());
    }
  }
  return format;
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

  if (!(IS_VLOG_ON(VL_RUNTIME_FRAMEWORK_MEMORY_ALLOCATE_CHECK))) {
    mem_manager_ = std::make_shared<AscendMemoryManager>();
  } else {
    mem_manager_ = std::make_shared<EnhancedAscendMemoryManager>();
  }
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->Initialize();
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
  if (tools::TftConfig::GetInstance() != nullptr && (tools::TftConfig::GetInstance()->IsEnableWatchdog() ||
                                                     tools::TftConfig::GetInstance()->IsEnableSaveHcclOpStatus())) {
    device::ascend::HcclWatchDogManager::GetInstance().DestroyHandler();
  }

  // DestroyHccl must be called before FreeDeviceMemory, watch_hccl_dog and hccl_adapter are in this function
  (void)DestroyHccl();

  BindDeviceToCurrentThread(true);
  AscendStreamMng::GetInstance().DestroyAllRtEvents();
  if (!AscendStreamMng::GetInstance().DestroyAllStreams()) {
    MS_LOG(EXCEPTION) << "Fail to destroy all streams when reset device.";
  }
  // Release memory.
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
    mem_manager_ = nullptr;
  }

  (void)ErrorManagerAdapter::Finalize();

  // All unmap/free operations will fail after calling aclrtResetDevice in ResetDevice,
  // so it must be called before that.
  AscendVmmAdapter::GetInstance().ClearAllMemory();
  AscendHalManager::GetInstance().ResetDevice(device_id_);

  initialized_ = false;
}

bool AscendResManager::IsEnableVmm() const { return AscendVmmAdapter::IsEnabled(); }

bool AscendResManager::AllocateMemory(DeviceAddress *const &address, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(mem_manager_);

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
    device_ptr = mem_manager_->MallocMemFromMemPool(address->GetSize(), address->from_persistent_mem(),
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

  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->MallocMemFromMemPool(size, false, false, stream_id);
}

void *AscendResManager::AllocateStaticMemory(size_t size, uint32_t stream_id) const {
  AscendHalManager::GetInstance().SetContext(device_id_);

  return mem_manager_->MallocMemFromMemPool(size, true, false, stream_id);
}

size_t AscendResManager::GetMaxUsedMemorySize() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetMaxUsedMemorySize();
}

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

void AscendResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->FreeMemFromMemPool(ptr);
}

void AscendResManager::FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                                       const std::vector<size_t> &keep_addr_sizes) const {
  AscendMemoryPool::GetInstance().FreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

void AscendResManager::DefragMemory() { AscendMemoryPool::GetInstance().DefragMemory(); }

// Relevant function to manage memory statistics
size_t AscendResManager::GetTotalMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalMemStatistics();
}

size_t AscendResManager::GetTotalUsedMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalUsedMemStatistics();
}

size_t AscendResManager::GetTotalIdleMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalIdleMemStatistics();
}

size_t AscendResManager::GetTotalEagerFreeMemStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetTotalEagerFreeMemStatistics();
}

size_t AscendResManager::GetUsedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetUsedMemPeakStatistics();
}

size_t AscendResManager::GetReservedMemPeakStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetReservedMemPeakStatistics();
}

std::unordered_map<std::string, std::size_t> AscendResManager::GetBlockCountsStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetBlockCountsStatistics();
}

std::unordered_map<std::string, std::size_t> AscendResManager::GetBlockUnitSizeStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetBlockUnitSizeStatistics();
}

DeviceMemInfo AscendResManager::GetCommonMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetCommonMemBlocksInfoStatistics();
}

DeviceMemInfo AscendResManager::GetPersistentMemBlocksInfoStatistics() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetPersistentMemBlocksInfoStatistics();
}

void AscendResManager::ResetMaxMemoryReserved() {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto memory_pool = mem_manager_->GetMemoryPool();
  MS_EXCEPTION_IF_NULL(memory_pool);
  memory_pool->ResetMaxMemReserved();
}

void AscendResManager::ResetMaxMemoryAllocated() {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto memory_pool = mem_manager_->GetMemoryPool();
  MS_EXCEPTION_IF_NULL(memory_pool);
  memory_pool->ResetMaxMemAllocated();
}

size_t AscendResManager::EmptyCache() {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto memory_pool = mem_manager_->GetMemoryPool();
  MS_EXCEPTION_IF_NULL(memory_pool);
  return memory_pool->EmptyCache();
}

void AscendResManager::SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
  (void)mem_manager_->SwapIn(host_ptr, device_ptr, mem_size, stream);
}

void AscendResManager::SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
  (void)mem_manager_->SwapOut(device_ptr, host_ptr, mem_size, stream);
}

std::vector<void *> AscendResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                               uint32_t stream_id) const {
  AscendHalManager::GetInstance().SetContext(device_id_);

  MS_EXCEPTION_IF_NULL(mem_manager_);
  std::vector<size_t> aligned_size_list;
  for (auto size : size_list) {
    auto align_size = device::MemoryManager::GetCommonAlignSize(size);
    aligned_size_list.emplace_back(align_size);
  }
  return mem_manager_->MallocContinuousMemFromMemPool(aligned_size_list, stream_id);
}

DeviceAddressPtr AscendResManager::CreateDeviceAddress() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_address = std::make_shared<DeviceAddress>(nullptr, 0, kAscendDevice);
  device_address->SetDeviceType(device::GetDeviceTypeByName(ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET)));
  return device_address;
}

DeviceAddressPtr AscendResManager::CreateDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector,
                                                       const Format &format, TypeId type_id,
                                                       const std::string &device_name, uint32_t stream_id) const {
  auto device_address =
    std::make_shared<DeviceAddress>(ptr, size, shape_vector, format, type_id, kAscendDevice, stream_id);
  return device_address;
}

bool AscendResManager::SyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                                size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(dst_device_sync);
  MS_EXCEPTION_IF_NULL(src_device_sync);
  if (dst_device_sync->GetDeviceType() == DeviceType::kAscend && src_device_sync->GetDeviceType() == DeviceType::kCPU) {
    return SyncHostToDevice(dst_device_sync, src_device_sync, stream_id);
  }
  if (dst_device_sync->GetDeviceType() == DeviceType::kCPU && src_device_sync->GetDeviceType() == DeviceType::kAscend) {
    return SyncDeviceToHost(dst_device_sync, src_device_sync, stream_id);
  }
  return SyncDeviceToDevice(dst_device_sync, src_device_sync, stream_id);
}

bool AscendResManager::AsyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                                 size_t stream_id, bool keep_src) const {
  MS_EXCEPTION_IF_NULL(dst_device_sync);
  MS_EXCEPTION_IF_NULL(src_device_sync);
  if (dst_device_sync->GetDeviceType() == DeviceType::kAscend && src_device_sync->GetDeviceType() == DeviceType::kCPU) {
    return AsyncHostToDevice(dst_device_sync, src_device_sync, stream_id, keep_src);
  }
  if (dst_device_sync->GetDeviceType() == DeviceType::kCPU && src_device_sync->GetDeviceType() == DeviceType::kAscend) {
    return AsyncDeviceToHost(dst_device_sync, src_device_sync, stream_id);
  }
  return AsyncDeviceToDevice(dst_device_sync, src_device_sync, stream_id);
}

namespace {
bool SyncStreamForCopy(const AscendResManager *const res_manager, size_t stream_id) {
  MS_EXCEPTION_IF_NULL(res_manager);
  bool ret = res_manager->SyncStream(stream_id);
  if (!ret) {
    MS_LOG(WARNING) << "Uce flag: " << UCEException::GetInstance().get_uce_flag()
                    << ", force stop flag: " << UCEException::GetInstance().get_force_stop_flag();
    if (UCEException::GetInstance().get_uce_flag()) {
      MS_LOG(EXCEPTION) << "UCEError occurs when execute.";
    } else if (UCEException::GetInstance().get_force_stop_flag()) {
      MS_LOG(EXCEPTION) << "ForceStopError occurs when execute.";
    }
    MS_LOG(EXCEPTION) << "Sync stream error!";
  }
  MS_LOG(DEBUG) << "SyncStream Finish!";
  return true;
}
}  // namespace

bool AscendResManager::SyncDeviceToHost(const DeviceAddressPtr &dst_device_sync,
                                        const DeviceAddressPtr &src_device_sync, size_t stream_id) const {
  if (!AsyncDeviceToHost(dst_device_sync, src_device_sync, stream_id)) {
    return false;
  }
  return SyncStreamForCopy(this, stream_id);
}

bool AscendResManager::SyncHostToDevice(const DeviceAddressPtr &dst_device_sync,
                                        const DeviceAddressPtr &src_device_sync, size_t stream_id) const {
  if (!AsyncHostToDevice(dst_device_sync, src_device_sync, stream_id, false)) {
    return false;
  }
  return SyncStreamForCopy(this, stream_id);
}

bool AscendResManager::SyncDeviceToDevice(const DeviceAddressPtr &dst_device_sync,
                                          const DeviceAddressPtr &src_device_sync, size_t stream_id) const {
  if (!AsyncDeviceToDevice(dst_device_sync, src_device_sync, stream_id)) {
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

const std::set<std::string> op_need_trans_format = {
  kOpFormat_NHWC,    kOpFormat_HWCN,        kOpFormat_NC1HWC0,       kOpFormat_FRAC_Z,   kOpFormat_C1HWNCoC0,
  kOpFormat_FRAC_NZ, kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_NDC1HWC0, kOpFormat_FRACTAL_Z_3D};

ShapeVector GetDeviceShape(ShapeVector *host_shape, const DeviceAddress *src_device_address) {
  MS_EXCEPTION_IF_NULL(host_shape);
  ShapeVector device_shape;
  auto node_index = src_device_address->GetNodeIndex();
  if (src_device_address->format() == kOpFormat_FRAC_NZ || src_device_address->format() == kOpFormat_NCDHW) {
    device_shape = trans::TransShapeToDevice(*host_shape, src_device_address->format(), node_index.first,
                                             node_index.second, src_device_address->type_id());
  } else {
    if (!src_device_address->GetShapeVector().empty()) {
      host_shape->clear();
      *host_shape = src_device_address->GetShapeVector();
    }
    *host_shape = trans::PaddingShape(*host_shape, src_device_address->format());
    device_shape = trans::TransShapeToDevice(*host_shape, src_device_address->format(), node_index.first,
                                             node_index.second, src_device_address->type_id());
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
                  << " stream id:" << stream_id;
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
                                                     const DeviceAddress *src_device_address, size_t stream_id) const {
  MS_LOG(DEBUG) << "Copy device to host for different format, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  const auto &src_format = src_device_address->format();
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
  ShapeVector host_shape = dst_device_address->GetShapeVector();
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  auto device_shape = GetDeviceShape(&host_shape, src_device_address);
  MS_LOG(DEBUG) << "Host shape:" << host_shape << " device shape:" << device_shape
                << " format:" << src_device_address->format();
  auto node_index = src_device_address->GetNodeIndex();
  if (src_device_address->type_id() == dst_device_address->type_id()) {
    const trans::FormatArgs format_args{
      host_tmp.data(), src_device_address->GetSize(), kOpFormat_NCHW, src_device_address->format(), host_shape,
      device_shape,    src_device_address->type_id()};
    if (!trans::TransFormatFromDeviceToHost(format_args, dst_device_address->GetDevicePtr(), node_index.first,
                                            node_index.second)) {
      MS_LOG(ERROR) << "Trans format failed for dst device tensor:" << dst_device_address->ToString();
      return false;
    }
    return true;
  }
  const trans::FormatArgs format_args{
    host_tmp.data(), src_device_address->GetSize(), kOpFormat_NCHW, src_device_address->format(), host_shape,
    device_shape,    src_device_address->type_id()};
  auto trans_format_host = std::vector<uint8_t>(src_device_address->GetSize());
  if (!trans::TransFormatFromDeviceToHost(format_args, trans_format_host.data(), node_index.first, node_index.second)) {
    MS_LOG(ERROR) << "Trans format failed for dst device tensor:" << dst_device_address->ToString();
    return false;
  }
  auto shape_size = abstract::ShapeSize(host_shape);
  const trans::TypeIdArgs type_args{trans_format_host.data(), shape_size, src_device_address->type_id(),
                                    dst_device_address->type_id(), dst_device_address->GetSize()};
  if (!trans::TransDataType(type_args, dst_device_address->GetDevicePtr())) {
    MS_LOG(ERROR) << "Trans data type failed for dst device tensor:" << dst_device_address->ToString();
    return false;
  }
  return true;
}

bool AscendResManager::CopyDeviceToHostForDiffType(const DeviceAddress *dst_device_address,
                                                   const DeviceAddress *src_device_address, size_t stream_id) const {
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
  if (src_device_address->type_id() == kNumberTypeFloat32 && dst_device_address->type_id() == kNumberTypeFloat64) {
    if (src_device_address->GetSize() / sizeof(float) != dst_device_address->GetSize() / sizeof(double)) {
      MS_LOG(ERROR) << "Invalid src_size for device address" << src_device_address->ToString()
                    << ", dst_size for device address" << dst_device_address->ToString();
      return false;
    }
    FloatToDouble(dst_device_address->GetDevicePtr(), host_tmp.data(), src_device_address->GetSize() / sizeof(float));
    return true;
  }
  auto host_shape = dst_device_address->GetShapeVector();
  auto shape_size = abstract::ShapeSize(host_shape);
  const trans::TypeIdArgs type_args{host_tmp.data(), shape_size, src_device_address->type_id(),
                                    dst_device_address->type_id(), src_device_address->GetSize()};

  if (!trans::TransDataType(type_args, dst_device_address->GetDevicePtr())) {
    MS_LOG(ERROR) << "Trans data type failed for dst device address:" << dst_device_address->ToString();
    return false;
  }
  return true;
}

bool AscendResManager::AsyncDeviceToHost(const DeviceAddressPtr &dst_device_sync,
                                         const DeviceAddressPtr &src_device_sync, size_t stream_id) const {
  const auto &dst_device_address = dynamic_cast<const DeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const DeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  if (src_device_address->GetTensorStorageInfo() != nullptr || dst_device_address->GetTensorStorageInfo() != nullptr) {
    MS_LOG(WARNING) << "Invalid sync device to host for tensor storage info in device address:"
                    << src_device_address->ToString() << " and:" << dst_device_address->ToString();
  }
  BindDeviceToCurrentThread(false);

  // Check format.
  static const std::set<std::string> basic_format = {kOpFormat_NCHW, kOpFormat_DEFAULT, kOpFormat_NCDHW, kOpFormat_ND};
  if (basic_format.find(src_device_address->format()) == basic_format.end()) {
    return CopyDeviceToHostForDiffFormat(dst_device_address, src_device_address, stream_id);
  }

  // Check type.
  if (src_device_address->type_id() != dst_device_address->type_id()) {
    return CopyDeviceToHostForDiffType(dst_device_address, src_device_address, stream_id);
  }

  MS_LOG(DEBUG) << "Copy device to host, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  return BaseCopy(dst_device_address->GetDevicePtr(), src_device_address->GetDevicePtr(), dst_device_address->GetSize(),
                  ACL_MEMCPY_DEVICE_TO_HOST, stream_id);
}

bool AscendResManager::CopyHostToDevice(const DeviceAddress *dst_device_address,
                                        const DeviceAddress *src_device_address, const void *src, uint64_t size,
                                        aclrtMemcpyKind kind, size_t stream_id,
                                        const DeviceAddressPtr src_device_sync) const {
  if (dst_device_address->type_id() != kObjectTypeString) {
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
                                                     const DeviceAddress *src_device_address, size_t stream_id) const {
  MS_LOG(DEBUG) << "Copy host to device for different format, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  const auto &dst_format = dst_device_address->format();
  if (op_need_trans_format.find(dst_format) == op_need_trans_format.end()) {
    MS_LOG(ERROR) << "Can not find format transfer function for format:" << dst_format
                  << " dst device address:" << dst_device_address->ToString();
    return false;
  }

  ShapeVector host_shape = src_device_address->GetShapeVector();
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  auto node_index = dst_device_address->GetNodeIndex();
  std::vector<int64_t> device_shape;
  if (dst_format != kOpFormat_FRAC_NZ) {
    host_shape = trans::PaddingShape(host_shape, dst_format);
    MS_LOG(DEBUG) << "Padding shape from:" << src_device_address->GetShapeVector() << " to:" << host_shape
                  << " for device address:" << src_device_address->ToString();
  }
  device_shape = trans::TransShapeToDevice(host_shape, dst_format, node_index.first, node_index.second,
                                           dst_device_address->type_id());
  MS_LOG(DEBUG) << "Host shape:" << host_shape << " device shape:" << device_shape
                << " for device address:" << dst_device_address->ToString();
  // Trans type.
  std::vector<uint8_t> tmp_host_for_type_trans;
  void *tmp_host_ptr = src_device_address->GetDevicePtr();
  if (src_device_address->type_id() != dst_device_address->type_id()) {
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{src_device_address->GetMutablePtr(), shape_size, src_device_address->type_id(),
                                      dst_device_address->type_id(), src_device_address->GetSize()};
    tmp_host_for_type_trans = std::vector<uint8_t>(dst_device_address->GetSize());
    tmp_host_ptr = tmp_host_for_type_trans.data();
    auto ret = trans::TransDataType(type_args, tmp_host_ptr);
    if (!ret) {
      MS_LOG(ERROR) << "Trans data type failed for dst device address:" << dst_device_address->ToString();
      return false;
    }
  }

  // Trans format.
  const trans::FormatArgs format_args{
    tmp_host_ptr, dst_device_address->GetSize(), kOpFormat_NCHW, dst_format, host_shape,
    device_shape, dst_device_address->type_id()};
  auto host_tmp = std::vector<uint8_t>(dst_device_address->GetSize());
  if (!trans::TransFormat(format_args, host_tmp.data(), node_index.first, node_index.second)) {
    MS_LOG(ERROR) << "Trans format failed.";
    return false;
  }

  bool ret = CopyHostToDevice(dst_device_address, src_device_address, host_tmp.data(), dst_device_address->GetSize(),
                              ACL_MEMCPY_HOST_TO_DEVICE, stream_id);
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
                                                   const DeviceAddress *src_device_address, size_t stream_id) const {
  MS_LOG(DEBUG) << "Copy host to device for different type, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  std::vector<uint8_t> host_tmp = std::vector<uint8_t>(dst_device_address->GetSize());
  if (dst_device_address->type_id() == kNumberTypeFloat32 && src_device_address->type_id() == kNumberTypeFloat64) {
    if (src_device_address->GetSize() / sizeof(double) != dst_device_address->GetSize() / sizeof(float)) {
      MS_INTERNAL_EXCEPTION(ArgumentError) << "Invalid src_size for device address" << src_device_address->ToString()
                                           << ", dst_size for device address" << dst_device_address->ToString();
    }
    DoubleToFloat(host_tmp.data(), src_device_address->GetDevicePtr(), dst_device_address->GetSize() / sizeof(float));
  } else {
    ShapeVector host_shape = src_device_address->GetShapeVector();
    if (host_shape.empty()) {
      (void)host_shape.emplace_back(1);
    }
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{src_device_address->GetDevicePtr(), shape_size, src_device_address->type_id(),
                                      dst_device_address->type_id(), src_device_address->GetSize()};
    if (!trans::TransDataType(type_args, host_tmp.data())) {
      MS_LOG(ERROR) << "Trans data type failed for device address:" << dst_device_address->ToString();
      return false;
    }
  }
  // Sync device to host.
  if (!CopyHostToDevice(dst_device_address, src_device_address, host_tmp.data(), dst_device_address->GetSize(),
                        ACL_MEMCPY_HOST_TO_DEVICE, stream_id)) {
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
                                         const DeviceAddressPtr &src_device_sync, size_t stream_id,
                                         bool keep_src) const {
  const auto &dst_device_address = dynamic_cast<const DeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const DeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  if (src_device_address->GetTensorStorageInfo() != nullptr || dst_device_address->GetTensorStorageInfo() != nullptr) {
    MS_LOG(EXCEPTION) << "Invalid sync host to device for tensor storage info in device address:"
                      << src_device_address->ToString() << " and:" << dst_device_address->ToString();
  }
  BindDeviceToCurrentThread(false);
  // Check hete info.

  // Check format.
  static const std::set<std::string> basic_format = {kOpFormat_NCHW, kOpFormat_DEFAULT, kOpFormat_NCDHW, kOpFormat_ND};
  if (basic_format.find(dst_device_address->format()) == basic_format.end() &&
      src_device_address->format() != dst_device_address->format()) {
    return CopyHostToDeviceForDiffFormat(dst_device_address, src_device_address, stream_id);
  }

  // Check type.
  if (src_device_address->type_id() != dst_device_address->type_id()) {
    return CopyHostToDeviceForDiffType(dst_device_address, src_device_address, stream_id);
  }
  if (src_device_address->GetSize() != dst_device_address->GetSize() &&
      dst_device_address->type_id() != kObjectTypeString) {
    MS_LOG(WARNING) << "Invalid size for host to device copy, host device address:" << src_device_address->ToString()
                    << " device address:" << dst_device_address->ToString();
  }
  MS_LOG(DEBUG) << "Copy host to device, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  return CopyHostToDevice(dst_device_address, src_device_address, src_device_address->GetDevicePtr(),
                          src_device_address->GetSize(), ACL_MEMCPY_HOST_TO_DEVICE, stream_id,
                          keep_src ? src_device_sync : nullptr);
}

bool AscendResManager::SyncDeviceToDeviceWithDiffFormatType(const DeviceAddressPtr &dst_device_sync,
                                                            const DeviceAddressPtr &src_device_sync,
                                                            size_t stream_id) const {
  const auto &dst_device_address = dynamic_cast<const DeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const DeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  MS_LOG(DEBUG) << "Copy device to device for different format, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  auto host_shape = src_device_address->GetShapeVector();
  if (host_shape.empty()) {
    MS_LOG(WARNING) << "Host shape of source device address is empty, emplace back shape [1],  device address size: "
                    << src_device_address->GetSize()
                    << ", device address type: " << TypeIdLabel(src_device_address->type_id());
    (void)host_shape.emplace_back(1);
  }
  auto host_tensor = std::make_shared<tensor::Tensor>(src_device_address->type_id(), host_shape);
  MS_EXCEPTION_IF_NULL(host_tensor);
  MS_EXCEPTION_IF_NULL(host_tensor->device_address());
  const auto &host_device_address = dynamic_cast<const DeviceAddress *>(host_tensor->device_address().get());
  MS_EXCEPTION_IF_NULL(host_device_address);
  std::vector<uint8_t> host_tmp;
  if (host_device_address->GetDevicePtr() == nullptr) {
    host_tmp = std::vector<uint8_t>(src_device_address->GetSize());
    host_device_address->SetDevicePtr(host_tmp.data());
  }
  if (!SyncDeviceToHost(host_tensor->device_address(), src_device_sync, stream_id)) {
    MS_LOG(ERROR)
      << "Sync device to device failed at the stage of sync device to intermediate Tensor, src device address:"
      << src_device_address->ToString();
    return false;
  }
  if (!SyncHostToDevice(dst_device_sync, host_tensor->device_address(), stream_id)) {
    MS_LOG(ERROR)
      << "Sync device to device failed at the stage of sync intermediate tensor to device, dst device address:"
      << dst_device_address->ToString();
    return false;
  }
  return true;
}

bool AscendResManager::AsyncDeviceToDevice(const DeviceAddressPtr &dst_device_sync,
                                           const DeviceAddressPtr &src_device_sync, size_t stream_id) const {
  const auto &dst_device_address = dynamic_cast<const DeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const DeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  BindDeviceToCurrentThread(true);
  if (dst_device_address->format() != src_device_address->format() ||
      dst_device_address->type_id() != src_device_address->type_id()) {
    MS_LOG(INFO) << "Can not copy from device to device directly, format or type is different, src(format:"
                 << src_device_address->format() << ", type_id:" << TypeIdLabel(src_device_address->type_id())
                 << "), dst(format:" << dst_device_address->format()
                 << ", type_id:" << TypeIdLabel(dst_device_address->type_id())
                 << ", use the intermediate Tensor copy instead.";
    return SyncDeviceToDeviceWithDiffFormatType(dst_device_sync, src_device_sync, stream_id);
  }
  MS_LOG(DEBUG) << "Copy device to device, src device address:" << src_device_address->ToString()
                << " dst device address:" << dst_device_address->ToString() << " stream id:" << stream_id;
  if (dst_device_address->GetDevicePtr() == src_device_address->GetDevicePtr()) {
    MS_LOG(DEBUG) << "Same addr, no need memcpy data.";
    return true;
  }
  if (dst_device_address->type_id() > kMonadTypeBegin && dst_device_address->type_id() < kMonadTypeEnd) {
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
  MS_EXCEPTION_IF_NULL(mem_manager_);
  std::vector<size_t> before_padding_sizes = GetUniqueTensorListSize(tensor_list);
  if (enable_mem_align == false) {
    size_t total_size = std::accumulate(before_padding_sizes.begin(), before_padding_sizes.end(), IntToSize(0));
    auto stream_id = DefaultStream();
    auto total_align_size = device::MemoryManager::GetCommonAlignSize(total_size);
    auto device_ptr = mem_manager_->MallocMemFromMemPool(total_align_size, false, false, stream_id);
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
      auto format = GetFormat(tensor);
      auto device_address = CreateDeviceAddress(reinterpret_cast<void *>(ptr), before_padding_sizes[i], tensor->shape(),
                                                format, tensor->data_type(), device_name, stream_id);
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
    auto align_size = device::MemoryManager::GetCommonAlignSize(size);
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
    auto format = GetFormat(tensor);
    auto device_address = CreateDeviceAddress(ptr, before_padding_sizes[i], tensor->shape(), format,
                                              tensor->data_type(), device_name, stream_id);
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
  ShapeVector shape = {int64_t(size / UnitSizeInBytes(tensor_list[start]->data_type()))};
  auto tensor = tensor::from_spec(tensor_list[start]->data_type(), shape, device::DeviceType::kNone);
  MS_EXCEPTION_IF_NULL(tensor_list[start]->device_address());
  auto ptr = tensor_list[start]->device_address()->GetMutablePtr();

  auto stream_id = DefaultStream();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  auto device_address = CreateDeviceAddress(ptr, size, shape, Format::ND, tensor->data_type(), device_name, stream_id);
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
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  auto device_address = CreateDeviceAddress(reinterpret_cast<uint8_t *>(ptr) + offset_size, tensor_size, shape,
                                            Format::ND, type_id, device_name, stream_id);
  MS_LOG(DEBUG) << "Create DeviceAddress, offset size to ptr0:" << offset_size << ", tensor_size:" << tensor_size
                << ", shape:" << shape << ", data_type:" << TypeIdToString(type_id);
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

std::vector<uint64_t> AscendResManager::GetOptimizerTimestamps() {
  OptimizerEventInfo::GetInstance().GetOptimizerTimestamp(false);
  auto hbm_error_time = UCEException::GetInstance().get_uce_occur_time();
  auto opt_start_timestamp = OptimizerEventInfo::GetInstance().get_optimizer_start_timestamp();
  auto opt_end_timestamp = OptimizerEventInfo::GetInstance().get_optimizer_end_timestamp();
  return std::vector<uint64_t>{hbm_error_time, opt_start_timestamp, opt_end_timestamp};
}

void AscendResManager::StopDevice(int32_t device_id) {
  UCEException::GetInstance().set_force_stop_flag(true);
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
  return mem_manager_->RecordEvent(task_id_on_stream, user_stream_id, memory_stream_addresses, input_event);
}

bool AscendResManager::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) {
  return mem_manager_->WaitEvent(task_id_on_stream, user_stream_id, memory_stream_id);
}

bool AscendResManager::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id) {
  return mem_manager_->WaitEvent(task_id_on_stream, user_stream_id);
}

bool AscendResManager::SyncAllEvents() { return mem_manager_->SyncAllEvents(); }

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
  ([](const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::DeviceContextKey host_key = {DeviceType::kAscend, device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    return host_context->device_res_manager_->SyncCopy(dst_device_sync, src_device_sync, stream_id);
  }),
  ([](const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id,
      bool keep_src) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::DeviceContextKey host_key = {DeviceType::kAscend, device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    return host_context->device_res_manager_->AsyncCopy(dst_device_sync, src_device_sync, stream_id, keep_src);
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
