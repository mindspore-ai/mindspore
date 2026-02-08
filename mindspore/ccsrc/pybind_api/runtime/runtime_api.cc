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
#include "pybind_api/runtime/runtime_api.h"
#include <string>
#include <memory>
#include "pybind11/pybind11.h"
#include "include/runtime/hardware_abstract/kernel_base/oplib/oplib.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/hardware_abstract/collective/collective_communication_lib.h"
#include "include/runtime/memory/mem_pool/mem_dynamic_allocator.h"
#include "include/backend/common/exec_order/online_execution_order_check.h"
#include "include/cluster/topology/ps_context.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "include/cluster/rpc/tcp_store.h"

using OpLib = mindspore::kernel::OpLib;
using PSContext = mindspore::ps::PSContext;
using CollectiveManager = mindspore::distributed::collective::CollectiveManager;
using TCPStoreClient = mindspore::distributed::cluster::TCPStoreClient;
using GroupOptions = mindspore::device::GroupOptions;
using DeviceContextManager = mindspore::device::DeviceContextManager;
using DeviceContext = mindspore::device::DeviceContext;
namespace py = pybind11;
namespace mindspore {

void RegRuntimeModule(py::module *m) {
  mindspore::hal::RegStream(m);
  mindspore::hal::RegEvent(m);
  mindspore::hal::RegCommHandle(m);
  mindspore::hal::RegMemory(m);
  mindspore::hal::RegUtils(m);
  mindspore::hal::RegResLimit(m);
  mindspore::runtime::RegRuntimeConf(m);
  RegDeviceManagerConf(m);
  (void)py::class_<OpLib, std::shared_ptr<OpLib>>(*m, "Oplib")
    .def(py::init())
    .def_static("reg_op", &OpLib::RegOp, "Register op info.");
  (void)py::class_<GroupOptions>(*m, "GroupOptions")
    .def(py::init<>())
    .def_readwrite("use_async", &GroupOptions::async)
    .def_readwrite("hccl_config", &GroupOptions::hccl_config);

  (void)py::class_<CollectiveManager, std::shared_ptr<CollectiveManager>>(*m, "CollectiveManager")
    .def_static("get_instance", &CollectiveManager::instance, "Get collective manager instance.")
    .def("initialized", &CollectiveManager::initialized, "Returns whether distributed module is initialized.")
    .def("create_group", &CollectiveManager::CreateCommunicationGroup, "Create collective group.",
         pybind11::arg("group_name"), pybind11::arg("rank_list"), pybind11::arg("options") = GroupOptions())
    .def("destroy_group", &CollectiveManager::DestroyCommunicationGroup, "Destroy collective group.")
    .def("remove_group_info", &CollectiveManager::RemoveGroupInfoForARF, "Remove group info for arf.")
    .def("get_group_map", &CollectiveManager::get_group_map, "Get the group map")
    .def("get_local_rank_id", &CollectiveManager::GetLocalRankId, "Get the node rank id.")
    .def("get_local_group_size", &CollectiveManager::GetLocalGroupSize, "Get the node rank id.")
    .def("get_world_rank_from_group_rank", &CollectiveManager::GetWorldRankFromGroupRank,
         "Get world rank by group rank.")
    .def("get_group_rank_from_world_rank", &CollectiveManager::GetGroupRankFromWorldRank,
         "Get group rank by world rank.")
    .def("get_rank_id", &CollectiveManager::GetRankId, "Get the node rank id.")
    .def("get_group_size", &CollectiveManager::GetGroupSize, "Get the nodes number in the collective communication.")
    .def("get_group_ranks", &CollectiveManager::GetGroupRanks, "Get group ranks for the specified communication group.")
    .def("get_comm_name", &CollectiveManager::GetCommName,
         "Get inner communicator name for the specified communication group.")
    .def("resume_hccl_comm", &CollectiveManager::ResumeHcclComm, "resume hccl comm.")
    .def("wait_all_comm_init", &CollectiveManager::WaitAllCommInitDone, "Wait for all communicators to be initialized.")
    .def("comm_switch_nic", &CollectiveManager::CommSwitchNic,
         "Switch network interface card between the primary and the secondary NIC.");
  (void)py::class_<TCPStoreClient, std::shared_ptr<TCPStoreClient>>(*m, "TCPStoreClient")
    .def(py::init<const std::string &, int64_t, bool, int64_t, int64_t, bool>())
    .def("get", &TCPStoreClient::GetKey, "Get key.")
    .def("set", &TCPStoreClient::SetKey, "Set key.")
    .def("add", &TCPStoreClient::AddKey, "Add key.")
    .def("delete_key", &TCPStoreClient::DeleteKey, "Delete key.");

  (void)py::class_<PSContext, std::shared_ptr<PSContext>>(*m, "PSContext")
    .def_static("get_instance", &PSContext::instance, "Get PS context instance.")
    .def("set_ps_enable", &PSContext::SetPSEnable, "Set PS mode enabled or disabled.")
    .def("is_ps_mode", &PSContext::is_ps_mode, "Get PS mode enable-disable status.")
    .def("reset", &PSContext::Reset, "Reset PS context attributes.")
    .def("is_worker", &PSContext::is_worker, "Get whether the role of this process is Worker.")
    .def("is_server", &PSContext::is_server, "Get whether the role of this process is PServer.")
    .def("is_scheduler", &PSContext::is_scheduler, "Get whether the role of this process is Scheduler.")
    .def("insert_hash_table_size", &PSContext::InsertHashTableSize, "Insert hash table size.")
    .def("reinsert_hash_table_size", &PSContext::ReInsertHashTableSize,
         "Insert hash table size with new parameter name.")
    .def("insert_accumu_init_info", &PSContext::InsertAccumuInitInfo, "Insert accumulation initialization value.")
    .def("clone_hash_table", &PSContext::CloneHashTable, "Clone a hash table.")
    .def("set_cache_enable", &PSContext::set_cache_enable, "Set ps mode cache enable or not.")
    .def("set_cache_size", &PSContext::set_cache_size, "Set embedding cache size for ps cache mode.")
    .def("cache_enable", &PSContext::cache_enable, "Get ps mode cache enable or not.")
    .def("set_sparse_format", &PSContext::set_sparse_format, "Set the storage format of the embedding table.")
    .def("set_rank_id", &PSContext::set_rank_id, "Set rank id for worker on ps mode.")
    .def("set_server_mode", &PSContext::set_server_mode, "Set server mode.")
    .def("server_mode", &PSContext::server_mode, "Get server mode.")
    .def("set_ms_role", &PSContext::set_ms_role, "Set role for this process.")
    .def("ms_role", &PSContext::ms_role, "Get role for this process.")
    .def("set_worker_num", &PSContext::set_worker_num, "Set worker number.")
    .def("worker_num", &PSContext::worker_num, "Get worker number.")
    .def("set_server_num", &PSContext::set_server_num, "Set server number.")
    .def("server_num", &PSContext::server_num, "Get server number.")
    .def("set_scheduler_ip", &PSContext::set_scheduler_ip, "Set scheduler ip.")
    .def("scheduler_ip", &PSContext::scheduler_ip, "Get scheduler ip.")
    .def("set_scheduler_port", &PSContext::set_scheduler_port, "Set scheduler port.")
    .def("scheduler_port", &PSContext::scheduler_port, "Get scheduler port.")
    .def("set_enable_ssl", &PSContext::set_enable_ssl, "Set PS SSL mode enabled or disabled.")
    .def("enable_ssl", &PSContext::enable_ssl, "Get PS SSL mode enabled or disabled.")
    .def("set_client_password", &PSContext::set_client_password, "Set the client password to decode the p12 file.")
    .def("client_password", &PSContext::client_password, "Get the client password to decode the p12 file.")
    .def("set_server_password", &PSContext::set_server_password, "Set the server password to decode the p12 file.")
    .def("server_password", &PSContext::server_password, "Get the server password to decode the p12 file.")
    .def("set_config_file_path", &PSContext::set_config_file_path,
         "Set configuration files required by the communication layer.")
    .def("config_file_path", &PSContext::config_file_path,
         "Get configuration files required by the communication layer.")
    .def("set_checkpoint_load_status", &PSContext::set_checkpoint_load_status, "Set checkpoint load status.")
    .def("store_warm_up_ptr_by_tensor", &PSContext::StoreWarmUpPtrByTensor, "Store warm up host cache by tensor.")
    .def("store_warm_up_ptr_by_tensor_list", &PSContext::StoreWarmUpPtrByTensorList,
         "Store warm up host cache by tensor list");
  (void)py::class_<DeviceContextManager, std::shared_ptr<DeviceContextManager>>(*m, "DeviceContextManager")
    .def_static("get_instance", &DeviceContextManager::GetInstance, py::return_value_policy::reference,
                "Get device context manager instance.")
    .def("get_device_context", &DeviceContextManager::GetDeviceContext, "Return device context object.");
  (void)py::class_<DeviceContext, std::shared_ptr<DeviceContext>>(*m, "DeviceContext")
    .def("initialized", &DeviceContext::initialized, "Return whether this device backend is successfully initialized.");
  DeviceContextManager::GetInstance().RegisterDeviceStatelessFunc(m);
  (void)py::class_<mindspore::runtime::Process, std::shared_ptr<mindspore::runtime::Process>>(*m,
                                                                                              "CommExecOrderChecker")
    .def_static("get_instance", &mindspore::runtime::Process::GetInstance, py::return_value_policy::reference,
                "Get CommExecOrderChecker instance.")
    .def("start_collect_exec_order", &mindspore::runtime::Process::StartCollectExecOrder, "Start collect exec order.")
    .def("stop_collect_exec_order", &mindspore::runtime::Process::StopCollectExecOrder, "Stop collect exec order.");
}

}  // namespace mindspore
