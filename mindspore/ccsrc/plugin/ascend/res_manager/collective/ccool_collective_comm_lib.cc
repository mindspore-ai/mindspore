/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/collective/ccool_collective_comm_lib.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/float8_e4m3fn.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "plugin/ascend/res_manager/collective/leaper_trans.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
namespace mindspore {
namespace device {
namespace ascend {
using GroupOptions = mindspore::device::GroupOptions;
// The keys for parsed information of rank table file.
constexpr char kRankTableDevice[] = "device";
constexpr char kRankTableAzID[] = "az_id";
constexpr char kRankTableRankID[] = "rank_id";
constexpr char kRankTableServerID[] = "server_id";
constexpr char kRankTableServerIP[] = "server_ip";
constexpr char kRankTableServerList[] = "server_list";
constexpr char kRankTableClusterList[] = "cluster_list";
constexpr size_t kPtrIndex0 = 0;
constexpr size_t kPtrIndex1 = 1;
constexpr size_t kPtrIndex2 = 2;
constexpr size_t kPtrIndex3 = 3;
constexpr uint32_t kRankStep = 2;
constexpr uint32_t kStartPort = 21234;
constexpr size_t kFloat32Size = 4;
constexpr size_t kFloat64Size = 8;

bool CCOOLGroupCheckNotEmpty(const std::string &group) {
  if ((group).length() == 0) {
    MS_LOG(ERROR) << "The length of group name should not be 0";
    return false;
  }
  return true;
}

CcoolCollectiveCommLib &CcoolCollectiveCommLib::GetInstance() {
  static CcoolCollectiveCommLib instance;
  return instance;
}

CcoolCollectiveCommLib::CcoolCollectiveCommLib() {
  global_group_name_ = kHCCLGlobalGroupName;
  helper_comm_lib_instance_ = nullptr;
}

bool CcoolCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) {
  if (initialized_) {
    return false;
  }
  std::string global_ranktable_path = common::GetEnv("RANK_TABLE_FILE");
  if (!global_ranktable_path.empty()) {
    rank_az_map_.resize(global_rank_size);
    rank_ip_map_.resize(global_rank_size);
    std::map<string, string> server_az_map = {};
    std::ifstream rank_table(global_ranktable_path);
    rank_table >> global_rank_table_;
    nlohmann::json clusters = global_rank_table_[kRankTableClusterList];
    for (const nlohmann::json &cluster : clusters) {
      string cluster_az_id = cluster[kRankTableAzID];
      for (const nlohmann::json &server : cluster[kRankTableServerList]) {
        server_az_map.emplace(server[kRankTableServerID].get<std::string>(), cluster_az_id);
      }
    }
    nlohmann::json servers = global_rank_table_[kRankTableServerList];
    for (const nlohmann::json &server : servers) {
      std::string server_ip = server[kRankTableServerIP].get<string>();
      std::string server_id = server[kRankTableServerID].get<string>();
      for (const nlohmann::json &rank : server[kRankTableDevice]) {
        uint32_t rank_id = std::stoul(rank[kRankTableRankID].get<string>());
        std::string az_id = server_az_map.at(server_id);
        rank_az_map_[rank_id] = az_id;
        rank_ip_map_[rank_id] = server_ip;
      }
    }
  } else {
    MS_LOG(WARNING) << "RANK_TABLE_FILE not set";
    rank_az_map_ = std::vector<std::string>(global_rank_size, std::string(kCcoolDefaultAzId));
  }
  AscendCollectiveCommLib::GetInstance().Initialize(global_rank, global_rank_size, local_rank_id);
  AscendStreamMng::GetInstance().CreateStream(&inner_stream_id_, 0);
  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  local_rank_id_ = local_rank_id;
  initialized_ = true;
  finalized_ = false;

  return true;
}

bool CcoolCollectiveCommLib::DestroyDeviceCommunicationGroup(const std::string &group_name) {
  return CCOOLGroupCheckNotEmpty(group_name);
}

bool CcoolCollectiveCommLib::DestroyCommunicationGroup(const std::string &group_name) {
  if (!CCOOLGroupCheckNotEmpty(group_name)) {
    return false;
  }
  CHECK_RET((groups_.count(group_name) != 0), true, "The CCOOL group " + group_name + " does not exist.");

  return groups_[group_name]->Finalize();
}

bool CcoolCollectiveCommLib::CreateDeviceCommunicationGroup(const std::string &group_name,
                                                            const std::vector<uint32_t> &group_ranks) {
  return CCOOLGroupCheckNotEmpty(group_name);
}

bool CcoolCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                      const std::vector<uint32_t> &group_ranks,
                                                      uint32_t local_group_rank, uint32_t local_group_size,
                                                      const GroupOptions &config) {
  if (!CCOOLGroupCheckNotEmpty(group_name)) {
    return false;
  }
  CHECK_RET((groups_.count(group_name) == 0), true, "The CCOOL group " + group_name + " has already existed.");

  CcoolCommunicationGroupPtr group = std::make_shared<CcoolCommunicationGroup>(group_name, group_ranks, global_rank_id_,
                                                                               local_group_rank, local_group_size);
  CHECK_IF_NULL(group);

  group->SetHostCommLib(helper_comm_lib_instance_);
  RETURN_IF_FALSE_WITH_LOG(group->InitAscendCommGroup(rank_az_map_, rank_ip_map_),
                           "Ccool failed to InitAscendCommGroup " + group_name);
  groups_[group_name] = group;

  return true;
}

static const std::map<TypeId, size_t> kCcoolCollectiveCommLibDtypeSizeMap = {
  {TypeId::kNumberTypeBool, sizeof(bool)},
  {TypeId::kNumberTypeInt8, sizeof(int8_t)},
  {TypeId::kNumberTypeInt16, sizeof(int16_t)},
  {TypeId::kNumberTypeInt32, sizeof(int32_t)},
  {TypeId::kNumberTypeInt64, sizeof(int64_t)},
  {TypeId::kNumberTypeUInt8, sizeof(uint8_t)},
  {TypeId::kNumberTypeUInt16, sizeof(uint16_t)},
  {TypeId::kNumberTypeUInt32, sizeof(uint32_t)},
  {TypeId::kNumberTypeUInt64, sizeof(uint64_t)},
  {TypeId::kNumberTypeFloat16, sizeof(float16)},
  {TypeId::kNumberTypeFloat32, kFloat32Size},
  {TypeId::kNumberTypeFloat64, kFloat64Size},
  {TypeId::kNumberTypeInt, sizeof(int)},
  {TypeId::kNumberTypeUInt, sizeof(uint)},
  {TypeId::kNumberTypeFloat, sizeof(uint)},
  {TypeId::kNumberTypeBFloat16, sizeof(bfloat16)},
  {TypeId::kNumberTypeHiFloat8, sizeof(hifloat8)},
  {TypeId::kNumberTypeFloat8E5M2, sizeof(float8_e5m2)},
  {TypeId::kNumberTypeFloat8E4M3FN, sizeof(float8_e4m3fn)},
};

size_t CcoolCollectiveCommLib::GetDtypeSize(TypeId type) {
  auto iter = kCcoolCollectiveCommLibDtypeSizeMap.find(type);
  if (iter == kCcoolCollectiveCommLibDtypeSizeMap.end()) {
    MS_LOG(EXCEPTION) << "Unsupported data type " << type;
  }
  return iter->second;
}

bool CcoolCollectiveCommLib::LaunchReduceOperations(void *dst_buff, void *src_buff, void *workspace_buff,
                                                    size_t data_size, size_t count, TypeId data_type,
                                                    CollectiveOpReduceType reduce_op, void *stream_ptr) {
  // step1: create KernelTensor from buff ptr
  abstract::BaseShapePtr tensor_shape = std::make_shared<abstract::TensorShape>();
  tensor_shape->SetShapeVector({static_cast<int64_t>(count)});
  std::shared_ptr<KernelTensor> src_tensor = std::make_shared<KernelTensor>();
  src_tensor->SetType(std::make_shared<TensorType>(TypeIdToType(data_type)));
  src_tensor->SetShape(tensor_shape);
  src_tensor->set_device_ptr(src_buff);
  src_tensor->set_size(data_size);

  std::shared_ptr<KernelTensor> dst_tensor = std::make_shared<KernelTensor>();
  dst_tensor->SetType(std::make_shared<TensorType>(TypeIdToType(data_type)));
  dst_tensor->SetShape(tensor_shape);
  dst_tensor->set_device_ptr(dst_buff);
  dst_tensor->set_size(data_size);

  std::shared_ptr<KernelTensor> temp_tensor = std::make_shared<KernelTensor>();
  temp_tensor->SetType(std::make_shared<TensorType>(TypeIdToType(data_type)));
  temp_tensor->SetShape(tensor_shape);
  temp_tensor->set_device_ptr(workspace_buff);
  temp_tensor->set_size(data_size);

  // step2: create AclnnKernelMod from reduce_op
  std::shared_ptr<kernel::AclnnKernelMod> ops;
  switch (reduce_op) {
    case CollectiveOpReduceType::Reduce_Max: {
      MS_LOG(DEBUG) << "Run reduce_op: MAX start:";
      ops = kernel::Factory<kernel::AclnnKernelMod>::Instance().Create("Maximum");
    } break;
    case CollectiveOpReduceType::Reduce_Min: {
      MS_LOG(DEBUG) << "Run reduce_op: MIN start:";
      ops = kernel::Factory<kernel::AclnnKernelMod>::Instance().Create("Minimum");
    } break;
    case CollectiveOpReduceType::Reduce_Prod: {
      MS_LOG(DEBUG) << "Run reduce_op: PROD start:";
      ops = kernel::Factory<kernel::AclnnKernelMod>::Instance().Create("Mul");
    } break;
    case CollectiveOpReduceType::Reduce_Sum: {
      MS_LOG(DEBUG) << "Run reduce_op: SUM start:";
      ops = kernel::Factory<kernel::AclnnKernelMod>::Instance().Create("Add");
    } break;
    default: {
      MS_LOG(EXCEPTION) << "Get unexpected reduce_op: " << reduce_op;
      ops = nullptr;
    } break;
  }
  if (ops == nullptr) {
    MS_LOG(EXCEPTION) << "Run reduce_op init failed.";
    return false;
  }

  // step3: launch reduce op
  ops->GetWorkSpaceInfo({src_tensor.get(), dst_tensor.get()}, {dst_tensor.get()});
  ops->Launch({src_tensor.get(), dst_tensor.get()}, {temp_tensor.get()}, {dst_tensor.get()}, stream_ptr);
  return true;
}

bool CcoolCollectiveCommLib::InterClusterSimpleAllReduce(void *buff, size_t count, TypeId data_type,
                                                         CollectiveOpReduceType reduce_op,
                                                         CcoolCommunicationGroupPtr group, void *stream_ptr,
                                                         const std::vector<uint32_t> &inter_cluster_ranks) {
  constexpr size_t kSendDataSize = 8;
  size_t dtype_size = GetDtypeSize(data_type);
  size_t size = count * dtype_size;
  void *send_data = nullptr;
  void *recv_data = nullptr;
  void *npu_data = nullptr;
  void *workspace_data = nullptr;
  AscendEvent event{ACL_EVENT_SYNC};
  AscendEvent mem_event{ACL_EVENT_SYNC};
  size_t stream_id = AscendStreamMng::GetInstance().GetStreamId(stream_ptr);
  auto acl_ret = CALL_ASCEND_API(aclrtMallocHost, &send_data, size);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterSimpleAllReduce aclrtMallocHost for send_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtMallocHost, &recv_data, size);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterSimpleAllReduce aclrtMallocHost for recv_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtMalloc, &npu_data, size, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterSimpleAllReduce aclrtMalloc for npu_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtMalloc, &workspace_data, size, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterSimpleAllReduce aclrtMalloc for workspace_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, send_data, size, buff, size, ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterSimpleAllReduce aclrtMemcpyAsync Device to Host failed!";
    return false;
  }
  mem_event.RecordEvent(stream_id);
  mem_event.SyncEvent();
  if (size >= kSendDataSize) {
    MS_LOG(WARNING) << "inter cluster sendrecv send_data = " << reinterpret_cast<float *>(send_data)[0] << ", "
                    << reinterpret_cast<float *>(send_data)[1];
  }

  uint32_t dst_rank = inter_cluster_ranks[0] == global_rank_id_ ? inter_cluster_ranks[1] : inter_cluster_ranks[0];
  LeaperConnInfo conn_info = group->GetConnInfo(dst_rank);
  LeaperTrans::GetInstance().SendRecv(send_data, recv_data, size, size, conn_info);

  if (size >= kSendDataSize) {
    MS_LOG(WARNING) << "inter cluster sendrecv recv_data = " << reinterpret_cast<float *>(recv_data)[0] << ", "
                    << reinterpret_cast<float *>(recv_data)[1];
  }

  acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, npu_data, size, recv_data, size, ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterSimpleAllReduce aclrtMemcpyAsync Host to Device failed!";
    return false;
  }

  bool ret = LaunchReduceOperations(buff, npu_data, workspace_data, size, count, data_type, reduce_op, stream_ptr);
  if (!ret) {
    MS_LOG(ERROR) << "InterClusterSimpleAllReduce LaunchReduceOperations failed!";
    return false;
  }
  event.RecordEvent(stream_id);
  event.SyncEvent();

  acl_ret = CALL_ASCEND_API(aclrtFree, npu_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterSimpleAllReduce aclrtFree npu_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtFree, workspace_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterSimpleAllReduce aclrtFree workspace_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtFreeHost, send_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterSimpleAllReduce aclrtFreeHost send_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtFreeHost, recv_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterSimpleAllReduce aclrtFreeHost recv_data failed!";
    return false;
  }

  return ret;
}

bool CcoolCollectiveCommLib::InterClusterAllReduceCompute(const std::vector<void *> &ptr_vector, size_t count,
                                                          const std::vector<uint32_t> &inter_cluster_ranks,
                                                          CollectiveOpReduceType reduce_op, TypeId data_type,
                                                          CcoolCommunicationGroupPtr group, const size_t &stream_id,
                                                          void *stream_ptr, AscendEvent *mem_event) {
  LeaperConnInfo conn_info;
  LeaperConnInfo conn_info_recv;
  size_t recv_size;
  size_t recv_start;
  size_t send_size;
  size_t send_start;
  size_t rank_size = inter_cluster_ranks.size();
  size_t dtype_size = GetDtypeSize(data_type);
  auto iter = std::find(inter_cluster_ranks.begin(), inter_cluster_ranks.end(), global_rank_id_);
  uint32_t local_rank = static_cast<uint32_t>(std::distance(inter_cluster_ranks.begin(), iter));
  size_t segment_size = count / rank_size;
  const size_t residual = count % rank_size;
  MS_LOG(INFO) << "[AZ > 2] local_rank = " << local_rank << ", rank_size = " << rank_size << ", count = " << count
               << ", segment_size = " << segment_size << ", dtype_size = " << dtype_size << ", residual = " << residual;

  std::vector<size_t> segment_sizes(rank_size, segment_size);
  for (size_t i = 0; i < residual; ++i) {
    segment_sizes[i]++;
  }
  std::vector<size_t> segment_starts(rank_size);
  segment_starts[0] = 0;
  for (size_t i = 1; i < segment_starts.size(); ++i) {
    segment_starts[i] = segment_starts[i - 1] + segment_sizes[i - 1];
  }

  // avoid deadlock
  if (local_rank % kRankStep == 0) {
    conn_info_recv = group->GetConnInfo(inter_cluster_ranks[(local_rank - 1 + rank_size) % rank_size]);
    conn_info = group->GetConnInfo(inter_cluster_ranks[(local_rank + 1) % rank_size]);
  } else {
    conn_info = group->GetConnInfo(inter_cluster_ranks[(local_rank + 1) % rank_size]);
    conn_info_recv = group->GetConnInfo(inter_cluster_ranks[(local_rank - 1 + rank_size) % rank_size]);
  }
  conn_info.recv_fds = conn_info_recv.recv_fds;

  auto acl_ret = 0;
  auto buff = ptr_vector[kPtrIndex0];
  auto send_data = ptr_vector[kPtrIndex1];
  auto npu_data = ptr_vector[kPtrIndex2];
  auto workspace_data = ptr_vector[kPtrIndex3];
  // ring-reducescatter
  for (size_t i = 0; i < rank_size - 1; i++) {
    const size_t send_seg_id = ((local_rank - i) + rank_size) % rank_size;
    const size_t recv_seg_id = ((local_rank - i - 1) + rank_size) % rank_size;
    send_start = segment_starts[send_seg_id] * dtype_size;
    recv_start = segment_starts[recv_seg_id] * dtype_size;
    send_size = segment_sizes[send_seg_id] * dtype_size;
    recv_size = segment_sizes[recv_seg_id] * dtype_size;
    LeaperTrans::GetInstance().SendRecv(reinterpret_cast<uint8_t *>(send_data) + send_start,
                                        reinterpret_cast<uint8_t *>(send_data) + recv_start, send_size, recv_size,
                                        conn_info);
    // Wait for recv to complete before reduction
    acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, reinterpret_cast<uint8_t *>(npu_data) + recv_start, recv_size,
                              reinterpret_cast<uint8_t *>(send_data) + recv_start, recv_size, ACL_MEMCPY_HOST_TO_DEVICE,
                              stream_ptr);
    if (acl_ret != ACL_RT_SUCCESS) {
      MS_LOG(ERROR) << "InterClusterAllReduce aclrtMemcpyAsync Host to Device failed!";
      return false;
    }
    LaunchReduceOperations(reinterpret_cast<uint8_t *>(buff) + recv_start,
                           reinterpret_cast<uint8_t *>(npu_data) + recv_start, workspace_data, recv_size,
                           segment_sizes[recv_seg_id], data_type, reduce_op, stream_ptr);
    acl_ret =
      CALL_ASCEND_API(aclrtMemcpyAsync, reinterpret_cast<uint8_t *>(send_data) + recv_start, recv_size,
                      reinterpret_cast<uint8_t *>(buff) + recv_start, recv_size, ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr);
    if (acl_ret != ACL_RT_SUCCESS) {
      MS_LOG(ERROR) << "InterClusterAllReduce aclrtMemcpyAsync Device to Host failed!";
      return false;
    }
    mem_event->RecordEvent(stream_id);
    mem_event->SyncEvent();
  }

  // ring-allgather
  for (size_t i = 0; i < rank_size - 1; ++i) {
    const size_t send_seg_id = ((local_rank - i + 1) + rank_size) % rank_size;
    const size_t recv_seg_id = ((local_rank - i) + rank_size) % rank_size;
    send_start = segment_starts[send_seg_id] * dtype_size;
    recv_start = segment_starts[recv_seg_id] * dtype_size;
    send_size = segment_sizes[send_seg_id] * dtype_size;
    recv_size = segment_sizes[recv_seg_id] * dtype_size;
    LeaperTrans::GetInstance().SendRecv(reinterpret_cast<uint8_t *>(send_data) + send_start,
                                        reinterpret_cast<uint8_t *>(send_data) + recv_start, send_size, recv_size,
                                        conn_info);
  }
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllReduceCompute failed!";
    return false;
  }
  return true;
}

bool CcoolCollectiveCommLib::InterClusterAllReduce(void *buff, size_t count, TypeId data_type,
                                                   CollectiveOpReduceType reduce_op, CcoolCommunicationGroupPtr group,
                                                   void *stream_ptr) {
  const std::vector<uint32_t> inter_cluster_ranks = group->GetInterClusterRanks();
  // Simplify the process by using send recv instead of allreduce in two AZ communication sets.
  if (inter_cluster_ranks.size() == kRankStep) {
    return InterClusterSimpleAllReduce(buff, count, data_type, reduce_op, group, stream_ptr, inter_cluster_ranks);
  }
  AscendEvent event{ACL_EVENT_SYNC};
  AscendEvent mem_event{ACL_EVENT_SYNC};
  size_t stream_id = AscendStreamMng::GetInstance().GetStreamId(stream_ptr);
  size_t dtype_size = GetDtypeSize(data_type);
  size_t size = count * dtype_size;
  void *send_data = nullptr;
  void *npu_data = nullptr;
  void *workspace_data = nullptr;
  auto acl_ret = CALL_ASCEND_API(aclrtMallocHost, &send_data, size);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllReduce aclrtMallocHost for send_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtMalloc, &npu_data, size, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllReduce aclrtMalloc for npu_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtMalloc, &workspace_data, size, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllReduce aclrtMalloc for workspace_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, send_data, size, buff, size, ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllReduce aclrtMemcpyAsync Device to Host failed!";
    return false;
  }
  mem_event.RecordEvent(stream_id);
  mem_event.SyncEvent();
  std::vector<void *> ptr_vector = {buff, send_data, npu_data, workspace_data};

  auto ret = InterClusterAllReduceCompute(ptr_vector, count, inter_cluster_ranks, reduce_op, data_type, group,
                                          stream_id, stream_ptr, &mem_event);
  if (!ret) {
    MS_LOG(ERROR) << "InterClusterAllReduce run compute failed!";
    return false;
  }

  acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, buff, size, send_data, size, ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllReduce aclrtMemcpyAsync Host to Device failed!";
    return false;
  }
  event.RecordEvent(stream_id);
  event.SyncEvent();
  acl_ret = CALL_ASCEND_API(aclrtFree, npu_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllReduce aclrtFree npu_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtFree, workspace_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllReduce aclrtFree workspace_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtFreeHost, send_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllReduce aclrtFreeHost send_data failed!";
    return false;
  }
  return true;
}

bool CcoolCollectiveCommLib::InterClusterAllGather(void *send_buff, std::vector<void *> recv_buff_list, size_t size,
                                                   CcoolCommunicationGroupPtr group, void *stream_ptr) {
  const std::vector<uint32_t> inter_cluster_ranks = group->GetInterClusterRanks();
  auto iter = std::find(inter_cluster_ranks.begin(), inter_cluster_ranks.end(), global_rank_id_);
  uint32_t local_rank = static_cast<uint32_t>(std::distance(inter_cluster_ranks.begin(), iter));
  MS_LOG(INFO) << "inter cluster allgather ranks = " << inter_cluster_ranks
               << ", inter cluster local rank = " << local_rank;
  AscendEvent event{ACL_EVENT_SYNC};
  AscendEvent mem_event{ACL_EVENT_SYNC};
  LeaperConnInfo conn_info;
  LeaperConnInfo conn_info_recv;
  size_t stream_id = AscendStreamMng::GetInstance().GetStreamId(stream_ptr);
  auto acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, recv_buff_list[local_rank], size, send_buff, size,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllGather aclrtMemcpyAsync device to device failed!";
    return false;
  }
  void *send_data = nullptr;
  void *recv_data = nullptr;
  size_t rank_size = inter_cluster_ranks.size();
  acl_ret = CALL_ASCEND_API(aclrtMallocHost, &send_data, size);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllGather aclrtMallocHost for send_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtMallocHost, &recv_data, size);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllGather aclrtMallocHost for recv_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, send_data, size, recv_buff_list[local_rank], size,
                            ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllGather aclrtMemcpyAsync device to host failed!";
    return false;
  }
  mem_event.RecordEvent(stream_id);
  mem_event.SyncEvent();
  if (local_rank % kRankStep == 0) {
    conn_info_recv = group->GetConnInfo(inter_cluster_ranks[(local_rank - 1 + rank_size) % rank_size]);
    conn_info = group->GetConnInfo(inter_cluster_ranks[(local_rank + 1) % rank_size]);
  } else {
    conn_info = group->GetConnInfo(inter_cluster_ranks[(local_rank + 1) % rank_size]);
    conn_info_recv = group->GetConnInfo(inter_cluster_ranks[(local_rank - 1 + rank_size) % rank_size]);
  }
  conn_info.recv_fds = conn_info_recv.recv_fds;
  // ring-allgather
  for (size_t i = 0; i < rank_size - 1; ++i) {
    const size_t recv_seg_id = ((local_rank - i - 1) + rank_size) % rank_size;
    LeaperTrans::GetInstance().SendRecv(send_data, recv_data, size, size, conn_info);
    acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, recv_buff_list[recv_seg_id], size, recv_data, size,
                              ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
    if (acl_ret != ACL_RT_SUCCESS) {
      MS_LOG(ERROR) << "InterClusterAllGather aclrtMemcpyAsync host to device failed!";
      return false;
    }
    acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, send_data, size, recv_buff_list[recv_seg_id], size,
                              ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr);
    if (acl_ret != ACL_RT_SUCCESS) {
      MS_LOG(ERROR) << "InterClusterAllGather aclrtMemcpyAsync device to host failed!";
      return false;
    }
    event.RecordEvent(stream_id);
    event.SyncEvent();
  }

  acl_ret = CALL_ASCEND_API(aclrtFreeHost, send_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllGather aclrtFreeHost send_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtFreeHost, recv_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterAllGather aclrtFreeHost recv_data failed!";
    return false;
  }
  return true;
}

bool CcoolCollectiveCommLib::InterClusterReduceScatter(const std::vector<void *> &send_buff_list, void *recv_buff,
                                                       size_t recv_count, TypeId data_type,
                                                       CollectiveOpReduceType reduce_op,
                                                       CcoolCommunicationGroupPtr group, void *stream_ptr) {
  const std::vector<uint32_t> inter_cluster_ranks = group->GetInterClusterRanks();
  auto iter = std::find(inter_cluster_ranks.begin(), inter_cluster_ranks.end(), global_rank_id_);
  uint32_t local_rank = static_cast<uint32_t>(std::distance(inter_cluster_ranks.begin(), iter));
  size_t stream_id = AscendStreamMng::GetInstance().GetStreamId(stream_ptr);
  MS_LOG(INFO) << "inter cluster reduce scatter ranks = " << inter_cluster_ranks
               << ", inter cluster local rank = " << local_rank << "recv_count = " << recv_count;
  AscendEvent event{ACL_EVENT_SYNC};
  AscendEvent mem_event{ACL_EVENT_SYNC};
  LeaperConnInfo conn_info;
  LeaperConnInfo conn_info_recv;
  void *send_data = nullptr;
  void *recv_data = nullptr;
  void *npu_data = nullptr;
  size_t rank_size = inter_cluster_ranks.size();
  size_t dtype_size = GetDtypeSize(data_type);
  size_t size = recv_count * dtype_size;

  auto acl_ret = CALL_ASCEND_API(aclrtMallocHost, &send_data, size);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterReduceScatter aclrtMallocHost for send_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtMallocHost, &recv_data, size);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterReduceScatter aclrtMallocHost for recv_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtMalloc, &npu_data, size, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterReduceScatter aclrtMalloc for npu_data failed!";
    return false;
  }

  if (local_rank % kRankStep == 0) {
    conn_info_recv = group->GetConnInfo(inter_cluster_ranks[(local_rank - 1 + rank_size) % rank_size]);
    conn_info = group->GetConnInfo(inter_cluster_ranks[(local_rank + 1) % rank_size]);
  } else {
    conn_info = group->GetConnInfo(inter_cluster_ranks[(local_rank + 1) % rank_size]);
    conn_info_recv = group->GetConnInfo(inter_cluster_ranks[(local_rank - 1 + rank_size) % rank_size]);
  }
  conn_info.recv_fds = conn_info_recv.recv_fds;

  acl_ret =
    CALL_ASCEND_API(aclrtMemcpyAsync, send_data, size, send_buff_list[((local_rank - 1) + rank_size) % rank_size], size,
                    ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterReduceScatter aclrtMemcpyAsync device to host failed!";
    return false;
  }
  mem_event.RecordEvent(stream_id);
  mem_event.SyncEvent();
  // ring-reducescatter
  for (size_t i = 0; i < rank_size - 1; i++) {
    const size_t recv_seg_id = ((local_rank - i - 2) + rank_size) % rank_size;
    LeaperTrans::GetInstance().SendRecv(send_data, recv_data, size, size, conn_info);
    // Wait for recv to complete before reduction
    acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, npu_data, size, recv_data, size, ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
    if (acl_ret != ACL_RT_SUCCESS) {
      MS_LOG(ERROR) << "InterClusterReduceScatter aclrtMemcpyAsync host to device failed!";
      return false;
    }
    LaunchReduceOperations(send_buff_list[recv_seg_id], npu_data, send_buff_list[recv_seg_id], size, recv_count,
                           data_type, reduce_op, stream_ptr);
    acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, send_data, size, send_buff_list[recv_seg_id], size,
                              ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr);
    if (acl_ret != ACL_RT_SUCCESS) {
      MS_LOG(ERROR) << "InterClusterReduceScatter aclrtMemcpyAsync device to host failed!";
      return false;
    }

    mem_event.RecordEvent(stream_id);
    mem_event.SyncEvent();
  }
  acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, recv_buff, size, send_data, size, ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterReduceScatter aclrtMemcpyAsync host to device failed!";
    return false;
  }
  event.RecordEvent(stream_id);
  event.SyncEvent();

  acl_ret = CALL_ASCEND_API(aclrtFree, npu_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterReduceScatter aclrtFree npu_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtFreeHost, send_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterReduceScatter aclrtFreeHost send_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtFreeHost, recv_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "InterClusterReduceScatter aclrtFreeHost recv_data failed!";
    return false;
  }
  return true;
}

bool CcoolCollectiveCommLib::AllGather(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                       const std::string &group_name, void *stream) {
  MS_EXCEPTION_IF_NULL(send_buff);
  MS_EXCEPTION_IF_NULL(recv_buff);
  MS_EXCEPTION_IF_NULL(stream);

  auto &comm_lib = AscendCollectiveCommLib::GetInstance();
  CcoolCommunicationGroupPtr group = std::dynamic_pointer_cast<CcoolCommunicationGroup>(groups_[group_name]);
  const std::vector<uint32_t> inter_cluster_ranks = group->GetInterClusterRanks();
  size_t cluster_count = inter_cluster_ranks.size();
  MS_LOG(INFO) << "CcoolAllGather: inter cluster ranks = " << inter_cluster_ranks << ", group_name = " << group_name;
  // compatible with HCCL in one cluster
  if (cluster_count == 1) {
    return comm_lib.AllGather(send_buff, recv_buff, send_count, data_type, group_name, stream);
  }

  MS_LOG(INFO) << "CcoolAllGather start, stream id = " << AscendStreamMng::GetInstance().GetStreamId(stream)
               << ", send_count = " << send_count << ", cluster_count = " << cluster_count;

  // step1: cross az allgather
  std::vector<void *> data_list;
  size_t send_size = send_count * GetDtypeSize(data_type);
  size_t stride_size = send_size * group->GetInnerClusterRanks().size();
  for (size_t i = 0; i < cluster_count; i++) {
    data_list.push_back(static_cast<uint8_t *>(recv_buff) + i * stride_size);
  }
  bool ret = InterClusterAllGather(const_cast<void *>(send_buff), data_list, send_size, group, stream);
  if (!ret) {
    MS_LOG(ERROR) << "CcoolAllGather failed on InterClusterAllGather";
    return ret;
  }

  // step2: inner az allgather
  for (size_t i = 0; i < cluster_count; i++) {
    // note: HCCL support in-place allgather
    ret = comm_lib.AllGather(data_list[i], data_list[i], send_count, data_type, group_name, stream);
    if (!ret) {
      MS_LOG(ERROR) << "CcoolAllGather failed on HcclAllGather";
      break;
    }
  }
  return ret;
}

bool CcoolCollectiveCommLib::AllReduce(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                       CollectiveOpReduceType reduce_op, const std::string &group_name, void *stream) {
  MS_EXCEPTION_IF_NULL(send_buff);
  MS_EXCEPTION_IF_NULL(recv_buff);
  MS_EXCEPTION_IF_NULL(stream);

  auto &comm_lib = AscendCollectiveCommLib::GetInstance();
  CcoolCommunicationGroupPtr group = std::dynamic_pointer_cast<CcoolCommunicationGroup>(groups_[group_name]);
  const std::vector<uint32_t> inter_cluster_ranks = group->GetInterClusterRanks();
  size_t cluster_count = inter_cluster_ranks.size();

  MS_LOG(INFO) << "CcoolAllReduce: inter cluster ranks = " << inter_cluster_ranks << ", group_name = " << group_name;
  // compatible with HCCL in one cluster
  if (cluster_count == 1) {
    return comm_lib.AllReduce(send_buff, recv_buff, send_count, data_type, reduce_op, group_name, stream);
  }

  // record on stream, wait on inner_stream
  AscendEvent event{ACL_EVENT_SYNC};
  AscendEvent rs_event{ACL_EVENT_SYNC};
  AscendEvent ag_event{ACL_EVENT_SYNC};
  AscendEvent mem_event{ACL_EVENT_SYNC};
  size_t stream_id = AscendStreamMng::GetInstance().GetStreamId(stream);
  aclrtStream inner_stream = AscendStreamMng::GetInstance().GetStream(inner_stream_id_);

  // Padding for HcclReduceScatter
  size_t hccl_rank_size = group->GetInnerClusterRanks().size();
  void *send_buff_padding = nullptr;
  void *recv_buff_padding = nullptr;
  uint64_t count = send_count / hccl_rank_size;
  uint64_t remain = send_count % hccl_rank_size;
  size_t buff_size = send_count * GetDtypeSize(data_type);
  if (remain == 0) {
    send_buff_padding = const_cast<void *>(send_buff);
    recv_buff_padding = recv_buff;
  } else {
    count++;
    size_t padding_size = (send_count + hccl_rank_size - remain) * GetDtypeSize(data_type);
    auto acl_ret_0 = CALL_ASCEND_API(aclrtMalloc, &send_buff_padding, padding_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (acl_ret_0 != ACL_RT_SUCCESS) {
      MS_LOG(ERROR) << "CcoolAllReduce aclrtMalloc for send_buff_padding failed!";
      return false;
    }
    acl_ret_0 = CALL_ASCEND_API(aclrtMalloc, &recv_buff_padding, padding_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (acl_ret_0 != ACL_RT_SUCCESS) {
      MS_LOG(ERROR) << "CcoolAllReduce aclrtMalloc for recv_buff_padding failed!";
      return false;
    }
    acl_ret_0 = CALL_ASCEND_API(aclrtMemcpyAsync, send_buff_padding, padding_size, send_buff, buff_size,
                                ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    if (acl_ret_0 != ACL_RT_SUCCESS) {
      MS_LOG(ERROR) << "CcoolAllReduce aclrtMemcpyAsync device to device failed!";
      return false;
    }
  }

  event.RecordEvent(stream_id);
  event.WaitEvent(inner_stream_id_);
  MS_LOG(INFO) << "CcoolAllReduce start, stream id = " << stream_id << ", inner stream id = " << inner_stream_id_
               << ", send_count = " << send_count << ", cluster_count = " << cluster_count
               << ", hccl rank size = " << hccl_rank_size << ", buff_size = " << buff_size << ", remain = " << remain;

  bool ret =
    comm_lib.ReduceScatter(send_buff_padding, recv_buff_padding, count, data_type, reduce_op, group_name, inner_stream);
  if (!ret) {
    MS_LOG(ERROR) << "CcoolAllReduce failed on HcclReduceScatter";
    return ret;
  }
  rs_event.RecordEvent(inner_stream_id_);
  rs_event.WaitEvent(stream_id);
  ret = InterClusterAllReduce(recv_buff_padding, count, data_type, reduce_op, group, stream);
  if (!ret) {
    MS_LOG(ERROR) << "CcoolAllReduce failed on InterClusterAllReduce";
    return ret;
  }

  ret = comm_lib.AllGather(recv_buff_padding, recv_buff_padding, count, data_type, group_name, stream);

  if (remain != 0) {
    auto acl_ret_1 = CALL_ASCEND_API(aclrtMemcpyAsync, recv_buff, buff_size, recv_buff_padding, buff_size,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    if (acl_ret_1 != ACL_RT_SUCCESS) {
      MS_LOG(ERROR) << "CcoolAllReduce aclrtMemcpyAsync device to device failed!";
      return false;
    }
    ag_event.RecordEvent(stream_id);
    ag_event.SyncEvent();
    acl_ret_1 = CALL_ASCEND_API(aclrtFree, send_buff_padding);
    if (acl_ret_1 != ACL_RT_SUCCESS) {
      MS_LOG(ERROR) << "CcoolAllReduce aclrtFree send_buff_padding failed!";
      return false;
    }
    acl_ret_1 = CALL_ASCEND_API(aclrtFree, recv_buff_padding);
    if (acl_ret_1 != ACL_RT_SUCCESS) {
      MS_LOG(ERROR) << "CcoolAllReduce aclrtFree recv_buff_padding failed!";
      return false;
    }
  }
  return ret;
}

bool CcoolCollectiveCommLib::Broadcast(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                       uint32_t root_rank, const std::string &group_name, void *stream) {
  MS_EXCEPTION_IF_NULL(send_buff);
  MS_EXCEPTION_IF_NULL(stream);

  auto &comm_lib = AscendCollectiveCommLib::GetInstance();
  CcoolCommunicationGroupPtr group = std::dynamic_pointer_cast<CcoolCommunicationGroup>(groups_[group_name]);
  const std::vector<uint32_t> inter_cluster_ranks = group->GetInterClusterRanks();
  size_t cluster_count = inter_cluster_ranks.size();
  MS_LOG(INFO) << "CcoolBroadcast: inter cluster ranks = " << inter_cluster_ranks;
  // compatible with HCCL in one cluster
  if (cluster_count == 1) {
    return comm_lib.Broadcast(send_buff, recv_buff, send_count, data_type, root_rank, group_name, stream);
  }

  MS_LOG(ERROR) << "Ccool Broadcast not support";
  return false;
}

bool CcoolCollectiveCommLib::ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count, TypeId data_type,
                                           CollectiveOpReduceType reduce_op, const std::string &group_name,
                                           void *stream) {
  MS_EXCEPTION_IF_NULL(send_buff);
  MS_EXCEPTION_IF_NULL(recv_buff);
  MS_EXCEPTION_IF_NULL(stream);

  auto &comm_lib = AscendCollectiveCommLib::GetInstance();
  CcoolCommunicationGroupPtr group = std::dynamic_pointer_cast<CcoolCommunicationGroup>(groups_[group_name]);
  const std::vector<uint32_t> inter_cluster_ranks = group->GetInterClusterRanks();
  size_t cluster_count = inter_cluster_ranks.size();
  MS_LOG(INFO) << "CcoolReduceScatter: inter cluster ranks = " << inter_cluster_ranks
               << ", group_name = " << group_name;
  // compatible with HCCL in one cluster
  if (cluster_count == 1) {
    return comm_lib.ReduceScatter(send_buff, recv_buff, recv_count, data_type, reduce_op, group_name, stream);
  }

  MS_LOG(WARNING) << "CcoolReduceScatter start, stream id = " << AscendStreamMng::GetInstance().GetStreamId(stream)
                  << ", recv_count = " << recv_count << ", cluster_count = " << cluster_count;

  // Prepare buffer for inner az reduce scatter
  std::vector<void *> send_data_list;
  std::vector<void *> recv_data_list;
  size_t recv_size = recv_count * GetDtypeSize(data_type);
  size_t stride_size = recv_size * group->GetInnerClusterRanks().size();
  auto acl_ret = 0;
  for (size_t i = 0; i < cluster_count; i++) {
    send_data_list.push_back(static_cast<uint8_t *>(const_cast<void *>(send_buff)) + i * stride_size);

    void *recv_data_buff = nullptr;
    acl_ret = CALL_ASCEND_API(aclrtMalloc, &recv_data_buff, recv_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    recv_data_list.push_back(recv_data_buff);
  }
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "CcoolReduceScatter aclrtMalloc for recv_data_buff failed!";
    return false;
  }

  // step1: inner az reduce scatter
  bool ret = false;
  for (size_t i = 0; i < cluster_count; i++) {
    // note: HCCL support in-place reduce scatter
    ret = comm_lib.ReduceScatter(send_data_list[i], recv_data_list[i], recv_count, data_type, reduce_op, group_name,
                                 stream);
    if (!ret) {
      MS_LOG(EXCEPTION) << "CcoolReduceScatter failed on HcclReduceScatter";
      break;
    }
  }

  // step2: cross az reduce scatter
  ret = InterClusterReduceScatter(recv_data_list, recv_buff, recv_count, data_type, reduce_op, group, stream);
  if (!ret) {
    MS_LOG(EXCEPTION) << "CcoolReduceScatter failed on InterClusterReduceScatter";
  }
  for (size_t i = 0; i < cluster_count; i++) {
    acl_ret = CALL_ASCEND_API(aclrtFree, recv_data_list[i]);
  }
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "CcoolReduceScatter aclrtFree recv_data_list failed!";
    return false;
  }

  return ret;
}

bool CcoolCollectiveCommLib::Send(const void *send_buff, size_t count, TypeId data_type, uint32_t peer,
                                  const std::string &group_name, void *stream) {
  MS_EXCEPTION_IF_NULL(send_buff);
  MS_EXCEPTION_IF_NULL(stream);

  auto &comm_lib = AscendCollectiveCommLib::GetInstance();
  CcoolCommunicationGroupPtr group = std::dynamic_pointer_cast<CcoolCommunicationGroup>(groups_[group_name]);
  const std::vector<uint32_t> inner_cluster_ranks = group->GetInnerClusterRanks();
  auto group_to_global_ranks = group->group_to_global_ranks();
  uint32_t global_peer = group_to_global_ranks[peer];
  MS_LOG(WARNING) << " global_peer = " << global_peer;
  MS_LOG(INFO) << "CcoolSend: inner cluster ranks = " << inner_cluster_ranks << ", peer = " << peer;
  peer = global_peer;
  // compatible with HCCL in one cluster
  if (std::find(inner_cluster_ranks.begin(), inner_cluster_ranks.end(), peer) != inner_cluster_ranks.end()) {
    auto iter = std::find(inner_cluster_ranks.begin(), inner_cluster_ranks.end(), peer);
    uint32_t peer_inner = static_cast<uint32_t>(std::distance(inner_cluster_ranks.begin(), iter));
    return comm_lib.Send(send_buff, count, data_type, peer_inner, group_name, stream);
  }

  MS_LOG(INFO) << "Ccool Send, peer = " << peer << ", group_name = " << group_name;
  AscendEvent mem_event{ACL_EVENT_SYNC};
  void *send_data = nullptr;
  size_t dtype_size = GetDtypeSize(data_type);
  size_t size = count * dtype_size;
  size_t stream_id = AscendStreamMng::GetInstance().GetStreamId(stream);
  auto acl_ret = CALL_ASCEND_API(aclrtMallocHost, &send_data, size);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "Ccool Send aclrtMallocHost for send_data failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, send_data, size, send_buff, size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "Ccool Send aclrtMemcpyAsync device to host failed!";
    return false;
  }
  mem_event.RecordEvent(stream_id);
  mem_event.SyncEvent();
  MS_LOG(INFO) << "Ccool Send, src_port = " << kStartPort + global_rank_id_ << ", dest_port = " << kStartPort + peer;
  LeaperConnInfo conn_info = group->GetConnInfo(peer);

  LeaperTrans::GetInstance().SendRecv(send_data, nullptr, size, size, conn_info);
  acl_ret = CALL_ASCEND_API(aclrtFreeHost, send_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "Ccool Send aclrtFreeHost send_data failed!";
    return false;
  }
  return true;
}

bool CcoolCollectiveCommLib::Recv(void *recv_buff, size_t count, TypeId data_type, uint32_t peer,
                                  const std::string &group_name, void *stream) {
  MS_EXCEPTION_IF_NULL(recv_buff);
  MS_EXCEPTION_IF_NULL(stream);

  auto &comm_lib = AscendCollectiveCommLib::GetInstance();
  CcoolCommunicationGroupPtr group = std::dynamic_pointer_cast<CcoolCommunicationGroup>(groups_[group_name]);
  const std::vector<uint32_t> inner_cluster_ranks = group->GetInnerClusterRanks();
  auto group_to_global_ranks = group->group_to_global_ranks();
  uint32_t global_peer = group_to_global_ranks[peer];
  MS_LOG(INFO) << "global_peer = " << global_peer;
  MS_LOG(INFO) << "CcoolRecv: inner cluster ranks = " << inner_cluster_ranks << ", peer = " << peer;
  peer = global_peer;
  // compatible with HCCL in one cluster
  if (std::find(inner_cluster_ranks.begin(), inner_cluster_ranks.end(), peer) != inner_cluster_ranks.end()) {
    auto iter = std::find(inner_cluster_ranks.begin(), inner_cluster_ranks.end(), peer);
    uint32_t peer_inner = static_cast<uint32_t>(std::distance(inner_cluster_ranks.begin(), iter));
    return comm_lib.Recv(recv_buff, count, data_type, peer_inner, group_name, stream);
  }

  MS_LOG(INFO) << "Ccool Recv, peer = " << peer << ", group_name = " << group_name;
  AscendEvent mem_event{ACL_EVENT_SYNC};
  void *recv_data = nullptr;
  size_t dtype_size = GetDtypeSize(data_type);
  size_t size = count * dtype_size;
  size_t stream_id = AscendStreamMng::GetInstance().GetStreamId(stream);
  auto acl_ret = CALL_ASCEND_API(aclrtMallocHost, &recv_data, size);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "Ccool Recv aclrtMallocHost for recv_data failed!";
    return false;
  }
  MS_LOG(INFO) << "Ccool Recv, src_port = " << kStartPort + global_rank_id_ << ", dest_port = " << kStartPort + peer;
  LeaperConnInfo conn_info = group->GetConnInfo(peer);

  LeaperTrans::GetInstance().SendRecv(nullptr, recv_data, size, size, conn_info);
  acl_ret = CALL_ASCEND_API(aclrtMemcpyAsync, recv_buff, size, recv_data, size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "Ccool Recv aclrtMemcpyAsync host to device failed!";
    return false;
  }
  mem_event.RecordEvent(stream_id);
  mem_event.SyncEvent();
  acl_ret = CALL_ASCEND_API(aclrtFreeHost, recv_data);
  if (acl_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "Ccool Recv aclrtFreeHost recv_data failed!";
    return false;
  }
  return true;
}

void CcoolCollectiveCommLib::SetHelperCommLib(CollectiveCommunicationLib *comm_lib) {
  MS_EXCEPTION_IF_NULL(comm_lib);
  helper_comm_lib_instance_ = comm_lib;
}

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
