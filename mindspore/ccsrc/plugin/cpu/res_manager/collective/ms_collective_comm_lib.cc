/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "plugin/cpu/res_manager/collective/ms_collective_comm_lib.h"
#include <complex>
#include <set>
#include "utils/ms_context.h"
#include "include/backend/distributed/constants.h"
#include "runtime/hardware_abstract/collective/collective_communication_lib.h"
#include "plugin/cpu/res_manager/collective/allreduce_impl.h"

namespace mindspore {
namespace device {
namespace {
using complex64 = std::complex<float>;
}  // namespace
namespace cpu {
using GroupOptions = mindspore::device::GroupOptions;
using distributed::GetRetryNumBasedOnScale;
using distributed::kClusterScaleBound;
using distributed::SleepBasedOnScale;
using distributed::cluster::topology::kDefaultRetryInterLower;
using distributed::cluster::topology::kDefaultRetryInterUpper;
using distributed::cluster::topology::kEnvNodeTimeOut;
using distributed::cluster::topology::kEnvRetryIntervalLower;
using distributed::cluster::topology::kEnvRetryIntervalUpper;

// These keywords is used for synchronization of collective communication's metadata(eg. unique id).
constexpr char kGroupInfoPrefix[] = "group_info_";
constexpr char kGroupName[] = "group_name";
constexpr char kUniqueId[] = "unique_id";
MsCollectiveCommLib::MsCollectiveCommLib() {
  // Generate the global group name with node role.
  global_group_name_ = kMCCLGlobalGroupName;
  MS_LOG(INFO) << "Global group name of MindSpore collective communication library is " << global_group_name_;
}

bool MsCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) {
  if (initialized_) {
    MS_LOG(WARNING) << "MsCollectiveCommLib has already been initialized.";
    return true;
  }

  // Only use AllReduceLauncher when this is CPU backend.
  // Do not initialize AllReduceLauncher if this is a large-scale cluster.
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode ||
      (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice &&
       global_rank_size <= kClusterScaleBound) ||
      MsContext::GetInstance()->ascend_soc_version() == kAscendVersion310p) {
    launcher_ = std::make_unique<AllReduceLauncher>();
    CHECK_IF_NULL(launcher_);
    if (!launcher_->Initialize()) {
      MS_LOG(EXCEPTION) << "Failed to initialize the allreduce launcher.";
    }
    node_ = launcher_->collective_node();
    MS_LOG(INFO) << "Node initialize success!";
  }

  client_node_ =
    std::dynamic_pointer_cast<distributed::cluster::topology::TcpNodeBase>(ClusterContext::instance()->node_base());

  constexpr size_t kSizeNum3 = 3;
  std::string timeout_env = common::GetEnv(kEnvNodeTimeOut);
  if (!timeout_env.empty()) {
    MS_LOG(INFO) << "MS_NODE_TIMEOUT env set by user: " << timeout_env;
    retry_count_ = GetRetryNumBasedOnScale(IntToUint(std::stoi(timeout_env)), kSizeNum3);
  } else {
    retry_count_ = GetRetryNumBasedOnScale(kMSCollectiveRetryTime, kSizeNum3);
  }
  MS_LOG(INFO) << "Query retry count is " << retry_count_;

  int random_time_lower = common::GetEnv(kEnvRetryIntervalLower).empty()
                            ? kDefaultRetryInterLower
                            : std::stoi(common::GetEnv(kEnvRetryIntervalLower));
  int random_time_upper = common::GetEnv(kEnvRetryIntervalUpper).empty()
                            ? kDefaultRetryInterUpper
                            : std::stoi(common::GetEnv(kEnvRetryIntervalUpper));
  MS_LOG(INFO) << "Interval of retry allgather hostname lower and upper are " << random_time_lower << " and "
               << random_time_upper;
  rand_distrib_ = std::uniform_int_distribution<>(random_time_lower, random_time_upper);

  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  local_rank_id_ = local_rank_id;
  initialized_ = true;
  finalized_ = false;
  return true;
}

bool MsCollectiveCommLib::Finalize() {
  if (launcher_ != nullptr) {
    return launcher_->Finalize();
  }
  return true;
}

bool MsCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                   const std::vector<uint32_t> &group_ranks, uint32_t local_group_rank,
                                                   uint32_t local_group_size, const GroupOptions &config) {
  if (groups_.count(group_name) != 0) {
    MS_LOG(WARNING) << "The group " << group_name << " has already existed.";
    return true;
  }

  MsCommunicationGroupPtr group = std::make_shared<MsCommunicationGroup>(group_name, group_ranks, global_rank_id_,
                                                                         local_group_rank, local_group_size);
  CHECK_IF_NULL(group);
  groups_[group_name] = group;
  return true;
}

bool MsCollectiveCommLib::AllGatherHostHashName(size_t host_hash_name, std::vector<size_t> *host_hash_names) {
  CHECK_IF_NULL(host_hash_names);
  CHECK_IF_NULL(client_node_);

  auto role = common::GetEnv(distributed::kEnvRole);
  // If init_process_group by TcpStore, the role of tcp_client_node is assigned to 'MS_CLIENT'.
  if (ClusterContext::instance()->is_init_by_store()) {
    role = distributed::kEnvRoleOfClient;
  }
  bool success = false;

  // Retry every random time interval.
  std::random_device rd;
  std::mt19937 gen(rd());
  size_t retry = (client_node_->enable_recovery()) ? SIZE_MAX : retry_count_;
  while (!success && --retry > 0) {
    auto hostnames = client_node_->GetHostNames(role);
    if (hostnames.size() < host_hash_names->size()) {
      auto sleep_time = rand_distrib_(gen);
      MS_LOG(WARNING) << "Retry to get hostname from the meta server node...Retry time: " << retry << "/"
                      << retry_count_ << ", sleep " << sleep_time;
      SleepBasedOnScale(sleep_time);
      continue;
    } else if (hostnames.size() > host_hash_names->size()) {
      MS_LOG(ERROR) << "Invalid number of hostnames, expected number of hostnames: " << host_hash_names->size()
                    << ", actual number of hostnames: " << hostnames.size();
      return false;
    }

    for (size_t i = 0; i < host_hash_names->size(); i++) {
      size_t host_hash = std::hash<std::string>()(hostnames[i]);
      (*host_hash_names)[i] = host_hash;
    }
    success = true;
  }
  if (!success) {
    MS_LOG(EXCEPTION) << "Failed to AllGather host's hash name due to timeout.";
  }

  return true;
}

bool MsCollectiveCommLib::BroadcastUniqueID(const std::string &group_name, size_t root_info_size, void *root_info) {
  CHECK_IF_NULL(root_info);
  CHECK_IF_NULL(client_node_);
  auto group = GetGroup(group_name);
  CHECK_IF_NULL(group);

  uint32_t group_rank_id = group->GetGroupRank(client_node_->rank_id());
  if (group_rank_id == 0) {
    while (!SendUniqueID(group_name, root_info_size, root_info)) {
      MS_LOG(WARNING) << "Send unique id to scheduler failed, retrying...";
      if (finalized_.load()) {
        return false;
      }

      std::this_thread::sleep_for(std::chrono::seconds(kWaitDuration));
    }
  } else {
    while (!QueryUniqueID(group_name, root_info_size, root_info)) {
      MS_LOG(WARNING) << "Query unique id from scheduler failed, retrying...";
      if (finalized_.load()) {
        return false;
      }

      std::this_thread::sleep_for(std::chrono::seconds(kWaitDuration));
    }
  }
  return true;
}

void MsCollectiveCommLib::ClearUniqueID(const std::string &group_name) const {
  CHECK_IF_NULL(client_node_);
  // clear unique id after broadcast success.
  std::string node_role_prefix = client_node_->role() + "_";
  std::string group_info_key = node_role_prefix + kGroupInfoPrefix + group_name;
  (void)client_node_->DeleteMetadata(group_info_key);
}

bool MsCollectiveCommLib::SendUniqueID(const std::string &group_name, size_t root_info_size,
                                       const void *root_info) const {
  CHECK_IF_NULL(root_info);
  CHECK_IF_NULL(client_node_);

  // Create the group info which contains the unique id and send it to the meta server.
  std::string node_role_prefix = client_node_->role() + "_";
  std::string group_info_key = node_role_prefix + kGroupInfoPrefix + group_name;

  bool success = false;
  // It this is not recovery scenario, retry for 3*200s, which is 10 minutes.
  const size_t interval = 3;
  size_t retry = (client_node_->enable_recovery()) ? SIZE_MAX : retry_count_;
  while (!success && --retry > 0) {
    success = client_node_->PutMetadata(group_info_key, root_info, root_info_size);
    if (!success) {
      MS_LOG(WARNING) << "Failed to send unique id for group " << group_name << ". Retry time: " << retry << "/"
                      << retry_count_;
      SleepBasedOnScale(interval);
    }
  }
  if (!success) {
    MS_LOG(EXCEPTION) << "Failed to send unique id to the meta server node due to timeout.";
  }
  return true;
}

bool MsCollectiveCommLib::QueryUniqueID(const std::string &group_name, size_t root_info_size, void *root_info) {
  CHECK_IF_NULL(root_info);
  CHECK_IF_NULL(client_node_);

  std::string node_role_prefix = client_node_->role() + "_";
  std::string group_info_key = node_role_prefix + kGroupInfoPrefix + group_name;
  bool success = false;

  // Retry every random time interval.
  std::random_device rd;
  std::mt19937 gen(rd());
  size_t retry = (client_node_->enable_recovery()) ? SIZE_MAX : retry_count_;
  while (!success && --retry > 0) {
    auto unique_id = client_node_->GetMetadata(group_info_key);
    if (unique_id.length() > 0) {
      auto ret = memcpy_s(root_info, root_info_size, unique_id.data(), unique_id.length());
      if (ret != EOK) {
        MS_LOG(WARNING) << "The memcpy_s error, errorno(" << ret << ")";
        return false;
      }
      success = true;
    } else {
      auto sleep_time = rand_distrib_(gen);
      MS_LOG(WARNING) << "Retry to lookup the unique id for group " << group_name
                      << " from the meta server node...Retry time: " << retry << "/" << retry_count_ << ", sleep "
                      << sleep_time;
      SleepBasedOnScale(sleep_time);
    }
  }
  if (!success) {
    const auto &group_info = groups_.at(group_name);
    uint32_t root_rank = group_info->group_ranks().at(0);
    MS_LOG(EXCEPTION)
      << "Failed to fetch the unique id of the collective lib from the meta server node. Maybe the root rank process "
         "of this group has exited or has not executed to QueryUniqueID step. Please check root rank: "
      << root_rank << "'s log.";
  }
  return true;
}

bool MsCollectiveCommLib::AllReduce(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                    CollectiveOpReduceType reduce_op, const std::string &group_name, void *) {
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return true;
  }
  static const std::set<CollectiveOpReduceType> valid_ops = {
    CollectiveOpReduceType::Reduce_Sum, CollectiveOpReduceType::Reduce_Max, CollectiveOpReduceType::Reduce_Min,
    CollectiveOpReduceType::Reduce_Prod};
  if (valid_ops.count(reduce_op) == 0) {
    MS_LOG(EXCEPTION) << "AllReduce only support reduce sum, max and min.";
  }
  CommunicationGroupInfo group_info = {};
  if (!CheckIfVal(send_buff, recv_buff, group_name, &group_info)) {
    return false;
  }
  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return CollectiveOpsImpl::GetInstance().AllReduce<int8_t>(send_buff, recv_buff, send_count,
                                                                static_cast<int>(reduce_op), node_, group_info);
    case TypeId::kNumberTypeInt16:
      return CollectiveOpsImpl::GetInstance().AllReduce<int16_t>(send_buff, recv_buff, send_count,
                                                                 static_cast<int>(reduce_op), node_, group_info);
    case TypeId::kNumberTypeInt32:
    case TypeId::kNumberTypeInt:
      return CollectiveOpsImpl::GetInstance().AllReduce<int32_t>(send_buff, recv_buff, send_count,
                                                                 static_cast<int>(reduce_op), node_, group_info);
    case TypeId::kNumberTypeInt64:
      return CollectiveOpsImpl::GetInstance().AllReduce<int64_t>(send_buff, recv_buff, send_count,
                                                                 static_cast<int>(reduce_op), node_, group_info);
    case TypeId::kNumberTypeFloat32:
    case TypeId::kNumberTypeFloat:
      return CollectiveOpsImpl::GetInstance().AllReduce<float>(send_buff, recv_buff, send_count,
                                                               static_cast<int>(reduce_op), node_, group_info);
    case TypeId::kNumberTypeFloat16:
      return CollectiveOpsImpl::GetInstance().AllReduce<float16>(send_buff, recv_buff, send_count,
                                                                 static_cast<int>(reduce_op), node_, group_info);
    case TypeId::kNumberTypeBFloat16:
      return CollectiveOpsImpl::GetInstance().AllReduce<bfloat16>(send_buff, recv_buff, send_count,
                                                                  static_cast<int>(reduce_op), node_, group_info);
    default:
      return false;
  }
}

bool MsCollectiveCommLib::AllGather(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                    const std::string &group_name, void *) {
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return true;
  }
  CommunicationGroupInfo group_info = {};
  if (!CheckIfVal(send_buff, recv_buff, group_name, &group_info)) {
    return false;
  }
  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return CollectiveOpsImpl::GetInstance().AllGather<int8_t>(send_buff, recv_buff, send_count, node_, group_info);
    case TypeId::kNumberTypeInt32:
    case TypeId::kNumberTypeInt:
      return CollectiveOpsImpl::GetInstance().AllGather<int32_t>(send_buff, recv_buff, send_count, node_, group_info);
    case TypeId::kNumberTypeInt64:
      return CollectiveOpsImpl::GetInstance().AllGather<int64_t>(send_buff, recv_buff, send_count, node_, group_info);
    case TypeId::kNumberTypeFloat32:
    case TypeId::kNumberTypeFloat:
      return CollectiveOpsImpl::GetInstance().AllGather<float>(send_buff, recv_buff, send_count, node_, group_info);
    case TypeId::kNumberTypeFloat16:
      return CollectiveOpsImpl::GetInstance().AllGather<float16>(send_buff, recv_buff, send_count, node_, group_info);
    case TypeId::kNumberTypeBFloat16:
      return CollectiveOpsImpl::GetInstance().AllGather<bfloat16>(send_buff, recv_buff, send_count, node_, group_info);
    case TypeId::kNumberTypeUInt8:
      return CollectiveOpsImpl::GetInstance().AllGather<uint8_t>(send_buff, recv_buff, send_count, node_, group_info);
    case TypeId::kNumberTypeUInt16:
      return CollectiveOpsImpl::GetInstance().AllGather<uint16_t>(send_buff, recv_buff, send_count, node_, group_info);
    case TypeId::kNumberTypeUInt32:
      return CollectiveOpsImpl::GetInstance().AllGather<uint32_t>(send_buff, recv_buff, send_count, node_, group_info);
    case TypeId::kNumberTypeUInt64:
      return CollectiveOpsImpl::GetInstance().AllGather<uint64_t>(send_buff, recv_buff, send_count, node_, group_info);
    case TypeId::kNumberTypeFloat64:
      return CollectiveOpsImpl::GetInstance().AllGather<double>(send_buff, recv_buff, send_count, node_, group_info);
    default:
      return false;
  }
}
bool MsCollectiveCommLib::Send(const void *send_buff, size_t send_count, TypeId data_type, uint32_t dst_rank,
                               const std::string &group_name, void *) {
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return true;
  }
  CHECK_IF_NULL(send_buff);
  CommunicationGroupInfo group_info = {};
  if (!CheckIfVal(group_name, &group_info)) {
    return false;
  }
  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return CollectiveOpsImpl::GetInstance().Send<int8_t>(send_buff, send_count, dst_rank, node_, group_info);
    case TypeId::kNumberTypeInt32:
    case TypeId::kNumberTypeInt:
      return CollectiveOpsImpl::GetInstance().Send<int32_t>(send_buff, send_count, dst_rank, node_, group_info);
    case TypeId::kNumberTypeInt64:
      return CollectiveOpsImpl::GetInstance().Send<int64_t>(send_buff, send_count, dst_rank, node_, group_info);
    case TypeId::kNumberTypeFloat32:
    case TypeId::kNumberTypeFloat:
      return CollectiveOpsImpl::GetInstance().Send<float>(send_buff, send_count, dst_rank, node_, group_info);
    case TypeId::kNumberTypeFloat16:
      return CollectiveOpsImpl::GetInstance().Send<float16>(send_buff, send_count, dst_rank, node_, group_info);
    case TypeId::kNumberTypeBFloat16:
      return CollectiveOpsImpl::GetInstance().Send<bfloat16>(send_buff, send_count, dst_rank, node_, group_info);
    case TypeId::kNumberTypeUInt8:
      return CollectiveOpsImpl::GetInstance().Send<uint8_t>(send_buff, send_count, dst_rank, node_, group_info);
    case TypeId::kNumberTypeUInt16:
      return CollectiveOpsImpl::GetInstance().Send<uint16_t>(send_buff, send_count, dst_rank, node_, group_info);
    case TypeId::kNumberTypeUInt32:
      return CollectiveOpsImpl::GetInstance().Send<uint32_t>(send_buff, send_count, dst_rank, node_, group_info);
    case TypeId::kNumberTypeUInt64:
      return CollectiveOpsImpl::GetInstance().Send<uint64_t>(send_buff, send_count, dst_rank, node_, group_info);
    case TypeId::kNumberTypeFloat64:
      return CollectiveOpsImpl::GetInstance().Send<double>(send_buff, send_count, dst_rank, node_, group_info);
    default:
      return false;
  }
}
bool MsCollectiveCommLib::Recv(void *recv_buff, size_t recv_count, TypeId data_type, uint32_t src_rank,
                               const std::string &group_name, void *) {
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return true;
  }
  CHECK_IF_NULL(recv_buff);
  CommunicationGroupInfo group_info = {};
  if (!CheckIfVal(group_name, &group_info)) {
    return false;
  }
  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return CollectiveOpsImpl::GetInstance().Recv<int8_t>(recv_buff, recv_count, src_rank, node_, group_info);
    case TypeId::kNumberTypeInt32:
    case TypeId::kNumberTypeInt:
      return CollectiveOpsImpl::GetInstance().Recv<int32_t>(recv_buff, recv_count, src_rank, node_, group_info);
    case TypeId::kNumberTypeInt64:
      return CollectiveOpsImpl::GetInstance().Recv<int64_t>(recv_buff, recv_count, src_rank, node_, group_info);
    case TypeId::kNumberTypeFloat32:
    case TypeId::kNumberTypeFloat:
      return CollectiveOpsImpl::GetInstance().Recv<float>(recv_buff, recv_count, src_rank, node_, group_info);
    case TypeId::kNumberTypeFloat16:
      return CollectiveOpsImpl::GetInstance().Recv<float16>(recv_buff, recv_count, src_rank, node_, group_info);
    case TypeId::kNumberTypeBFloat16:
      return CollectiveOpsImpl::GetInstance().Recv<bfloat16>(recv_buff, recv_count, src_rank, node_, group_info);
    case TypeId::kNumberTypeUInt8:
      return CollectiveOpsImpl::GetInstance().Recv<uint8_t>(recv_buff, recv_count, src_rank, node_, group_info);
    case TypeId::kNumberTypeUInt16:
      return CollectiveOpsImpl::GetInstance().Recv<uint16_t>(recv_buff, recv_count, src_rank, node_, group_info);
    case TypeId::kNumberTypeUInt32:
      return CollectiveOpsImpl::GetInstance().Recv<uint32_t>(recv_buff, recv_count, src_rank, node_, group_info);
    case TypeId::kNumberTypeUInt64:
      return CollectiveOpsImpl::GetInstance().Recv<uint64_t>(recv_buff, recv_count, src_rank, node_, group_info);
    case TypeId::kNumberTypeFloat64:
      return CollectiveOpsImpl::GetInstance().Recv<double>(recv_buff, recv_count, src_rank, node_, group_info);
    default:
      return false;
  }
}

bool MsCollectiveCommLib::Gather(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                 uint32_t root_rank, const std::string &group_name, void *) {
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return true;
  }
  CommunicationGroupInfo group_info = {};
  if (!CheckIfVal(send_buff, recv_buff, group_name, &group_info)) {
    return false;
  }
  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return CollectiveOpsImpl::GetInstance().Gather<int8_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                             group_info);
    case TypeId::kNumberTypeInt32:
    case TypeId::kNumberTypeInt:
      return CollectiveOpsImpl::GetInstance().Gather<int32_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                              group_info);
    case TypeId::kNumberTypeInt64:
      return CollectiveOpsImpl::GetInstance().Gather<int64_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                              group_info);
    case TypeId::kNumberTypeFloat32:
    case TypeId::kNumberTypeFloat:
      return CollectiveOpsImpl::GetInstance().Gather<float>(send_buff, recv_buff, send_count, root_rank, node_,
                                                            group_info);
    case TypeId::kNumberTypeFloat16:
      return CollectiveOpsImpl::GetInstance().Gather<float16>(send_buff, recv_buff, send_count, root_rank, node_,
                                                              group_info);
    case TypeId::kNumberTypeBFloat16:
      return CollectiveOpsImpl::GetInstance().Gather<bfloat16>(send_buff, recv_buff, send_count, root_rank, node_,
                                                               group_info);
    case TypeId::kNumberTypeUInt8:
      return CollectiveOpsImpl::GetInstance().Gather<uint8_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                              group_info);
    case TypeId::kNumberTypeUInt16:
      return CollectiveOpsImpl::GetInstance().Gather<uint16_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                               group_info);
    case TypeId::kNumberTypeUInt32:
      return CollectiveOpsImpl::GetInstance().Gather<uint32_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                               group_info);
    case TypeId::kNumberTypeUInt64:
      return CollectiveOpsImpl::GetInstance().Gather<uint64_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                               group_info);
    case TypeId::kNumberTypeFloat64:
      return CollectiveOpsImpl::GetInstance().Gather<double>(send_buff, recv_buff, send_count, root_rank, node_,
                                                             group_info);
    default:
      return false;
  }
}

bool MsCollectiveCommLib::Scatter(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                  uint32_t root_rank, const std::string &group_name, void *) {
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return true;
  }
  CommunicationGroupInfo group_info = {};
  if (!CheckIfVal(send_buff, recv_buff, group_name, &group_info)) {
    return false;
  }
  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return CollectiveOpsImpl::GetInstance().Scatter<int8_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                              group_info);
    case TypeId::kNumberTypeInt32:
    case TypeId::kNumberTypeInt:
      return CollectiveOpsImpl::GetInstance().Scatter<int32_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                               group_info);
    case TypeId::kNumberTypeInt64:
      return CollectiveOpsImpl::GetInstance().Scatter<int64_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                               group_info);
    case TypeId::kNumberTypeFloat32:
    case TypeId::kNumberTypeFloat:
      return CollectiveOpsImpl::GetInstance().Scatter<float>(send_buff, recv_buff, send_count, root_rank, node_,
                                                             group_info);
    case TypeId::kNumberTypeFloat16:
      return CollectiveOpsImpl::GetInstance().Scatter<float16>(send_buff, recv_buff, send_count, root_rank, node_,
                                                               group_info);
    case TypeId::kNumberTypeBFloat16:
      return CollectiveOpsImpl::GetInstance().Scatter<bfloat16>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                group_info);
    case TypeId::kNumberTypeUInt8:
      return CollectiveOpsImpl::GetInstance().Scatter<uint8_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                               group_info);
    case TypeId::kNumberTypeUInt16:
      return CollectiveOpsImpl::GetInstance().Scatter<uint16_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                group_info);
    case TypeId::kNumberTypeUInt32:
      return CollectiveOpsImpl::GetInstance().Scatter<uint32_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                group_info);
    case TypeId::kNumberTypeUInt64:
      return CollectiveOpsImpl::GetInstance().Scatter<uint64_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                group_info);
    case TypeId::kNumberTypeFloat64:
      return CollectiveOpsImpl::GetInstance().Scatter<double>(send_buff, recv_buff, send_count, root_rank, node_,
                                                              group_info);
    default:
      return false;
  }
}

uint32_t MsCollectiveCommLib::GetGroupRanks(const std::string &group_name) {
  if (groups_.count(group_name) == 0) {
    MS_LOG(ERROR) << "The group " << group_name << " does not exist.";
    return false;
  }

  auto group = groups_[group_name];
  CHECK_IF_NULL(group);
  return group->group_size();
}

bool MsCollectiveCommLib::CheckIfVal(const std::string &group_name, CommunicationGroupInfo *group_info) {
  CHECK_IF_NULL(node_);
  CHECK_IF_NULL(group_info);

  if (groups_.count(group_name) == 0) {
    MS_LOG(ERROR) << "The group " << group_name << " does not exist.";
    return false;
  }

  auto group = groups_[group_name];
  CHECK_IF_NULL(group);
  group_info->size = group->group_size();
  group_info->global_rank = global_rank_id_;
  group_info->group_ranks = group->group_ranks();
  group_info->global_to_group_ranks = group->global_to_group_ranks();
  group_info->group_to_global_ranks = group->group_to_global_ranks();
  return true;
}

bool MsCollectiveCommLib::CheckIfVal(const void *send_buff, void *recv_buff, const std::string &group_name,
                                     CommunicationGroupInfo *group_info) {
  CHECK_IF_NULL(send_buff);
  CHECK_IF_NULL(recv_buff);
  return CheckIfVal(group_name, group_info);
}

bool MsCollectiveCommLib::Broadcast(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                    uint32_t root_rank, const std::string &group_name, void *) {
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return true;
  }
  CommunicationGroupInfo group_info = {};
  if (!CheckIfVal(send_buff, recv_buff, group_name, &group_info)) {
    return false;
  }

  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return CollectiveOpsImpl::GetInstance().Broadcast<int8_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                group_info);
    case TypeId::kNumberTypeInt32:
      [[fallthrough]];
    case TypeId::kNumberTypeInt:
      return CollectiveOpsImpl::GetInstance().Broadcast<int32_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                 group_info);
    case TypeId::kNumberTypeInt64:
      return CollectiveOpsImpl::GetInstance().Broadcast<int64_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                 group_info);
    case TypeId::kNumberTypeFloat32:
      [[fallthrough]];
    case TypeId::kNumberTypeFloat:
      return CollectiveOpsImpl::GetInstance().Broadcast<float>(send_buff, recv_buff, send_count, root_rank, node_,
                                                               group_info);
    case TypeId::kNumberTypeFloat16:
      return CollectiveOpsImpl::GetInstance().Broadcast<float16>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                 group_info);
    case TypeId::kNumberTypeBFloat16:
      return CollectiveOpsImpl::GetInstance().Broadcast<bfloat16>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                  group_info);
    case TypeId::kNumberTypeUInt8:
      return CollectiveOpsImpl::GetInstance().Broadcast<uint8_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                 group_info);
    case TypeId::kNumberTypeUInt16:
      return CollectiveOpsImpl::GetInstance().Broadcast<uint16_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                  group_info);
    case TypeId::kNumberTypeUInt32:
      return CollectiveOpsImpl::GetInstance().Broadcast<uint32_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                  group_info);
    case TypeId::kNumberTypeUInt64:
      return CollectiveOpsImpl::GetInstance().Broadcast<uint64_t>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                  group_info);
    case TypeId::kNumberTypeFloat64:
      return CollectiveOpsImpl::GetInstance().Broadcast<double>(send_buff, recv_buff, send_count, root_rank, node_,
                                                                group_info);
    default:
      return false;
  }
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
