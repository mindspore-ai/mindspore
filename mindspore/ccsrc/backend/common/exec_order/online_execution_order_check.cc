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
#include <memory>
#include <map>
#include <utility>
#include <string>
#include <vector>
#include "ops_utils/op_constants.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "mindspore/core/include/utils/ms_utils.h"
#include "runtime/hardware_abstract/utils.h"
#include "include/utils/comm_manager.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "include/backend/common/exec_order/online_execution_order_check.h"

namespace mindspore {
namespace runtime {
const size_t Process::kMaxAllGatherBuffSize = 1000;
const uint64_t Process::kFnvOffsetBasis = 14695981039346656037ULL;
const char Process::kCommGroupName[] = "hccl_world_group";
const char Process::kSendReceive[] = "S";

std::string GetRankByAttrName(const CNodePtr &cnode, const std::vector<uint32_t> &comm_ranks,
                              const std::string &attr_name) {
  MS_EXCEPTION_IF_NULL(cnode);

  if (!common::AnfAlgo::HasNodeAttr(attr_name, cnode)) {
    MS_LOG(EXCEPTION) << "Attribute " << attr_name << " not found in the node.";
  }

  int64_t rank_attr = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, attr_name);
  if (rank_attr < 0 || static_cast<size_t>(rank_attr) >= comm_ranks.size()) {
    MS_LOG(EXCEPTION) << "Invalid rank_attr value: " << rank_attr << ", or out of range for comm_ranks with size "
                      << comm_ranks.size();
  }

  return std::to_string(comm_ranks[static_cast<size_t>(rank_attr)]);
}

bool ParseCheckIteration(const std::string &check_iteration_from_user) {
  // Check if input is a decimal
  if (check_iteration_from_user.find('.') != std::string::npos) {
    MS_LOG(WARNING) << "Invalid value for check_iteration_from_user: " << check_iteration_from_user
                    << ". It should be a positive integer, not a decimal.";
    return false;
  }

  try {
    int check_iteration = std::stoll(check_iteration_from_user);
    if (check_iteration < 0) {
      MS_LOG(WARNING) << "Invalid value for check_iteration_from_user: " << check_iteration_from_user
                      << ". It should be 0 or a positive integer.";
      return false;
    }
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Invalid value for check_iteration_from_user: " << check_iteration_from_user
                    << ". It should be 0 or a positive integer.";
    return false;
  }

  return true;
}

uint32_t Process::GetRankSize() {
  static uint32_t rank_size = 1;
  static bool is_initialized = false;
  if (!is_initialized) {
    rank_size = distributed::collective::CollectiveManager::instance()->global_rank_size();
    is_initialized = true;
  }

  return rank_size;
}

std::string Process::GetRankID() {
  static uint32_t rank_id = 0;
  static bool is_initialized = false;
  if (!is_initialized) {
    if (distributed::collective::CollectiveManager::instance()->initialized()) {
      rank_id = CommManager::GetInstance().GetRank();
    }
    is_initialized = true;
  }

  return std::to_string(rank_id);
}

void Process::CheckCommOrderIteration(size_t total_running_count) {
  auto &cache = KernelCache::GetInstance();
  std::string check_iteration_from_user = GetRuntimeConfigValue(kRuntimeExecutionOrderCheckIteration);
  if (check_iteration_from_user.empty()) {
    return;
  }

  // The function is disabled by default. When the value of kRuntimeCommOrderCheckIteration is 0, only the first step is
  // verified. When the value of kRuntimeCommOrderCheckIteration is set to other, only the first step and the number of
  // steps are verified.
  bool valid_input = ParseCheckIteration(check_iteration_from_user);
  if (!valid_input) {
    return;
  }

  size_t check_iteration = std::stoull(check_iteration_from_user);
  if (check_iteration == 0) {
    cache.need_add = (total_running_count == 1);
    if (total_running_count == 2) {
      static const size_t first_step = 1;
      cache.SwapBuffers(first_step);
      ProcessKernels(first_step);
      ValidateCommGroupExecuteOrders(first_step);
    }
  } else {
    cache.need_add = (total_running_count % check_iteration == 0);
    if ((total_running_count - 1) % check_iteration == 0) {
      cache.SwapBuffers(static_cast<int>(total_running_count - 1));
      ProcessKernels(static_cast<int>(total_running_count - 1));
      ValidateCommGroupExecuteOrders(static_cast<int>(total_running_count - 1));
    }
  }
}

std::string Process::GetGroupFromPrim(const PrimitivePtr &prim) {
  auto group_attr = prim->GetAttr(kAttrGroup);
  if (!group_attr || !group_attr->isa<StringImm>()) {
    return "";
  }

  auto group_ptr = group_attr->cast<StringImmPtr>();
  if (!group_ptr) {
    MS_LOG(WARNING) << "Failed to cast group attribute to StringImm";
    return "";
  }
  return group_ptr->value();
}

std::pair<std::string, std::string> Process::GetKernelShapes(const CNodePtr &kernel) {
  auto input_node = kernel->input(kIndex1);
  std::string input_shape = "UnknownShape";
  if (input_node && input_node->abstract()) {
    auto input_shape_abs = input_node->abstract()->GetShapeTrack();
    input_shape = (input_shape_abs ? input_shape_abs->ToString() : "UnknownShape");
  } else {
    MS_LOG(WARNING) << "Input node or its abstract is null for kernel: " << kernel->fullname_with_scope();
  }

  std::string output_shape = "UnknownShape";
  auto output_abs = kernel->abstract();
  if (output_abs) {
    auto output_shape_abs = output_abs->GetShapeTrack();
    output_shape = (output_shape_abs ? output_shape_abs->ToString() : "UnknownShape");
  } else {
    MS_LOG(WARNING) << "Abstract is null for output node in kernel: " << kernel->fullname_with_scope();
  }
  return {input_shape, output_shape};
}

void Process::ProcessKernels(int step) {
  MS_LOG(INFO) << "The processing hash kernel logic in the online check of the execution sequence of the communication "
                  "operator starts.";
  ProcessResult *result = (step == kPynativeFlag) ? &pynative_results_ : &latest_results_[step];
  std::vector<std::any> kernels = KernelCache::GetInstance().GetBuffers(step);

  if (kernels.empty()) {
    MS_LOG(WARNING) << "No kernels to process.";
    return;
  }

  auto &group_to_hash = result->group_hashes;
  group_to_hash.reserve(kIndex20);

  // Process each kernel
  for (const auto &item : kernels) {
    if (item.type() == typeid(CNodePtr)) {
      auto kernel = std::any_cast<CNodePtr>(item);
      auto prim = GetCNodePrimitive(kernel);
      if (!prim) {
        MS_LOG(WARNING) << "Primitive is null for kernel: " << kernel->fullname_with_scope();
        continue;
      }

      const std::string group = GetGroupFromPrim(prim);
      if (group.empty()) {
        continue;
      }

      auto [input_shape, output_shape] = GetKernelShapes(kernel);

      // Fetch communication ranks for the group
      FetchCommRanksCache(group);

      const std::string primitive_str = prim->ToString();
      if (primitive_str == "Send" || primitive_str == "Receive") {
        // Send | Receive -> srcRank-DestRank Hash(SR-Shape)
        ProcessSendReceive(result, group, kernel, primitive_str, input_shape, output_shape);
      } else {
        // group: Hash(Primitive-InputShape)
        ProcessNormalGroupHash(result, group, primitive_str, input_shape);
      }

      // Skip kernel if it does not have the 'Group' attribute
      if (!common::AnfAlgo::HasNodeAttr(kAttrGroup, kernel)) {
        continue;
      }
    } else if (item.type() == typeid(CommPyboostKernelPtr)) {
      auto kernel = std::any_cast<CommPyboostKernelPtr>(item);
      FetchCommRanksCache(kernel->group);
      if (kernel->primitive == prim::kPrimDistCommIsend->name() ||
          kernel->primitive == prim::kPrimInnerCommIsend->name() ||
          kernel->primitive == prim::kPrimDistCommIrecv->name() ||
          kernel->primitive == prim::kPrimInnerCommIrecv->name()) {
        ProcessSendReceive(result, kernel->group, kernel, kernel->primitive, kernel->input_shape, kernel->output_shape);
      } else {
        ProcessNormalGroupHash(result, kernel->group, kernel->primitive, kernel->input_shape);
      }
    }

    MS_LOG(INFO) << "Process kernel size: " << kernels.size() << " for step: " << step;
  }
}

void Process::AllGatherExecuteOrderHash(std::unique_ptr<char[]> *output_host_buffer, int step) {
  MS_LOG(INFO) << "The processing allgather group hash in the online check of the execution sequence of the "
                  "communication operator starts.";
  ProcessResult process_result;
  if (step == kPynativeFlag) {
    process_result = pynative_results_;
  } else {
    process_result = latest_results_[step];
  }

  const auto &context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  uint32_t device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const std::string &device_target = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  const auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device::GetDeviceTypeByName(device_target), device_id});
  MS_EXCEPTION_IF_NULL(device_context);
  auto comm_stream_id = device_context->device_res_manager_->GetCommunicationStreamIDByGroup(kCommGroupName);

  std::vector<char> input_buffer(kMaxAllGatherBuffSize, '\0');
  size_t offset = 0;
  for (const auto &[group, hash] : process_result.group_hashes) {
    std::string combined = group + ":" + std::to_string(hash) + ",";
    auto ret_code =
      memcpy_s(input_buffer.data() + offset, kMaxAllGatherBuffSize - offset, combined.c_str(), combined.size());
    if (ret_code != EOK) {
      MS_LOG(WARNING) << "Failed to copy data, memcpy_s errorno: " << ret_code;
      return;
    }
    offset += combined.size();
  }

  auto input_device_tensor = std::make_shared<device::DeviceAddress>(
    nullptr, kMaxAllGatherBuffSize, device::GetDeviceTypeByName(device_target), comm_stream_id);

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "AllocMemoryForCheckCommExecutionOrder",
                                                 "AllocMemoryForCheckCommExecutionOrder", "", false);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "AllocMemoryForCheckCommExecutionOrder",
                                                 device::tracker::MemType::kOther, input_device_tensor->GetSize(),
                                                 input_device_tensor.get());

  if (!device_context->device_res_manager_->AllocateMemory(input_device_tensor.get(), comm_stream_id)) {
    MS_LOG(WARNING) << "Allocate device memory failed!";
    return;
  }
  device_context->device_res_manager_->SyncAllStreams();
  bool sync_ret = device_context->device_res_manager_->Copy(input_device_tensor->GetMutablePtr(), input_buffer.data(),
                                                            kMaxAllGatherBuffSize, device::CopyType::kH2D,
                                                            input_device_tensor->stream_id());
  if (!sync_ret) {
    MS_LOG(WARNING) << "Failed to sync input data to device.";
    return;
  }

  auto output_device_tensor = std::make_shared<device::DeviceAddress>(
    nullptr, kMaxAllGatherBuffSize * GetRankSize(), device::GetDeviceTypeByName(device_target), comm_stream_id);

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "AllocMemoryForCheckCommExecutionOrder",
                                                 device::tracker::MemType::kOther, output_device_tensor->GetSize(),
                                                 output_device_tensor.get());

  if (!device_context->device_res_manager_->AllocateMemory(output_device_tensor.get(), comm_stream_id)) {
    MS_LOG(WARNING) << "Allocate device memory failed!";
    return;
  }

  auto comm_lib = distributed::collective::CollectiveManager::instance()->device_comm_lib();
  MS_EXCEPTION_IF_NULL(comm_lib);
  MS_LOG(WARNING) << "comm_stream_id:" << comm_stream_id;
  void *stream_ptr = device_context->device_res_manager_->GetStream(comm_stream_id);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto ret = device_context->device_res_manager_->SyncAllStreams();
  if (!ret) {
    MS_LOG(WARNING) << "Sync Stream failed";
    return;
  }

  if (!comm_lib->AllGather(input_device_tensor->GetMutablePtr(), output_device_tensor->GetMutablePtr(),
                           kMaxAllGatherBuffSize, kNumberTypeUInt8, kCommGroupName, stream_ptr)) {
    MS_LOG(WARNING) << "AllGather operation failed for step " << step;
    return;
  }

  auto ret_ok = device_context->device_res_manager_->SyncAllStreams();
  if (!ret_ok) {
    MS_LOG(WARNING) << "Sync Stream failed";
    return;
  }
  device_context->device_res_manager_->SyncAllStreams();
  device_context->device_res_manager_->Copy((*output_host_buffer).get(), output_device_tensor->GetMutablePtr(),
                                            kMaxAllGatherBuffSize * GetRankSize(), device::CopyType::kD2H,
                                            output_device_tensor->stream_id());
  device_context->device_res_manager_->FreeMemory(output_device_tensor.get());
  device_context->device_res_manager_->FreeMemory(input_device_tensor.get());
}

void Process::ValidateCommGroupExecuteOrders(int step) {
  MS_LOG(INFO) << "The online verification of the execution sequence precision of the communication operator starts.";
  std::unique_ptr<char[]> output_host_buffer = std::make_unique<char[]>(kMaxAllGatherBuffSize * GetRankSize());
  AllGatherExecuteOrderHash(&output_host_buffer, step);

  std::map<std::string, std::map<uint64_t, size_t>> group_execute_order_hash;
  size_t offset_z = 0;

  for (size_t i = 0; i < GetRankSize(); ++i) {
    const char *buffer_ptr = output_host_buffer.get() + offset_z;
    const char *buffer_end = output_host_buffer.get() + offset_z + kMaxAllGatherBuffSize;

    while (buffer_ptr < buffer_end) {
      // Find the next comma or the end of the buffer
      const char *delimiter_pos = std::find(buffer_ptr, buffer_end, ',');
      size_t kvpair_length = delimiter_pos - buffer_ptr;

      if (kvpair_length == 0) break;

      // Find the colon separating the key and value
      const char *colon_pos = std::find(buffer_ptr, delimiter_pos, ':');
      if (colon_pos != delimiter_pos) {
        std::string key(buffer_ptr, colon_pos);
        uint64_t value = std::stoull(std::string(colon_pos + 1, delimiter_pos));
        group_execute_order_hash[key][value]++;
      }

      // Move the pointer to the next key-value pair
      buffer_ptr = delimiter_pos + 1;
    }

    offset_z += kMaxAllGatherBuffSize;
  }

  ValidateExecuteOrders(group_execute_order_hash);
}

void Process::ValidateExecuteOrders(const std::map<std::string, std::map<uint64_t, size_t>> &group_execute_order_hash) {
  for (const auto &[key, value_counts] : group_execute_order_hash) {
    if (value_counts.size() > 1) {
      MS_LOG(WARNING) << "Group: " << key << "has different order.";
    } else {
      if (!value_counts.empty()) {
        const auto &[value, count] = *value_counts.begin();

        size_t expected_count = 2;
        if (!key.empty() && key.front() != kSendReceive[0]) {
          size_t pos = key.rfind('_');
          if (pos != std::string::npos) {
            expected_count = std::stoull(key.substr(pos + 1));
          }
        }
        if (count != expected_count) {
          MS_LOG(WARNING) << "Group: " << key << " execute order appeared " << value << " value " << count
                          << " times but expected " << expected_count << " times";
        }
      } else {
        MS_LOG(WARNING) << "Group: " << key << " has no execute order.";
      }
    }
  }
  MS_LOG(WARNING) << "ValidateExecuteOrders pass, the communication execution order is correct.";
}

uint64_t Process::accumulate_hash(uint64_t current_hash, const std::string &str) {
  return std::accumulate(str.begin(), str.end(), current_hash,
                         [](uint64_t hash, char c) { return fnv1a_hash_update(hash, c); });
}

void Process::ProcessSendReceive(ProcessResult *result, const std::string &group, const KernelVariant &kernel,
                                 const std::string &primitive_str, const std::string &inputShape,
                                 const std::string &outputShape) {
  std::string rank_id = GetRankID();
  std::string other_rank;
  if (auto cnode_ptr = std::get_if<CNodePtr>(&kernel)) {
    other_rank = (primitive_str == "Send") ? GetRankByAttrName(*cnode_ptr, comm_rank_cache_[group], kAttrDestRank)
                                           : GetRankByAttrName(*cnode_ptr, comm_rank_cache_[group], kAttrSrcRank);
  } else if (auto pyboost_ptr = std::get_if<CommPyboostKernelPtr>(&kernel)) {
    other_rank = std::to_string((*pyboost_ptr)->rank);
  }
  std::string key =
    (rank_id < other_rank) ? kSendReceive + rank_id + "-" + other_rank : kSendReceive + other_rank + "-" + rank_id;

  std::string value_string = (primitive_str == "Send" || primitive_str == prim::kPrimDistCommIsend->name() ||
                              primitive_str == prim::kPrimInnerCommIsend->name())
                               ? kSendReceive + rank_id + other_rank + inputShape
                               : kSendReceive + other_rank + rank_id + outputShape;

  auto it = result->group_hashes.find(key);
  if (it == result->group_hashes.end()) {
    it = result->group_hashes.emplace(key, kFnvOffsetBasis).first;
  }

  uint64_t current_hash = it->second;
  current_hash = accumulate_hash(current_hash, value_string);
  it->second = current_hash;
}

void Process::ProcessNormalGroupHash(ProcessResult *result, const std::string &group, const std::string &primitive_str,
                                     const std::string &inputShape) {
  std::string group_key = group + "_" + std::to_string(comm_rank_cache_[group].size());
  auto it = result->group_hashes.find(group_key);
  if (it == result->group_hashes.end()) {
    it = result->group_hashes.emplace(group_key, kFnvOffsetBasis).first;
  }

  uint64_t current_hash = it->second;
  current_hash = accumulate_hash(current_hash, primitive_str);
  current_hash = accumulate_hash(current_hash, inputShape);
  it->second = current_hash;
}

void Process::FetchCommRanksCache(const std::string &group_name) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = comm_rank_cache_.find(group_name);
  if (it != comm_rank_cache_.end()) {
    return;
  }

  std::vector<uint32_t> comm_ranks;
  if (group_name == kCommGroupName) {
    comm_ranks.resize(GetRankSize());
    std::iota(comm_ranks.begin(), comm_ranks.end(), 0);
  } else {
    comm_ranks = distributed::collective::CollectiveManager::instance()->GetGroupRanks(group_name);
  }
  comm_rank_cache_[group_name] = comm_ranks;
}

void Process::StartCollectExecOrder() {
  auto &cache = KernelCache::GetInstance();
  cache.need_add = true;
}

void Process::StopCollectExecOrder() {
  auto &cache = KernelCache::GetInstance();
  cache.need_add = false;
  ProcessKernels();
  ValidateCommGroupExecuteOrders();
}
}  // namespace runtime
}  // namespace mindspore
