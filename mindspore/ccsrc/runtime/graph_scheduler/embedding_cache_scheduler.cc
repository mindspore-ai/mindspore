/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/embedding_cache_scheduler.h"

#include <string>
#include <memory>
#include <set>
#include <functional>
#include "runtime/graph_scheduler/actor/embedding_cache/embedding_cache_prefetch_actor.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "include/backend/distributed/embedding_cache/embedding_cache_utils.h"
#include "utils/ms_context.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace runtime {
using session::KernelGraph;
namespace {
bool CheckEnableEmbeddingCache() {
  return ps::PSContext::instance()->cache_enable() && distributed::cluster::ClusterContext::instance()->initialized() &&
         ps::PSContext::instance()->is_worker();
}

bool CheckEmbeddingCacheServer() {
  return ps::PSContext::instance()->cache_enable() && ps::PSContext::instance()->is_server();
}

// Whether device address exist.
bool NodeDeviceAddressExist(const DeviceContext *device_context, const AnfNodePtr &kernel, size_t index) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(device_context);
  if (AnfAlgo::OutputAddrExist(kernel, index)) {
    const auto &address = AnfAlgo::GetOutputAddr(kernel, index, false);
    MS_EXCEPTION_IF_NULL(address);
    return address->GetDeviceType() == device_context->GetDeviceType();
  }
  return false;
}

// Finalize ps cache module before throw an exception.
void FinalizeEmbeddingCachePrefetch(const std::string &exception) {
  MS_LOG(INFO) << "Begin finalize the EmbeddingCacheScheduler.";
  EmbeddingCacheScheduler::GetInstance().Finalize(false);
  MS_LOG(INFO) << "End finalize the EmbeddingCacheScheduler.";
  MS_LOG(EXCEPTION) << exception;
}

void GetFirstEmbeddingCacheTableInfo(const KernelGraph &graph, AnfNodePtr *const first_cache_input_index,
                                     size_t *const first_cache_size) {
  MS_EXCEPTION_IF_NULL(first_cache_input_index);
  MS_EXCEPTION_IF_NULL(first_cache_size);
  for (const auto &kernel : graph.execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel);
    if (kernel_name != kGatherOpName && kernel_name != kSparseGatherV2OpName && kernel_name != kGatherV2OpName &&
        kernel_name != kGatherV2DOpName) {
      continue;
    }
    auto input_param = common::AnfAlgo::GetPrevNodeOutput(kernel, 0, true);
    auto input_index = common::AnfAlgo::GetPrevNodeOutput(kernel, 1, true);
    MS_EXCEPTION_IF_NULL(input_param.first);
    MS_EXCEPTION_IF_NULL(input_index.first);
    auto param_name = input_param.first->fullname_with_scope();
    if (!embedding_cache_table_manager.IsEmbeddingCacheTable(param_name)) {
      continue;
    }
    auto size = embedding_cache_table_manager.QueryHashTableSize(param_name);
    while (input_index.first->isa<CNode>() &&
           ((common::AnfAlgo::GetCNodeName(input_index.first) == kCastOpName) ||
            (common::AnfAlgo::GetCNodeName(input_index.first) == kTensorMoveOpName))) {
      input_index = common::AnfAlgo::GetPrevNodeOutput(input_index.first, 0, true);
      MS_EXCEPTION_IF_NULL(input_index.first);
    }
    auto cnode = common::AnfAlgo::IsGraphKernel(input_index.first)
                   ? common::AnfAlgo::GetOutputOfGraphkernel(input_index)
                   : input_index.first;
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->isa<CNode>()) {
      FinalizeEmbeddingCachePrefetch("The EmbeddingLookup whose input index should be a CNode but got " +
                                     cnode->fullname_with_scope());
    }
    auto input_index_node_name = common::AnfAlgo::GetCNodeName(cnode);
    if (input_index_node_name != kGetNextOpName) {
      bool full_batch = parallel::ParallelContext::GetInstance()->full_batch();
      if ((!full_batch && (input_index_node_name != kUniqueOpName)) ||
          (full_batch && (input_index_node_name != kMinimumOpName))) {
        MS_LOG(ERROR) << "The input index of the EmbeddingLookup(" << kernel->fullname_with_scope()
                      << ") cache is from " << cnode->fullname_with_scope();
        FinalizeEmbeddingCachePrefetch(
          "The EmbeddingLookup whose input index isn't from dataset doesn't support cache in parameter server training "
          "mode.");
      }
    }
    *first_cache_input_index = cnode;
    *first_cache_size = size;
    MS_LOG(INFO) << "The input index of the first EmbeddingLookup cache is from " << cnode->fullname_with_scope()
                 << ", the cache size is " << size;
    return;
  }
}

void CheckSparseModeForEmbeddingCache(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto pre_node = common::AnfAlgo::GetPrevNodeOutput(node, 1, true);
  MS_EXCEPTION_IF_NULL(pre_node.first);
  while (pre_node.first->isa<CNode>() && (common::AnfAlgo::GetCNodeName(pre_node.first) != kUniqueOpName)) {
    pre_node = common::AnfAlgo::GetPrevNodeOutput(pre_node.first, 0, true);
    MS_EXCEPTION_IF_NULL(pre_node.first);
  }
  if (!(pre_node.first->isa<CNode>()) || (common::AnfAlgo::GetCNodeName(pre_node.first) != kUniqueOpName)) {
    FinalizeEmbeddingCachePrefetch(
      "The input_indices of kernel[SparseGatherV2] must be unique in parameter server cache mode");
  }

  pre_node = common::AnfAlgo::GetPrevNodeOutput(pre_node.first, 0, true);
  MS_EXCEPTION_IF_NULL(pre_node.first);
  while (pre_node.first->isa<CNode>() && ((common::AnfAlgo::GetCNodeName(pre_node.first) == kCastOpName) ||
                                          (common::AnfAlgo::GetCNodeName(pre_node.first) == kTensorMoveOpName))) {
    pre_node = common::AnfAlgo::GetPrevNodeOutput(pre_node.first, 0, true);
    MS_EXCEPTION_IF_NULL(pre_node.first);
  }
  if (!(pre_node.first->isa<CNode>()) || (common::AnfAlgo::GetCNodeName(pre_node.first) != kGetNextOpName)) {
    FinalizeEmbeddingCachePrefetch(
      "The input indices of kernel[Unique] must be produced from dataset directly and the indices value can not be "
      "changed before delivering to kernel[Unique] in parameter server cache mode.");
  }
}

void CheckGraphValidForEmbeddingCache(const KernelGraph &graph) {
  AnfNodePtr first_cache_input_index = nullptr;
  size_t first_cache_size = 0;
  GetFirstEmbeddingCacheTableInfo(graph, &first_cache_input_index, &first_cache_size);
  MS_EXCEPTION_IF_NULL(first_cache_input_index);
  for (const auto &kernel : graph.execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel);
    const std::set<std::string> kNeedCacheNodes = {kGatherOpName, kSparseGatherV2OpName, kGatherV2OpName,
                                                   kGatherV2DOpName};
    if (kNeedCacheNodes.count(kernel_name) == 0) {
      continue;
    }
    auto input_param = common::AnfAlgo::GetPrevNodeOutput(kernel, 0, true);
    auto input_index = common::AnfAlgo::GetPrevNodeOutput(kernel, 1, true);
    MS_EXCEPTION_IF_NULL(input_param.first);
    MS_EXCEPTION_IF_NULL(input_index.first);
    if (!input_param.first->isa<Parameter>()) {
      continue;
    }
    auto param_name = input_param.first->fullname_with_scope();
    // In ascend, change kSparseGatherV2OpName to kGatherV2OpName & set attr sparse: true
    bool is_sparse_gather = false;
    if (kernel_name == kGatherV2OpName && common::AnfAlgo::HasNodeAttr(kAttrIsSparse, kernel)) {
      is_sparse_gather = common::AnfAlgo::GetNodeAttr<bool>(kernel, kAttrIsSparse);
    }
    if (embedding_cache_table_manager.IsEmbeddingCacheTable(param_name) &&
        (kernel_name == kSparseGatherV2OpName || is_sparse_gather)) {
      CheckSparseModeForEmbeddingCache(kernel);
    }
    while (input_index.first->isa<CNode>() &&
           ((common::AnfAlgo::GetCNodeName(input_index.first) == kCastOpName) ||
            (common::AnfAlgo::GetCNodeName(input_index.first) == kTensorMoveOpName))) {
      input_index = common::AnfAlgo::GetPrevNodeOutput(input_index.first, 0, true);
      MS_EXCEPTION_IF_NULL(input_index.first);
    }
    auto cnode = common::AnfAlgo::IsGraphKernel(input_index.first)
                   ? common::AnfAlgo::GetOutputOfGraphkernel(input_index)
                   : input_index.first;
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode == first_cache_input_index) {
      if (!embedding_cache_table_manager.IsEmbeddingCacheTable(param_name)) {
        MS_LOG(ERROR) << "The EmbeddingLookup(" << kernel->fullname_with_scope() << ") doesn't enable cache.";
        FinalizeEmbeddingCachePrefetch(
          "All the embeddingLookups whose input indices are from dataset must enable cache at the same time when one "
          "of them enables cache in parameter server training mode.");
      }
      auto size = embedding_cache_table_manager.QueryHashTableSize(param_name);
      if (size != first_cache_size) {
        MS_LOG(ERROR) << "The cache size(" << size << ") of EmbeddingLookup(" << kernel->fullname_with_scope()
                      << ") is not the same as other EmbeddingLookup cache size(" << first_cache_size << ").";
        FinalizeEmbeddingCachePrefetch(
          "The cache sizes of embeddingLookups are not the same in parameter server training mode.");
      }
    } else if (embedding_cache_table_manager.IsEmbeddingCacheTable(param_name)) {
      MS_LOG(ERROR) << "The input index of the EmbeddingLookup(" << kernel->fullname_with_scope() << ") cache is from "
                    << cnode->fullname_with_scope();
      FinalizeEmbeddingCachePrefetch(
        "The EmbeddingLookup whose input index isn't from dataset doesn't support cache in parameter server training "
        "mode.");
    } else if (cnode->isa<CNode>() && (common::AnfAlgo::GetCNodeName(cnode) == kGetNextOpName)) {
      MS_LOG(ERROR) << "The EmbeddingLookup kernel(" << kernel->fullname_with_scope() << ") doesn't enable cache.";
      FinalizeEmbeddingCachePrefetch(
        "All EmbeddingLookup kernels whose input indices are from dataset must enable cache at the same time.");
    }
  }
}
}  // namespace

EmbeddingCacheScheduler &EmbeddingCacheScheduler::GetInstance() {
  static EmbeddingCacheScheduler instance{};
  if (!instance.initialized_) {
    instance.Initialize();
  }
  return instance;
}

void EmbeddingCacheScheduler::Initialize() {
  if (!CheckEnableEmbeddingCache()) {
    return;
  }
  if (initialized_) {
    return;
  }

  // Get or Create device context.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  DeviceContext *device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name, device_id});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  // Create and initialize EmbeddingCachePrefetchActor.
  embedding_cache_prefetch_actor_ = std::make_shared<EmbeddingCachePrefetchActor>(device_context);
  MS_EXCEPTION_IF_NULL(embedding_cache_prefetch_actor_);

  initialized_ = true;
}

void EmbeddingCacheScheduler::ParseBatchIdsNum(const KernelGraphPtr &graph) {
  if (parsed_batch_ids_num_) {
    return;
  }

  // 1. Find InitDataSetQueue kernel.
  MS_EXCEPTION_IF_NULL(graph);
  const auto &kernels = graph->execution_order();
  auto iter = find_if(kernels.begin(), kernels.end(), [](const CNodePtr &kernel) {
    MS_EXCEPTION_IF_NULL(kernel);
    return common::AnfAlgo::GetCNodeName(kernel) == kInitDatasetQueueOpName;
  });
  if (iter == kernels.end()) {
    MS_LOG(EXCEPTION) << "Can not find InitDataSetQueue kernel";
  }

  const auto &kernel = *iter;
  MS_EXCEPTION_IF_NULL(kernel);

  // 2. Get shape of InitDataSetQueue kernel.
  std::vector<std::vector<int64_t>> shapes;
  if (common::AnfAlgo::IsDynamicShape(kernel)) {
    shapes = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel, "max_shapes");
  } else {
    shapes = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel, "shapes");
  }
  auto types = common::AnfAlgo::GetNodeAttr<std::vector<TypePtr>>(kernel, "types");
  if (shapes.size() != types.size() || shapes.size() == 0 || types.size() == 0) {
    MS_LOG(EXCEPTION) << "Invalid shapes of op[InitDataSetQueue]: shapes size " << shapes.size() << ", types size "
                      << types;
  }

  const TypePtr &id_type = types.front();
  MS_EXCEPTION_IF_NULL(id_type);
  if (id_type->type_id() != kInt32->type_id() && id_type->type_id() != kInt->type_id()) {
    MS_LOG(EXCEPTION) << "Embedding cache mode need input ids with data type[" << kInt32->ToString() << " or "
                      << kInt->ToString() << "], but got[" << id_type->ToString() << "]";
  }

  // 3. Get batch ids num(not batch size).
  const auto &shape = shapes[0];
  auto batch_ids_num = LongToSize(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()));
  embedding_cache_table_manager.Initialize();
  embedding_cache_table_manager.set_batch_ids_num(batch_ids_num);

  parsed_batch_ids_num_ = true;
}

void EmbeddingCacheScheduler::AllocMemForEmbeddingCacheTable(const DeviceContext *device_context,
                                                             const KernelGraphPtr &graph) {
  if (allocated_embed_cache_mem_) {
    return;
  }

  embedding_cache_table_manager.AllocMemForEmbedding(device_context);
  allocated_embed_cache_mem_ = true;
}

void EmbeddingCacheScheduler::SetEmbedCachedParamAddress(const DeviceContext *device_context,
                                                         const KernelGraphPtr &graph) {
  if (!CheckEnableEmbeddingCache()) {
    return;
  }

  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  MS_EXCEPTION_IF_NULL(graph);

  // 1. Get batch ids number before allocate device memory for embedding cache table.
  ParseBatchIdsNum(graph);

  const std::vector<AnfNodePtr> &input_nodes = graph->input_nodes();
  bool exist_embedding_cache_table = std::any_of(input_nodes.begin(), input_nodes.end(), [](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    return embedding_cache_table_manager.IsEmbeddingCacheTable(node->fullname_with_scope());
  });
  if (!exist_embedding_cache_table) {
    return;
  }

  // Graph valid check.
  CheckGraphValidForEmbeddingCache(*graph);

  // 2. Set parameter device address to embedding cache table.
  for (const auto &node : input_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    const std::string &param_name = node->fullname_with_scope();
    if (!embedding_cache_table_manager.IsEmbeddingCacheTable(param_name)) {
      continue;
    }

    if (node->isa<Parameter>() && !NodeDeviceAddressExist(device_context, node, 0)) {
      MS_LOG(EXCEPTION) << "Not found device address for parameter: " << node->fullname_with_scope();
    }

    if (embedding_cache_table_manager.QueryEmbeddingDeviceAddress(param_name) == nullptr) {
      embedding_cache_table_manager.SetEmbeddingDeviceAddress(param_name, AnfAlgo::GetMutableOutputAddr(node, 0).get());
    }
  }

  // 3. Allocate device memory for embedding cache table.
  AllocMemForEmbeddingCacheTable(device_context, graph);
}

void EmbeddingCacheScheduler::SetDataSetChannel(const std::string &actor_id,
                                                const std::vector<KernelGraphPtr> &graphs) {
  if (!CheckEnableEmbeddingCache()) {
    return;
  }

  for (const auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    for (const auto &kernel_node : graph->execution_order()) {
      if (common::AnfAlgo::GetCNodeName(kernel_node) != kGetNextOpName) {
        continue;
      }

      if (!common::AnfAlgo::HasNodeAttr("shared_name", kernel_node)) {
        MS_LOG(EXCEPTION) << "Can not find attr[shared_name] of GetNext";
      }
      (void)data_prepare_aid_to_data_channel_.emplace(
        actor_id, common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "shared_name"));
      break;
    }
  }
}

void EmbeddingCacheScheduler::InitEmbeddingStorage(const std::vector<AnfNodePtr> &parameters) {
  if (!CheckEmbeddingCacheServer()) {
    return;
  }

  for (size_t i = 0; i < parameters.size(); i++) {
    MS_EXCEPTION_IF_NULL(parameters[i]);
    if (!parameters[i]->isa<Parameter>()) {
      MS_LOG(EXCEPTION) << "The node with name: " << parameters[i]->fullname_with_scope() << "is not a Parameter.";
    }

    ParameterPtr param = parameters[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    auto param_info = param->param_info();
    // Check whether enable embedding storage for the parameter.
    bool enable_embedding_storage =
      param_info && param_info->key() != -1 && param_info->cache_enable() && param_info->use_persistent_storage();
    if (!enable_embedding_storage) {
      continue;
    }

    auto embed_storage = embedding_storage_manager.Get(param_info->key());
    MS_EXCEPTION_IF_NULL(embed_storage);
    std::vector<DeviceTensorPtr> device_tensors = DeviceTensorStore::GetInstance().Fetch(param.get());
    if (device_tensors.size() != 1) {
      MS_LOG(EXCEPTION)
        << "The device tensor size for embedding table which enables embedding storage should be 1, but got:"
        << device_tensors.size();
    }

    // Initialize embedding storage instance.
    embed_storage->Initialize(device_tensors.front().get());
  }
}

void EmbeddingCacheScheduler::Schedule() {
  if (!initialized_ || scheduled_) {
    return;
  }

  // 1. Initialize embedding cache prefetch actor and build network connection inter process.
  MS_EXCEPTION_IF_NULL(embedding_cache_prefetch_actor_);
  embedding_cache_prefetch_actor_->Initialize();

  // 2. Spawn embedding cache prefetch actor.
  auto actor_manager = ActorMgr::GetActorMgrRef();
  // Bind single thread to execute embedding cache prefetch actor.
  (void)actor_manager->Spawn(embedding_cache_prefetch_actor_, false);

  // 3. Run embedding cache prefetch actor.
  ActorDispatcher::Send(embedding_cache_prefetch_actor_->GetAID(), &EmbeddingCachePrefetchActor::Run);

  scheduled_ = true;
}

void EmbeddingCacheScheduler::IncreaseGraphStep(const std::string &actor_id) const {
  if (!CheckEnableEmbeddingCache()) {
    return;
  }

  auto iter = data_prepare_aid_to_data_channel_.find(actor_id);
  if (iter != data_prepare_aid_to_data_channel_.end()) {
    MS_EXCEPTION_IF_NULL(embedding_cache_prefetch_actor_);
    embedding_cache_prefetch_actor_->IncreaseGraphStep(iter->second);
  }
}

void EmbeddingCacheScheduler::SyncEmbeddingTable() const {
  MS_EXCEPTION_IF_NULL(embedding_cache_prefetch_actor_);
  embedding_cache_prefetch_actor_->SyncEmbeddingTable();
}

void EmbeddingCacheScheduler::Finalize(bool sync_embedding_table) {
  std::lock_guard<std::mutex> lock(finalize_mutex_);
  if (!initialized_ || finalized_) {
    return;
  }

  if (sync_embedding_table) {
    SyncEmbeddingTable();
  }

  MS_EXCEPTION_IF_NULL(embedding_cache_prefetch_actor_);
  // Stop the embedding cache prefetch_actor.
  bool finalize_remote = sync_embedding_table;
  embedding_cache_prefetch_actor_->Finalize(finalize_remote);

  embedding_cache_table_manager.Finalize();

  embedding_storage_manager.Clear();

  initialized_ = false;
  finalized_ = true;
}
}  // namespace runtime
}  // namespace mindspore
