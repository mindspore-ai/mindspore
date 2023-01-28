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

#ifndef MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_EMBEDDING_CACHE_PREFETCH_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_EMBEDDING_CACHE_PREFETCH_ACTOR_H_

#include <map>
#include <set>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <random>

#include "runtime/graph_scheduler/actor/actor_common.h"
#include "ir/anf.h"
#include "backend/common/session/kernel_graph.h"
#include "distributed/cluster/cluster_context.h"
#include "distributed/rpc/tcp/tcp_client.h"
#include "distributed/rpc/tcp/tcp_server.h"
#include "utils/hash_map.h"
#include "include/common/random.h"
#include "distributed/embedding_cache/embedding_cache_utils.h"

// Note: After the code in ps/ps_cache are removed into runtime/addons/embedding_cache/,
// the follow include file and using declaration of ps will be removed.
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#include "ps/ps_context.h"
using mindspore::ps::PSContext;
using mindspore::ps::PsDataPrefetch;

namespace mindspore {
namespace runtime {
using kernel::Address;
using kernel::AddressPtr;
using kernel::AddressPtrList;

class DeviceEmbeddingOperation;
class Sender;
class Receiver;
using SenderPtr = std::shared_ptr<Sender>;
using ReceiverPtr = std::shared_ptr<Receiver>;
using SendRecvPair = std::pair<SenderPtr, ReceiverPtr>;
using SendRecvPairList = std::vector<SendRecvPair>;

using distributed::EmbeddingCacheStatisticsInfo;
using distributed::EmbeddingDeviceCache;
using distributed::EmbeddingHostCache;
using distributed::HashTableInfo;
using distributed::INVALID_INDEX_VALUE;
using distributed::INVALID_STEP_VALUE;

using distributed::cluster::ActorRouteTableProxy;
using distributed::cluster::ActorRouteTableProxyPtr;
using distributed::rpc::TCPClient;
using distributed::rpc::TCPServer;

using DataType = float;
using Generator = random::Philox;
using Distribution = random::NormalDistribution<double>;

// The EmbeddingCachePrefetchActor is used to cache large embedding table scenarios. The cache level is: Device
// Cache->Local Host Cache->Remote Cache. This Actor is used to perform Local and Device Cache hit analysis and cache
// prefetching (the feature weights corresponding to the ids of subsequent batches are assigned in advance Prefetching
// into the Device Cache, so that it is pipelined with the calculation on the Device side), cache prefetching may
// involve RPC communication with the Server side.
class EmbeddingCachePrefetchActor : public ActorBase {
 public:
  explicit EmbeddingCachePrefetchActor(device::DeviceContext *device_context)
      : ActorBase("EmbeddingCachePrefetchActor"), device_context_(device_context), cpu_device_context_(nullptr) {}

  ~EmbeddingCachePrefetchActor() override = default;

  // Initialize embedding cache prefetch actor.
  // 1. Build and Link rpc operators between local cache and remote cache.
  // 2. Build network connection of rpc operators.
  void Initialize();

  // Perform local cache hit analysis, prefetch the feature vector corresponding to the next batch into the cache.
  void Run();

  // Increase the global step of compute graph.
  void IncreaseGraphStep(const std::string &channel_name);

  // Sync latest embedding table to remote.
  void SyncEmbeddingTable();

  // Finalize embedding cache prefetch actor and push latest embedding from local cache to remote cache.
  void Finalize();

  // Wait the computed graph finish current step when there is not enough free memory space in the cache, in order to
  // delete the feature vector used by the current step from the cache.
  bool WaitGraphRun();

  // Reset EmbeddingHashMap for device and local host cache.
  bool ResetEmbeddingHashMap();

  // Insert weights into the local host embedding cache.
  bool InsertLocalHostCache(size_t embedding_size, size_t insert_indices_size, const int *insert_indices,
                            const float *insert_data, float *hash_table_addr);

  // Lookup embeddings from local host embedding cache.
  bool LookupLocalHostCache(size_t embedding_size, size_t indices_num, const float *hash_table_addr,
                            const int *indices_addr, float *output_addr);

 private:
  // Perform Local and Device Cache hit/miss analysis and prefetch cache for missing embeddings.
  bool PrefetchCache();

  // Increase the current global step of cache prefetching operation.
  bool IncreaseStep();

  // Update the current computed graph's step to real global step at the time when this actor starts to prefetch cache
  // for a batch ids.
  void set_current_graph_step() { graph_running_step_ = graph_step_; }

  // Push non-hotspot embeddings on local host cache to remote.
  bool PushCacheFromLocalHostToRemote(const HashTableInfo &hash_info);
  // Pull missing embeddings on local cache from remote.
  bool PullCacheFromRemoteToLocalHost(const HashTableInfo &hash_info);

  // Initialize local cache values using the random number generator.
  bool InitLocalCacheForNewIds(const HashTableInfo &hash_info);

  // Lookup embedding from Remote and get embeddings via RPC.
  bool PullEembeddingsFromRemote(int32_t param_key, const int *ids, size_t ids_num, std::vector<float> *outputs);
  // Push the local embedding cache that requires evict to the remote.
  bool PushEmbeddingsToRemote(int32_t param_key, const int *ids, size_t ids_num, const float *embeddings,
                              size_t embeddings_len);

  // Get the id range of each server's embedding table slice.
  void GetRemoteEmbeddingSliceBound();

  // In a multi-server scenario, the embeddings need to be segmented, and each server saves the embeddings of
  // different feature id ranges. Therefore, when the local side performs the push or pull embeddings operation, the
  // embeddings and ids need to be divided, and then communicate with the corresponding remote: Partition ids by
  // remote embedding slice bound and get unique ids.
  bool PartitionIds(const int *ids, size_t ids_num, std::vector<std::vector<int>> *slice_ids_list);
  // Partition ids end embeddings by remote embedding slice bound.
  bool PartitionIdsAndEmbeddings(const int *ids, size_t ids_num, const float *embeddings, size_t embeddings_len,
                                 std::vector<std::vector<int>> *slice_ids_list,
                                 std::vector<std::vector<float>> *slice_embeddings_list);

  // Send content to remote, such as ids or embeddings.
  // The parameter 'cache_operation' is cache operation name such as LookupEmbeddingCache and UpdateEmbeddingCache.
  bool SendToRemote(const std::string &cache_operation, int32_t param_key, size_t server_rank_id, size_t embedding_dim,
                    const void *keys, size_t keys_len, const void *values = nullptr, size_t values_len = 0,
                    bool finalize_remote = false, bool sync = true);
  // Wait response of remote and get return result.
  // The parameter 'cache_operation' is cache operation name such as LookupEmbeddingCache and UpdateEmbeddingCache.
  std::unique_ptr<std::vector<char>> ReceiveFromRemote(const std::string &cache_operation, int32_t param_key,
                                                       size_t server_rank_id) const;
  // Retrieve embeddings by input ids order.
  bool RetrieveEmbeddings(const int *ids, size_t ids_num, const std::vector<std::vector<int>> &slice_ids_list,
                          const std::vector<std::unique_ptr<std::vector<char>>> &slice_embeddings_list,
                          std::vector<float> *outputs) const;

  // Send finalize request to remote and finalize it.
  bool FinalizeRemote();

  // Sync latest local host embedding cache to remote.
  bool SyncHostEmbeddingTable();
  // Sync latest device embedding cache to remote.
  bool SyncDeviceEmbeddingTable();

  // The cache prefetch phase may involve RPC communication with the server, implemented through Sender and
  // Receiver.
  // Build rpc operators.
  void BuildRpcOperators();
  // Link rpc operators and build network connection.
  void LinkRpcOperators();

  // Get dataset channel name.
  std::string channel_name();
  // Set dataset channel name.
  void set_channel_name(const std::string channel_name);

  // When the device cache does not reach 100% hit, the cache needs to be updated, which involves cache insertion and
  // deletion. That is, push the non-hotspot embeddings on the local side to the remote, and pull the missing embeddings
  // on the local side from the remote.
  bool UpdateCache();

  // Do lookup embedding table operation.
  void LookupEmbeddingTable(size_t indices_num, size_t outer_dim_size, size_t first_dim_size, const float *input_addr,
                            const int *indices_addr, float *output_addr);

  // Wait data channel ready.
  void WaitDataChannelInit();

  // Wait initialize parameters on remote.
  // Prevents the subsequent prefetch cache from failing due to the long initialization time of the large parameter on
  // the remote side.
  void WaitInitParametersOnRemote();

  // Set current error information before finalizing actor.
  void SetErrorInfo(const std::string &error_info);

  // The operations for the embedding on the device.
  DeviceEmbeddingOperation *emb_ops_{nullptr};

  // Record sender and receiver pairs for different cache operation, server and parameter key.
  // key: cache operation(such as LookupEmbeddingCache and UpdateEmbeddingCache)
  // value: sender and receiver pairs for this kind of cache operation.
  mindspore::HashMap<std::string, std::vector<SendRecvPairList>> rpc_operators_;

  // The device interface.
  device::DeviceContext *device_context_;
  // The CPU device context used for allocating rpc message data.
  device::DeviceContext *cpu_device_context_;
  // The device stream used to async memcpy operators and launch device kernels, such as embedding cache look up and
  // update kernel.
  size_t stream_id_{0};

  // Full Embedding table row num, not less than the total number of feature ids.
  size_t vocab_size_{0};

  // Embedding cache size(row number of embedding cache) of local host cache.
  size_t local_host_cache_size_{0};

  // Statistics on the cache hit rate of the host and device and the information used to update cache.
  EmbeddingCacheStatisticsInfo statistics_info_;

  // Model parallelism is used between multiple workers, and local_embedding_slice_bounds_ records the feature range
  // corresponding to the embedding table slice of the process.
  std::pair<int, int> local_embedding_slice_bounds_;

  // Model parallelism is used between multiple workers, and local_device_cache_bounds_ records the local device cache
  // range corresponding to the embedding table slice of the process.
  std::pair<int, int> local_device_cache_bounds_;

  // In a multi-server scenario, the embeddings need to be segmented, and each server saves the embeddings of
  // different feature id ranges, remote_embedding_slice_bounds_ records the feature range of the embedding table
  // slice on each server.
  std::vector<std::pair<size_t, size_t>> remote_embedding_slice_bounds_;

  // Total server number of cluster.
  size_t server_num_{0};

  // The flag which indicates whether this actor is running to prefetch cache.
  std::atomic_bool running_{false};

  // The flag which indicates whether this actor is initialized.
  bool initialized_{false};
  // The flag which indicates whether this actor is finalized.
  bool finalized_{false};

  // Ensure that the Finalize function is multithreaded safe.
  std::mutex finalize_mutex_;

  // The flag which indicates whether finish sync embedding table.
  bool finish_sync_embedding_table_{false};
  std::mutex sync_embedding_table_mutex_;

  // The current global step of the computed graph.
  std::atomic_ulong graph_step_{0};
  // The computed graph's global step at the time when this actor starts to prefetch cache for a batch ids.
  size_t graph_running_step_{0};
  // The current global step of cache prefetching operation.
  size_t data_step_{0};

  // Dataset channel name, used in dataset switching scenarios.
  std::string channel_name_{""};
  // The mutex to access channel_name_.
  std::mutex channel_mutex_;

  // The flag indicates whether finish initializing parameters on remote..
  std::atomic_bool finish_init_parameters_on_remote_{false};

  // Data parser condition variable for prefetching cache, used to start and synchronize intermediate state for cache
  // prefetching.
  std::condition_variable data_parser_;
  // Data parser mutex for prefetching cache.
  std::mutex data_mutex_;

  // Whether device cache prefetching process needs to wait the computed graph finish current step when there is not
  // enough free memory space in the cache.
  bool device_cache_need_wait_graph_{false};
  // Whether local host cache prefetching process needs to wait the computed graph finish current step when there is not
  // enough free memory space in the cache.
  bool host_cache_need_wait_graph_{false};

  // Record latest error information user related.
  std::string error_info_{""};

  // The random number generator is used to initialize the embedding values when needed.
  std::unique_ptr<distributed::RandomGenerator<DataType, Generator, Distribution>> rnd_gen_;
};

// RpcOperator is used to do rpc with other processes in distributed execution.
// RpcOperator use inter process edge to identify paired rpc operators uniquely.
class RpcOperator {
 public:
  RpcOperator() : inter_process_edge_(""), route_table_proxy_(nullptr) {}
  virtual ~RpcOperator() = default;

  // Set the inter-process edge name for rpc operators.
  void set_inter_process_edge_name(const std::string &edge_name) { inter_process_edge_ = edge_name; }

  // Set the route table proxy for rpc operators.
  void set_actor_route_table_proxy(const ActorRouteTableProxyPtr &route_table_proxy) {
    route_table_proxy_ = route_table_proxy;
  }

 protected:
  // Unique edge name between rpc operator, format:
  // src role + src rank id -> dst role + dst rank id + embedding cache operation + parameter key.
  std::string inter_process_edge_;

  // Route table proxy for buildding network connection between nodes like workers and server.
  ActorRouteTableProxyPtr route_table_proxy_;
};

// Sender is used to send data to other process.
class Sender : public RpcOperator {
 public:
  explicit Sender(device::DeviceContext *cpu_device_context)
      : server_url_(""), client_(nullptr), cpu_device_context_(cpu_device_context) {}
  ~Sender() override;

  // Send buffer to peer.
  bool Send(const std::vector<ShapeVector> &shapes, const std::vector<TypeId> data_types,
            const AddressPtrList &data_list, bool finalize_remote = false, bool sync = true) const;

  // Set the receiver paired with the sender to get the 'from url' from the receiver.
  void set_receiver(const ReceiverPtr &receiver) { receiver_ = receiver; }

  // Lookup peer receiver's route and build network connection.
  bool ConnectServer();

 private:
  // Build the MessageBase include dynamic shape protobuf, which will be sent to peer receiver.
  // The message format is as below:
  // |--------22 bytes-------|-------sizeof(size_t)-------|-dynamic shape PB data size-| real data size |
  // |RPC_DYNAMIC_SHAPE_DATA | dynamic shape PB data size |---dynamic shape PB data----|---real data----|
  // The message.from (from url) must be set.
  std::unique_ptr<MessageBase> BuildRpcMessage(const std::vector<ShapeVector> &shapes,
                                               const std::vector<TypeId> data_types, const AddressPtrList &data_list,
                                               const std::string &from_url, const std::string &to_url,
                                               bool finalize_remote) const;

  // Free message after it's sent to remote.
  bool FreeMessage(void *data);

  // Calculate the dynamic shape message size.
  size_t CalDataSize(const std::vector<ShapeVector> &shapes, const std::vector<TypeId> data_types,
                     const AddressPtrList &data_list, bool finalize_remote) const;

  // The url of the peer receiver's tcp server.
  std::string server_url_;

  std::unique_ptr<TCPClient> client_;

  // The sender and the receiver are used in pairs. The information sent by the sender contains the url of the
  // corresponding receiver, so a reference to the receiver is maintained in the sender.
  ReceiverPtr receiver_;

  // The CPU device context used for allocating rpc message data.
  device::DeviceContext *cpu_device_context_;
};

// Receiver is used to receive data from other process.
class Receiver : public RpcOperator {
 public:
  explicit Receiver(device::DeviceContext *cpu_device_context)
      : ip_(""),
        port_(0),
        server_(nullptr),
        received_buffer_(nullptr),
        received_msg_(false),
        cpu_device_context_(cpu_device_context) {}
  ~Receiver() override;

  // Receive message from the peer sender, this interface is a synchronous interface and will wait for the message
  // until the timeout period is reached.
  std::unique_ptr<std::vector<char>> Receive();

  // Start receiver server and register this server address to route table in scheduler by proxy.
  bool StartServer();

  // Get the url of this receiver, format: ip:port.
  std::string get_url() const { return ip_ + ":" + std::to_string(port_); }

 private:
  // The message callback of the tcp server.
  MessageBase *HandleMessage(MessageBase *const msg);

  // Parse the dynamic shape protobuf message. The format is as below:
  // |--------22 bytes-------|-------sizeof(size_t)-------|-dynamic shape PB data size-| real data size |
  // |RPC_DYNAMIC_SHAPE_DATA | dynamic shape PB data size |---dynamic shape PB data----|---real data----|
  // The output parameter 'data' contains real data addr and size.
  bool ParseDynamicShapeData(const char *msg_body, size_t msg_len, std::pair<const void *, size_t> *data) const;

  // The callback set to rpc module to allocate message(Raw pointer).
  void *AllocateMessage(size_t size);

  // The network address of this receiver. It's generated automatically by rpc module.
  std::string ip_;
  uint32_t port_;

  std::unique_ptr<TCPServer> server_;

  // The buffer used save received content of message.
  std::unique_ptr<std::vector<char>> received_buffer_;

  // The flag indicates whether receive message successfully.
  std::atomic_bool received_msg_;

  // The interface 'Receive' is a synchronous, use condition variable to block thread and wait for the message.
  std::condition_variable received_msg_cv_;
  std::mutex received_msg_mtx_;

  // The CPU device context used for allocating rpc message data.
  device::DeviceContext *cpu_device_context_;
};

using EmbeddingCachePrefetchActorPtr = std::shared_ptr<EmbeddingCachePrefetchActor>;
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_ACTOR_EMBEDDING_CACHE_EMBEDDING_CACHE_PREFETCH_ACTOR_H_
