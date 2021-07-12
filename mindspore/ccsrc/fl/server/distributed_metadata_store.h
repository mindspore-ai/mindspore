/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_DISTRIBUTED_META_STORE_H_
#define MINDSPORE_CCSRC_FL_SERVER_DISTRIBUTED_META_STORE_H_

#include <string>
#include <memory>
#include <unordered_map>
#include "proto/ps.pb.h"
#include "fl/server/common.h"
#include "ps/core/server_node.h"
#include "ps/core/communicator/tcp_communicator.h"
#include "fl/server/consistent_hash_ring.h"

namespace mindspore {
namespace fl {
namespace server {
constexpr auto kModuleDistributedMetadataStore = "DistributedMetadataStore";
// This class is used for distributed metadata storage using consistent hash. All metadata is distributedly
// stored in all servers. Caller doesn't need to know which server stores the metadata. It only needs to know what kind
// of operations should be done to the metadata.

// The metadata stored in the server is in protobuffer format because it's easy for serializing and communicating. The
// type of the protobuffer struct is decided by the caller using protobuffer's API.
class DistributedMetadataStore {
 public:
  static DistributedMetadataStore &GetInstance() {
    static DistributedMetadataStore instance;
    return instance;
  }

  // Initialize metadata storage with the server node because communication is needed.
  void Initialize(const std::shared_ptr<ps::core::ServerNode> &server_node);

  // Register callbacks for the server to handle update/get metadata messages from other servers.
  void RegisterMessageCallback(const std::shared_ptr<ps::core::TcpCommunicator> &communicator);

  // Register metadata for the name with the initial value. This method should be only called once for each name.
  void RegisterMetadata(const std::string &name, const PBMetadata &meta);

  // Reset the metadata value for the name.
  void ResetMetadata(const std::string &name);

  // Update the metadata for the name. Parameter 'reason' is the reason why updating meta data failed.
  bool UpdateMetadata(const std::string &name, const PBMetadata &meta, std::string *reason = nullptr);

  // Get the metadata for the name.
  PBMetadata GetMetadata(const std::string &name);

  // Reinitialize the consistency hash ring and clear metadata after scaling operations are done.
  bool ReInitForScaling();

 private:
  DistributedMetadataStore()
      : server_node_(nullptr),
        communicator_(nullptr),
        local_rank_(0),
        server_num_(0),
        router_(nullptr),
        metadata_({}) {}
  ~DistributedMetadataStore() = default;
  DistributedMetadataStore(const DistributedMetadataStore &) = delete;
  DistributedMetadataStore &operator=(const DistributedMetadataStore &) = delete;

  // Initialize the consistent hash ring for distributed storage.
  void InitHashRing();

  // Callback for updating metadata request sent to the server.
  void HandleUpdateMetadataRequest(const std::shared_ptr<ps::core::MessageHandler> &message);

  // Callback for getting metadata request sent to the server.
  void HandleGetMetadataRequest(const std::shared_ptr<ps::core::MessageHandler> &message);

  // Do updating metadata in the server where the metadata for the name is stored.
  bool DoUpdateMetadata(const std::string &name, const PBMetadata &meta);

  // Members for the communication between servers.
  std::shared_ptr<ps::core::ServerNode> server_node_;
  std::shared_ptr<ps::core::TcpCommunicator> communicator_;
  uint32_t local_rank_;
  uint32_t server_num_;

  // Consistent hash ring. This is used for DistributedMetadataStore to find which server node the meta data is stored.
  std::shared_ptr<ConsistentHashRing> router_;

  // We store metadata which is serialized by ProtoBuffer so that data storage and data transmission API is easy to use.
  // Key: data name.
  // Value: ProtoBuffer Struct.
  std::unordered_map<std::string, PBMetadata> metadata_;

  // Because the metadata is read/written conccurently, we must ensure the operations are threadsafe.
  std::unordered_map<std::string, std::mutex> mutex_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_DISTRIBUTED_META_STORE_H_
