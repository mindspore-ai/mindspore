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

#include "fl/server/distributed_metadata_store.h"
#include <memory>
#include <string>
#include <vector>

namespace mindspore {
namespace fl {
namespace server {
void DistributedMetadataStore::Initialize(const std::shared_ptr<ps::core::ServerNode> &server_node) {
  MS_EXCEPTION_IF_NULL(server_node);
  server_node_ = server_node;
  local_rank_ = server_node_->rank_id();
  server_num_ = ps::PSContext::instance()->initial_server_num();
  InitHashRing();
  return;
}

void DistributedMetadataStore::RegisterMessageCallback(const std::shared_ptr<ps::core::TcpCommunicator> &communicator) {
  MS_EXCEPTION_IF_NULL(communicator);
  communicator_ = communicator;
  communicator_->RegisterMsgCallBack(
    "updateMetadata", std::bind(&DistributedMetadataStore::HandleUpdateMetadataRequest, this, std::placeholders::_1));
  communicator_->RegisterMsgCallBack(
    "getMetadata", std::bind(&DistributedMetadataStore::HandleGetMetadataRequest, this, std::placeholders::_1));
  communicator_->RegisterMsgCallBack(
    "getOneDeviceMeta",
    std::bind(&DistributedMetadataStore::HandleGetOneDeviceMetaRequest, this, std::placeholders::_1));
  return;
}

void DistributedMetadataStore::RegisterMetadata(const std::string &name, const PBMetadata &meta) {
  if (router_ == nullptr) {
    MS_LOG(WARNING) << "The consistent hash ring is not initialized yet.";
    return;
  }

  uint32_t stored_rank = router_->Find(name);
  if (local_rank_ == stored_rank) {
    if (metadata_.count(name) != 0) {
      MS_LOG(WARNING) << "The metadata for " << name << " is already registered.";
      return;
    }

    MS_LOG(INFO) << "Rank " << local_rank_ << " register storage for metadata " << name;
    metadata_[name] = meta;
    mutex_[name];
  }
  return;
}

void DistributedMetadataStore::ResetMetadata(const std::string &name) {
  if (router_ == nullptr) {
    MS_LOG(WARNING) << "The consistent hash ring is not initialized yet.";
    return;
  }

  uint32_t stored_rank = router_->Find(name);
  if (local_rank_ == stored_rank) {
    if (metadata_.count(name) == 0) {
      MS_LOG(WARNING) << "The metadata for " << name << " is not registered.";
      return;
    }

    MS_LOG(INFO) << "Rank " << local_rank_ << " reset metadata for " << name;
    std::unique_lock<std::mutex> lock(mutex_[name]);
    PBMetadata empty_meta;
    metadata_[name] = empty_meta;
  }
  return;
}

bool DistributedMetadataStore::UpdateMetadata(const std::string &name, const PBMetadata &meta, std::string *reason) {
  if (router_ == nullptr) {
    MS_LOG(WARNING) << "The consistent hash ring is not initialized yet.";
    return false;
  }

  uint32_t stored_rank = router_->Find(name);
  MS_LOG(DEBUG) << "Rank " << local_rank_ << " update value for " << name << " which is stored in rank " << stored_rank;
  if (local_rank_ == stored_rank) {
    if (!DoUpdateMetadata(name, meta)) {
      MS_LOG(WARNING) << "Updating meta data failed.";
      return false;
    }
  } else {
    PBMetadataWithName metadata_with_name;
    metadata_with_name.set_name(name);
    *metadata_with_name.mutable_metadata() = meta;
    std::shared_ptr<std::vector<unsigned char>> update_meta_rsp_msg = nullptr;
    if (!communicator_->SendPbRequest(metadata_with_name, stored_rank, ps::core::TcpUserCommand::kUpdateMetadata,
                                      &update_meta_rsp_msg)) {
      MS_LOG(WARNING) << "Sending updating metadata message to server " << stored_rank << " failed.";
      if (reason != nullptr) {
        *reason = kNetworkError;
      }
      return false;
    }

    MS_ERROR_IF_NULL_W_RET_VAL(update_meta_rsp_msg, false);
    std::string update_meta_rsp =
      std::string(reinterpret_cast<char *>(update_meta_rsp_msg->data()), update_meta_rsp_msg->size());
    if (update_meta_rsp != kSuccess) {
      MS_LOG(WARNING) << "Updating metadata in server " << stored_rank << " failed. " << update_meta_rsp;
      return false;
    }
  }
  return true;
}

PBMetadata DistributedMetadataStore::GetMetadata(const std::string &name) {
  if (router_ == nullptr) {
    MS_LOG(WARNING) << "The consistent hash ring is not initialized yet.";
    return {};
  }
  uint32_t stored_rank = router_->Find(name);
  MS_LOG(DEBUG) << "Rank " << local_rank_ << " get metadata for " << name << " which is stored in rank " << stored_rank;
  if (local_rank_ == stored_rank) {
    std::unique_lock<std::mutex> lock(mutex_[name]);
    return metadata_[name];
  } else {
    GetMetadataRequest get_metadata_req;
    get_metadata_req.set_name(name);
    PBMetadata get_metadata_rsp;

    std::shared_ptr<std::vector<unsigned char>> get_meta_rsp_msg = nullptr;
    if (!communicator_->SendPbRequest(get_metadata_req, stored_rank, ps::core::TcpUserCommand::kGetMetadata,
                                      &get_meta_rsp_msg)) {
      MS_LOG(WARNING) << "Sending getting metadata message to server " << stored_rank << " failed.";
      return get_metadata_rsp;
    }

    MS_ERROR_IF_NULL_W_RET_VAL(get_meta_rsp_msg, get_metadata_rsp);
    (void)get_metadata_rsp.ParseFromArray(get_meta_rsp_msg->data(), SizeToInt(get_meta_rsp_msg->size()));
    return get_metadata_rsp;
  }
}

bool DistributedMetadataStore::GetOneDeviceMeta(const std::string &fl_id, DeviceMeta *device_meta) {
  if (router_ == nullptr) {
    MS_LOG(WARNING) << "The consistent hash ring is not initialized yet.";
    return false;
  }
  const auto &name = kCtxDeviceMetas;
  uint32_t stored_rank = router_->Find(name);
  MS_LOG(DEBUG) << "Rank " << local_rank_ << " get metadata for " << name << " which is stored in rank " << stored_rank;
  if (local_rank_ == stored_rank) {
    return DoGetOneDeviceMeta(fl_id, device_meta);
  } else {
    GetOneDeviceMetaRequest get_metadata_req;
    get_metadata_req.set_fl_id(fl_id);

    std::shared_ptr<std::vector<unsigned char>> get_meta_rsp_msg = nullptr;
    if (!communicator_->SendPbRequest(get_metadata_req, stored_rank, ps::core::TcpUserCommand::kGetOneDeviceMeta,
                                      &get_meta_rsp_msg)) {
      MS_LOG(WARNING) << "Sending getting one client metadata message to server " << stored_rank << " failed.";
      return false;
    }
    MS_ERROR_IF_NULL_W_RET_VAL(get_meta_rsp_msg, false);
    GetOneDeviceMetaResponse get_metadata_rsp;
    auto ret = get_metadata_rsp.ParseFromArray(get_meta_rsp_msg->data(), SizeToInt(get_meta_rsp_msg->size()));
    if (!ret) {
      MS_LOG(WARNING) << "Parse response of getting one client metadata message to server " << stored_rank
                      << " failed.";
      return false;
    }
    if (!get_metadata_rsp.found()) {
      return false;
    }
    *device_meta = get_metadata_rsp.device_meta();
    return true;
  }
}

bool DistributedMetadataStore::ReInitForScaling() {
  // If DistributedMetadataStore is not initialized yet but the scaling event is triggered, do not throw exception.
  if (server_node_ == nullptr) {
    return true;
  }

  MS_LOG(INFO) << "Cluster scaling completed. Reinitialize for distributed metadata store.";
  local_rank_ = server_node_->rank_id();
  server_num_ = server_node_->server_num();
  MS_LOG(INFO) << "After scheduler scaling, this server's rank is " << local_rank_ << ", server number is "
               << server_num_;
  InitHashRing();

  // Clear old metadata.
  metadata_.clear();
  return true;
}

void DistributedMetadataStore::InitHashRing() {
  router_ = std::make_shared<ConsistentHashRing>(kDefaultVirtualNodeNum);
  MS_EXCEPTION_IF_NULL(router_);
  for (uint32_t i = 0; i < server_num_; i++) {
    bool ret = router_->Insert(i);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Add node " << i << " to router of meta storage failed.";
      return;
    }
  }
  return;
}

void DistributedMetadataStore::HandleUpdateMetadataRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  PBMetadataWithName meta_with_name;
  (void)meta_with_name.ParseFromArray(message->data(), SizeToInt(message->len()));
  const std::string &name = meta_with_name.name();
  MS_LOG(DEBUG) << "Update metadata for " << name;

  std::string update_meta_rsp_msg;
  if (!DoUpdateMetadata(name, meta_with_name.metadata())) {
    update_meta_rsp_msg = "Updating meta data failed.";
    MS_LOG(WARNING) << update_meta_rsp_msg;
  } else {
    update_meta_rsp_msg = "Success";
  }
  if (!communicator_->SendResponse(update_meta_rsp_msg.data(), update_meta_rsp_msg.size(), message)) {
    MS_LOG(WARNING) << "Sending response failed.";
    return;
  }
  return;
}

void DistributedMetadataStore::HandleGetMetadataRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  GetMetadataRequest get_metadata_req;
  (void)get_metadata_req.ParseFromArray(message->data(), SizeToInt(message->len()));
  const std::string &name = get_metadata_req.name();
  MS_LOG(DEBUG) << "Getting metadata for " << name;

  std::unique_lock<std::mutex> lock(mutex_[name]);
  if (metadata_.count(name) == 0) {
    MS_LOG(WARNING) << "The metadata of " << name << " is not registered.";
    return;
  }
  PBMetadata stored_meta = metadata_[name];
  std::string getting_meta_rsp_msg = stored_meta.SerializeAsString();
  if (!communicator_->SendResponse(getting_meta_rsp_msg.data(), getting_meta_rsp_msg.size(), message)) {
    MS_LOG(WARNING) << "Sending response failed.";
    return;
  }
  return;
}

void DistributedMetadataStore::HandleGetOneDeviceMetaRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  GetOneDeviceMetaRequest get_one_metadata_req;
  GetOneDeviceMetaResponse response;
  auto ret = get_one_metadata_req.ParseFromArray(message->data(), SizeToInt(message->len()));
  if (!ret) {
    MS_LOG(WARNING) << "Parse GetOneDeviceMetaRequest failed.";
    response.set_found(false);
  } else {
    const std::string &fl_id = get_one_metadata_req.fl_id();
    MS_LOG(DEBUG) << "Getting one client metadata for " << fl_id;

    auto found = DoGetOneDeviceMeta(fl_id, response.mutable_device_meta());
    response.set_found(found);
  }
  std::string getting_meta_rsp_msg = response.SerializeAsString();
  if (!communicator_->SendResponse(getting_meta_rsp_msg.data(), getting_meta_rsp_msg.size(), message)) {
    MS_LOG(WARNING) << "Sending response of GetOneDeviceMetaRequest failed.";
    return;
  }
}

bool DistributedMetadataStore::DoUpdateMetadata(const std::string &name, const PBMetadata &meta) {
  std::unique_lock<std::mutex> lock(mutex_[name]);
  if (metadata_.count(name) == 0) {
    MS_LOG(WARNING) << "The metadata of " << name << " is not registered.";
    return false;
  }
  if (meta.has_device_meta()) {
    auto &fl_id_to_meta_map = *metadata_[name].mutable_device_metas()->mutable_fl_id_to_meta();
    auto &device_meta_fl_id = meta.device_meta().fl_id();
    if (fl_id_to_meta_map.count(device_meta_fl_id) != 0) {
      MS_LOG(WARNING) << "The fl id " << device_meta_fl_id << " already exists.";
      return false;
    }
    auto &device_meta = meta.device_meta();
    fl_id_to_meta_map[device_meta_fl_id] = device_meta;
  } else if (meta.has_fl_id()) {
    auto client_list = metadata_[name].mutable_client_list();
    auto &fl_id = meta.fl_id().fl_id();
    // Check whether the new item already exists.
    bool add_flag = true;
    for (int i = 0; i < client_list->fl_id_size(); i++) {
      if (fl_id == client_list->fl_id(i)) {
        add_flag = false;
        break;
      }
    }
    if (add_flag) {
      client_list->add_fl_id(fl_id);
    }
  } else if (meta.has_update_model_threshold()) {
    auto update_model_threshold = metadata_[name].mutable_update_model_threshold();
    *update_model_threshold = meta.update_model_threshold();
  } else if (meta.has_prime()) {
    metadata_[name] = meta;
  } else {
    return DoUpdateEncryptMetadata(name, meta);
  }
  return true;
}

bool DistributedMetadataStore::DoUpdateEncryptMetadata(const std::string &name, const PBMetadata &meta) {
  if (meta.has_pair_client_keys()) {
    bool keys_update_succeed = UpdatePairClientKeys(name, meta);
    if (!keys_update_succeed) {
      MS_LOG(WARNING) << "Update pair_client_keys failed.";
      return false;
    }
  } else if (meta.has_pair_client_shares()) {
    bool shares_update_succeed = UpdatePairClientShares(name, meta);
    if (!shares_update_succeed) {
      MS_LOG(WARNING) << "Update pair_client_shares failed.";
      return false;
    }
  } else if (meta.has_one_client_noises()) {
    auto &client_noises = *metadata_[name].mutable_client_noises();
    if (client_noises.has_one_client_noises()) {
      MS_LOG(WARNING) << "Leader server updating value for " << name
                      << " failed: The Protobuffer of this value already exists.";
      client_noises.Clear();
    }
    client_noises.mutable_one_client_noises()->MergeFrom(meta.one_client_noises());
  } else if (meta.has_pair_client_list_sign()) {
    auto &client_list_sign_map = *metadata_[name].mutable_client_list_sign()->mutable_client_list_sign();
    auto &fl_id = meta.pair_client_list_sign().fl_id();
    // Check whether the new item already exists.
    if (client_list_sign_map.count(fl_id) != 0) {
      MS_LOG(WARNING) << "Leader server updating value for " << name << " failed: The Protobuffer of fl id " << fl_id
                      << " already exists.";
      return false;
    }
    auto &client_list_sign = meta.pair_client_list_sign().signature();
    client_list_sign_map[fl_id] = client_list_sign;
  } else if (meta.has_pair_key_attestation()) {
    auto &key_attestation_map = *metadata_[name].mutable_key_attestation()->mutable_key_attestations();
    auto &fl_id = meta.pair_key_attestation().fl_id();
    // Check whether the new item already exists.
    if (key_attestation_map.count(fl_id) != 0) {
      MS_LOG(WARNING) << "Leader server updating value for " << name << " failed: The Protobuffer of fl id " << fl_id
                      << " already exists.";
      return false;
    }
    auto &certificate = meta.pair_key_attestation().certificate();
    key_attestation_map[fl_id] = certificate;
  } else {
    MS_LOG(WARNING) << "Leader server updating value for " << name
                    << " failed: The Protobuffer of this value is not defined.";
    return false;
  }
  return true;
}

bool DistributedMetadataStore::DoGetOneDeviceMeta(const std::string &fl_id, DeviceMeta *device_meta) {
  if (device_meta == nullptr) {
    return false;
  }
  const auto &name = kCtxDeviceMetas;
  std::unique_lock<std::mutex> lock(mutex_[name]);
  auto meta_it = metadata_.find(name);
  if (meta_it == metadata_.end()) {
    MS_LOG(WARNING) << "The metadata of " << name << " is not registered.";
    return false;
  }
  auto &stored_meta = meta_it->second;
  const auto &fl_id_to_meta = stored_meta.device_metas().fl_id_to_meta();
  auto fl_it = fl_id_to_meta.find(fl_id);
  if (fl_it == fl_id_to_meta.end()) {
    return false;
  }
  *device_meta = fl_it->second;
  return true;
}

bool DistributedMetadataStore::UpdatePairClientKeys(const std::string &name, const PBMetadata &meta) {
  auto &client_keys_map = *metadata_[name].mutable_client_keys()->mutable_client_keys();
  auto &fl_id = meta.pair_client_keys().fl_id();
  auto &client_keys = meta.pair_client_keys().client_keys();
  // Check whether the new item already exists.
  bool add_flag = true;
  for (auto iter = client_keys_map.begin(); iter != client_keys_map.end(); ++iter) {
    if (fl_id == iter->first) {
      add_flag = false;
      MS_LOG(WARNING) << "Leader server updating value for " << name
                      << " failed: The Protobuffer of this value already exists.";
      break;
    }
  }
  if (add_flag) {
    client_keys_map[fl_id] = client_keys;
    return true;
  } else {
    return false;
  }
}

bool DistributedMetadataStore::UpdatePairClientShares(const std::string &name, const PBMetadata &meta) {
  auto &client_shares_map = *metadata_[name].mutable_client_shares()->mutable_client_secret_shares();
  auto &fl_id = meta.pair_client_shares().fl_id();
  auto &client_shares = meta.pair_client_shares().client_shares();
  // google::protobuf::Map< std::string, mindspore::fl::ps::core::SharesPb >::const_iterator iter;
  // Check whether the new item already exists.
  bool add_flag = true;
  for (auto iter = client_shares_map.begin(); iter != client_shares_map.end(); ++iter) {
    if (fl_id == iter->first) {
      add_flag = false;
      MS_LOG(WARNING) << "Leader server updating value for " << name
                      << " failed: The Protobuffer of this value already exists.";
      break;
    }
  }
  if (add_flag) {
    client_shares_map[fl_id] = client_shares;
    return true;
  } else {
    return false;
  }
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
