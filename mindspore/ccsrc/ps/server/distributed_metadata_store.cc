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

#include "ps/server/distributed_metadata_store.h"
#include <memory>
#include <string>
#include <vector>

namespace mindspore {
namespace ps {
namespace server {
void DistributedMetadataStore::Initialize(const std::shared_ptr<core::ServerNode> &server_node) {
  server_node_ = server_node;
  MS_EXCEPTION_IF_NULL(server_node);
  local_rank_ = server_node_->rank_id();
  server_num_ = PSContext::instance()->initial_server_num();
  InitHashRing();
  return;
}

void DistributedMetadataStore::RegisterMessageCallback(const std::shared_ptr<core::TcpCommunicator> &communicator) {
  communicator_ = communicator;
  MS_EXCEPTION_IF_NULL(communicator_);
  communicator_->RegisterMsgCallBack(
    "updateMetadata", std::bind(&DistributedMetadataStore::HandleUpdateMetadataRequest, this, std::placeholders::_1));
  communicator_->RegisterMsgCallBack(
    "getMetadata", std::bind(&DistributedMetadataStore::HandleGetMetadataRequest, this, std::placeholders::_1));
  return;
}

void DistributedMetadataStore::RegisterMetadata(const std::string &name, const PBMetadata &meta) {
  if (router_ == nullptr) {
    MS_LOG(ERROR) << "The consistent hash ring is not initialized yet.";
    return;
  }

  uint32_t stored_rank = router_->Find(name);
  if (local_rank_ == stored_rank) {
    if (metadata_.count(name) != 0) {
      MS_LOG(ERROR) << "The metadata for " << name << " is already registered.";
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
    MS_LOG(ERROR) << "The consistent hash ring is not initialized yet.";
    return;
  }

  uint32_t stored_rank = router_->Find(name);
  if (local_rank_ == stored_rank) {
    if (metadata_.count(name) == 0) {
      MS_LOG(ERROR) << "The metadata for " << name << " is not registered.";
      return;
    }

    MS_LOG(INFO) << "Rank " << local_rank_ << " reset metadata for " << name;
    std::unique_lock<std::mutex> lock(mutex_[name]);
    PBMetadata empty_meta;
    metadata_[name] = empty_meta;
  }
  return;
}

bool DistributedMetadataStore::UpdateMetadata(const std::string &name, const PBMetadata &meta) {
  if (router_ == nullptr) {
    MS_LOG(ERROR) << "The consistent hash ring is not initialized yet.";
    return false;
  }

  uint32_t stored_rank = router_->Find(name);
  MS_LOG(INFO) << "Rank " << local_rank_ << " update value for " << name << " which is stored in rank " << stored_rank;
  if (local_rank_ == stored_rank) {
    if (!DoUpdateMetadata(name, meta)) {
      MS_LOG(ERROR) << "Updating meta data failed.";
      return false;
    }
  } else {
    PBMetadataWithName metadata_with_name;
    metadata_with_name.set_name(name);
    *metadata_with_name.mutable_metadata() = meta;
    std::shared_ptr<std::vector<unsigned char>> update_meta_rsp_msg = nullptr;
    if (!communicator_->SendPbRequest(metadata_with_name, stored_rank, core::TcpUserCommand::kUpdateMetadata,
                                      &update_meta_rsp_msg)) {
      MS_LOG(ERROR) << "Sending updating metadata message to server " << stored_rank << " failed.";
      return false;
    }

    std::string update_meta_rsp =
      std::string(reinterpret_cast<char *>(update_meta_rsp_msg->data()), update_meta_rsp_msg->size());
    if (update_meta_rsp != kSuccess) {
      MS_LOG(ERROR) << "Updating metadata in server " << stored_rank << " failed. " << update_meta_rsp;
      return false;
    }
  }
  return true;
}

PBMetadata DistributedMetadataStore::GetMetadata(const std::string &name) {
  if (router_ == nullptr) {
    MS_LOG(ERROR) << "The consistent hash ring is not initialized yet.";
    return {};
  }

  uint32_t stored_rank = router_->Find(name);
  MS_LOG(INFO) << "Rank " << local_rank_ << " get metadata for " << name << " which is stored in rank " << stored_rank;
  if (local_rank_ == stored_rank) {
    std::unique_lock<std::mutex> lock(mutex_[name]);
    return metadata_[name];
  } else {
    GetMetadataRequest get_metadata_req;
    get_metadata_req.set_name(name);
    PBMetadata get_metadata_rsp;

    std::shared_ptr<std::vector<unsigned char>> get_meta_rsp_msg = nullptr;
    if (!communicator_->SendPbRequest(get_metadata_req, stored_rank, core::TcpUserCommand::kGetMetadata,
                                      &get_meta_rsp_msg)) {
      MS_LOG(ERROR) << "Sending getting metadata message to server " << stored_rank << " failed.";
      return get_metadata_rsp;
    }
    get_metadata_rsp.ParseFromArray(get_meta_rsp_msg->data(), get_meta_rsp_msg->size());
    return get_metadata_rsp;
  }
}

bool DistributedMetadataStore::ReInitForScaling() {
  // If DistributedMetadataStore is not initialized yet but the scaling event is triggered, do not throw exception.
  if (server_node_ == nullptr) {
    return true;
  }

  MS_LOG(INFO) << "Cluster scaling completed. Reinitialize for distributed metadata store.";
  local_rank_ = server_node_->rank_id();
  server_num_ = IntToUint(server_node_->server_num());
  MS_LOG(INFO) << "After scheduler scaling, this server's rank is " << local_rank_ << ", server number is "
               << server_num_;
  InitHashRing();

  // Clear old metadata.
  metadata_.clear();
  return true;
}

void DistributedMetadataStore::InitHashRing() {
  router_ = std::make_shared<ConsistentHashRing>(32);
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

void DistributedMetadataStore::HandleUpdateMetadataRequest(const std::shared_ptr<core::MessageHandler> &message) {
  if (message == nullptr) {
    MS_LOG(ERROR) << "Message is nullptr.";
    return;
  }

  PBMetadataWithName meta_with_name;
  meta_with_name.ParseFromArray(message->data(), message->len());
  const std::string &name = meta_with_name.name();
  MS_LOG(INFO) << "Update metadata for " << name;

  std::string update_meta_rsp_msg;
  if (!DoUpdateMetadata(name, meta_with_name.metadata())) {
    update_meta_rsp_msg = "Updating meta data failed.";
    MS_LOG(ERROR) << update_meta_rsp_msg;
  } else {
    update_meta_rsp_msg = "Success";
  }
  communicator_->SendResponse(update_meta_rsp_msg.data(), update_meta_rsp_msg.size(), message);
  return;
}

void DistributedMetadataStore::HandleGetMetadataRequest(const std::shared_ptr<core::MessageHandler> &message) {
  if (message == nullptr) {
    MS_LOG(ERROR) << "Message is nullptr.";
    return;
  }

  GetMetadataRequest get_metadata_req;
  get_metadata_req.ParseFromArray(message->data(), message->len());
  const std::string &name = get_metadata_req.name();
  MS_LOG(INFO) << "Getting metadata for " << name;

  std::unique_lock<std::mutex> lock(mutex_[name]);
  PBMetadata stored_meta = metadata_[name];
  std::string getting_meta_rsp_msg = stored_meta.SerializeAsString();
  communicator_->SendResponse(getting_meta_rsp_msg.data(), getting_meta_rsp_msg.size(), message);
  return;
}

bool DistributedMetadataStore::DoUpdateMetadata(const std::string &name, const PBMetadata &meta) {
  std::unique_lock<std::mutex> lock(mutex_[name]);
  if (meta.has_device_meta()) {
    auto &fl_id_to_meta_map = *metadata_[name].mutable_device_metas()->mutable_fl_id_to_meta();
    auto &fl_id = meta.device_meta().fl_id();
    auto &device_meta = meta.device_meta();
    fl_id_to_meta_map[fl_id] = device_meta;
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
  }
  return true;
}
}  // namespace server
}  // namespace ps
}  // namespace mindspore
