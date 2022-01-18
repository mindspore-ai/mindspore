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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "fl/server/kernel/round/update_model_kernel.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
constexpr uint32_t kRetryCountOfWaitWeightAggregation = 30;
void UpdateModelKernel::InitKernel(size_t threshold_count) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }
  InitClientVisitedNum();
  InitClientUploadLoss();
  executor_ = &Executor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor_);
  if (!executor_->initialized()) {
    MS_LOG(EXCEPTION) << "Executor must be initialized in server pipeline.";
    return;
  }

  PBMetadata client_list;
  DistributedMetadataStore::GetInstance().RegisterMetadata(kCtxUpdateModelClientList, client_list);
  LocalMetaStore::GetInstance().put_value(kCtxUpdateModelThld, threshold_count);
  LocalMetaStore::GetInstance().put_value(kCtxFedAvgTotalDataSize, kInitialDataSizeSum);
}

bool UpdateModelKernel::Launch(const uint8_t *req_data, size_t len,
                               const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_LOG(DEBUG) << "Launching UpdateModelKernel kernel.";
  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(WARNING) << reason;
    GenerateOutput(message, reason.c_str(), reason.size());
    return true;
  }

  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::RequestUpdateModel>()) {
    std::string reason = "The schema of RequestUpdateModel is invalid.";
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
    MS_LOG(WARNING) << reason;
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  ResultCode result_code = ReachThresholdForUpdateModel(fbb);
  if (result_code != ResultCode::kSuccess) {
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return ConvertResultCode(result_code);
  }

  const schema::RequestUpdateModel *update_model_req = flatbuffers::GetRoot<schema::RequestUpdateModel>(req_data);
  if (update_model_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestUpdateModel.";
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
    MS_LOG(WARNING) << reason;
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  // verify signature
  if (ps::PSContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(update_model_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
      MS_LOG(WARNING) << reason;
      GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed.";
      BuildUpdateModelRsp(fbb, schema::ResponseCode_OutOfTime, reason, "");
      MS_LOG(WARNING) << reason;
      GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
    MS_LOG(INFO) << "verify signature passed!";
  }
  DeviceMeta device_meta;
  result_code = VerifyUpdateModel(update_model_req, fbb, &device_meta);
  if (result_code != ResultCode::kSuccess) {
    MS_LOG(WARNING) << "Updating model failed.";
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return ConvertResultCode(result_code);
  }

  result_code = CountForUpdateModel(fbb, update_model_req);
  if (result_code != ResultCode::kSuccess) {
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return ConvertResultCode(result_code);
  }

  result_code = UpdateModel(update_model_req, fbb, device_meta);
  if (result_code != ResultCode::kSuccess) {
    MS_LOG(WARNING) << "Updating model failed.";
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return ConvertResultCode(result_code);
  }

  IncreaseAcceptClientNum();
  GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
  return true;
}

bool UpdateModelKernel::Reset() {
  MS_LOG(INFO) << "Update model kernel reset!";
  StopTimer();
  DistributedCountService::GetInstance().ResetCounter(name_);
  executor_->ResetAggregationStatus();
  DistributedMetadataStore::GetInstance().ResetMetadata(kCtxUpdateModelClientList);
  size_t &total_data_size = LocalMetaStore::GetInstance().mutable_value<size_t>(kCtxFedAvgTotalDataSize);
  total_data_size = 0;
  return true;
}

void UpdateModelKernel::OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &) {
  if (ps::PSContext::instance()->resetter_round() == ps::ResetterRound::kUpdateModel) {
    last_count_thread_ = std::make_unique<std::thread>([this]() {
      uint32_t retryCount = 0;
      while (!executor_->IsAllWeightAggregationDone() && retryCount <= kRetryCountOfWaitWeightAggregation) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        retryCount += 1;
      }

      size_t total_data_size = LocalMetaStore::GetInstance().value<size_t>(kCtxFedAvgTotalDataSize);
      MS_LOG(INFO) << "Total data size for iteration " << LocalMetaStore::GetInstance().curr_iter_num() << " is "
                   << total_data_size;
      if (ps::PSContext::instance()->encrypt_type() != ps::kPWEncryptType) {
        FinishIteration();
      }
    });
    last_count_thread_->detach();
  }
}

ResultCode UpdateModelKernel::ReachThresholdForUpdateModel(const std::shared_ptr<FBBuilder> &fbb) {
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    std::string reason = "Current amount for updateModel is enough. Please retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return ResultCode::kSuccessAndReturn;
  }
  return ResultCode::kSuccess;
}

ResultCode UpdateModelKernel::VerifyUpdateModel(const schema::RequestUpdateModel *update_model_req,
                                                const std::shared_ptr<FBBuilder> &fbb, DeviceMeta *device_meta) {
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, ResultCode::kSuccessAndReturn);
  MS_ERROR_IF_NULL_W_RET_VAL(device_meta, ResultCode::kSuccessAndReturn);
  size_t iteration = IntToSize(update_model_req->iteration());
  if (iteration != LocalMetaStore::GetInstance().curr_iter_num()) {
    auto next_req_time = LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp);
    std::string reason = "UpdateModel iteration number is invalid:" + std::to_string(iteration) +
                         ", current iteration:" + std::to_string(LocalMetaStore::GetInstance().curr_iter_num()) +
                         ". Retry later at time: " + std::to_string(next_req_time);
    BuildUpdateModelRsp(fbb, schema::ResponseCode_OutOfTime, reason, std::to_string(next_req_time));
    MS_LOG(WARNING) << reason;
    return ResultCode::kSuccessAndReturn;
  }

  std::string update_model_fl_id = update_model_req->fl_id()->str();
  MS_LOG(DEBUG) << "UpdateModel for fl id " << update_model_fl_id;

  bool found = DistributedMetadataStore::GetInstance().GetOneDeviceMeta(update_model_fl_id, device_meta);
  if (!found) {
    std::string reason = "devices_meta for " + update_model_fl_id + " is not set. Please retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return ResultCode::kSuccessAndReturn;
  }
  if (ps::PSContext::instance()->encrypt_type() == ps::kPWEncryptType) {
    std::vector<std::string> get_secrets_clients;
#ifdef ENABLE_ARMOUR
    mindspore::armour::CipherMetaStorage cipher_meta_storage;
    cipher_meta_storage.GetClientListFromServer(fl::server::kCtxGetSecretsClientList, &get_secrets_clients);
#endif
    if (find(get_secrets_clients.begin(), get_secrets_clients.end(), update_model_fl_id) ==
        get_secrets_clients.end()) {  // the client not in get_secrets_clients
      std::string reason = "fl_id: " + update_model_fl_id + " is not in get_secrets_clients. Please retry later.";
      BuildUpdateModelRsp(
        fbb, schema::ResponseCode_OutOfTime, reason,
        std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
      MS_LOG(WARNING) << reason;
      return ResultCode::kSuccessAndReturn;
    }
  }
  return ResultCode::kSuccess;
}

ResultCode UpdateModelKernel::UpdateModel(const schema::RequestUpdateModel *update_model_req,
                                          const std::shared_ptr<FBBuilder> &fbb, const DeviceMeta &device_meta) {
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, ResultCode::kSuccessAndReturn);
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req->fl_id(), ResultCode::kSuccessAndReturn);

  std::string update_model_fl_id = update_model_req->fl_id()->str();
  size_t data_size = device_meta.data_size();
  const auto &feature_map = ParseFeatureMap(update_model_req);
  if (feature_map.empty()) {
    std::string reason = "Feature map is empty.";
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
    MS_LOG(WARNING) << reason;
    return ResultCode::kSuccessAndReturn;
  }

  for (auto weight : feature_map) {
    weight.second[kNewDataSize].addr = &data_size;
    weight.second[kNewDataSize].size = sizeof(size_t);
    if (!executor_->HandleModelUpdate(weight.first, weight.second)) {
      std::string reason = "Updating weight " + weight.first + " failed.";
      BuildUpdateModelRsp(
        fbb, schema::ResponseCode_OutOfTime, reason,
        std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
      MS_LOG(WARNING) << reason;
      return ResultCode::kFail;
    }
  }

  FLId fl_id;
  fl_id.set_fl_id(update_model_fl_id);
  PBMetadata comm_value;
  *comm_value.mutable_fl_id() = fl_id;
  std::string update_reason = "";
  if (!DistributedMetadataStore::GetInstance().UpdateMetadata(kCtxUpdateModelClientList, comm_value, &update_reason)) {
    std::string reason = "Updating metadata of UpdateModelClientList failed. " + update_reason;
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return update_reason == kNetworkError ? ResultCode::kFail : ResultCode::kSuccessAndReturn;
  }
  UpdateClientUploadLoss(update_model_req->upload_loss());
  BuildUpdateModelRsp(fbb, schema::ResponseCode_SUCCEED, "success not ready",
                      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
  return ResultCode::kSuccess;
}

std::map<std::string, UploadData> UpdateModelKernel::ParseFeatureMap(
  const schema::RequestUpdateModel *update_model_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, {});
  std::map<std::string, UploadData> feature_map;
  auto fbs_feature_map = update_model_req->feature_map();
  MS_ERROR_IF_NULL_W_RET_VAL(fbs_feature_map, feature_map);
  for (uint32_t i = 0; i < fbs_feature_map->size(); i++) {
    std::string weight_full_name = fbs_feature_map->Get(i)->weight_fullname()->str();
    float *weight_data = const_cast<float *>(fbs_feature_map->Get(i)->data()->data());
    size_t weight_size = fbs_feature_map->Get(i)->data()->size() * sizeof(float);
    UploadData upload_data;
    upload_data[kNewWeight].addr = weight_data;
    upload_data[kNewWeight].size = weight_size;
    feature_map[weight_full_name] = upload_data;
  }
  return feature_map;
}

ResultCode UpdateModelKernel::CountForUpdateModel(const std::shared_ptr<FBBuilder> &fbb,
                                                  const schema::RequestUpdateModel *update_model_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(fbb, ResultCode::kSuccessAndReturn);
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, ResultCode::kSuccessAndReturn);
  std::string count_reason = "";
  if (!DistributedCountService::GetInstance().Count(name_, update_model_req->fl_id()->str(), &count_reason)) {
    std::string reason = "Counting for update model request failed. Please retry later. " + count_reason;
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return count_reason == kNetworkError ? ResultCode::kFail : ResultCode::kSuccessAndReturn;
  }
  return ResultCode::kSuccess;
}

sigVerifyResult UpdateModelKernel::VerifySignature(const schema::RequestUpdateModel *update_model_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req->fl_id(), sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req->timestamp(), sigVerifyResult::FAILED);

  std::string fl_id = update_model_req->fl_id()->str();
  std::string timestamp = update_model_req->timestamp()->str();
  int iteration = update_model_req->iteration();
  std::string iter_str = std::to_string(iteration);
  auto fbs_signature = update_model_req->signature();
  std::vector<unsigned char> signature;
  if (fbs_signature == nullptr) {
    MS_LOG(WARNING) << "signature in client_list_sign_req is nullptr";
    return sigVerifyResult::FAILED;
  }
  signature.assign(fbs_signature->begin(), fbs_signature->end());
  std::map<std::string, std::string> key_attestations;
  const fl::PBMetadata &key_attestations_pb_out =
    fl::server::DistributedMetadataStore::GetInstance().GetMetadata(kCtxClientKeyAttestation);
  const fl::KeyAttestation &key_attestation_pb = key_attestations_pb_out.key_attestation();
  auto iter = key_attestation_pb.key_attestations().begin();
  for (; iter != key_attestation_pb.key_attestations().end(); ++iter) {
    (void)key_attestations.emplace(std::pair<std::string, std::string>(iter->first, iter->second));
  }
  if (key_attestations.find(fl_id) == key_attestations.end()) {
    MS_LOG(WARNING) << "can not find key attestation for fl_id: " << fl_id;
    return sigVerifyResult::TIMEOUT;
  }

  std::vector<unsigned char> src_data;
  (void)src_data.insert(src_data.end(), timestamp.begin(), timestamp.end());
  (void)src_data.insert(src_data.end(), iter_str.begin(), iter_str.end());
  auto certVerify = mindspore::ps::server::CertVerify::GetInstance();
  unsigned char srcDataHash[SHA256_DIGEST_LENGTH];
  certVerify.sha256Hash(src_data.data(), SizeToInt(src_data.size()), srcDataHash, SHA256_DIGEST_LENGTH);
  if (!certVerify.verifyRSAKey(key_attestations[fl_id], srcDataHash, signature.data(), SHA256_DIGEST_LENGTH)) {
    return sigVerifyResult::FAILED;
  }
  if (!certVerify.verifyTimeStamp(fl_id, timestamp)) {
    return sigVerifyResult::TIMEOUT;
  }
  MS_LOG(INFO) << "verify signature for fl_id: " << fl_id << " success.";
  return sigVerifyResult::PASSED;
}

void UpdateModelKernel::BuildUpdateModelRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                            const std::string &reason, const std::string &next_req_time) {
  if (fbb == nullptr) {
    MS_LOG(WARNING) << "Input fbb is nullptr.";
    return;
  }
  auto fbs_reason = fbb->CreateString(reason);
  auto fbs_next_req_time = fbb->CreateString(next_req_time);

  schema::ResponseUpdateModelBuilder rsp_update_model_builder(*(fbb.get()));
  rsp_update_model_builder.add_retcode(static_cast<int>(retcode));
  rsp_update_model_builder.add_reason(fbs_reason);
  rsp_update_model_builder.add_next_req_time(fbs_next_req_time);
  auto rsp_update_model = rsp_update_model_builder.Finish();
  fbb->Finish(rsp_update_model);
  return;
}

REG_ROUND_KERNEL(updateModel, UpdateModelKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
