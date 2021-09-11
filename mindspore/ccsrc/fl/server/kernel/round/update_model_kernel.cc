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
#include "fl/server/kernel/round/update_model_kernel.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void UpdateModelKernel::InitKernel(size_t threshold_count) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }

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

bool UpdateModelKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                               const std::vector<AddressPtr> &outputs) {
  MS_LOG(INFO) << "Launching UpdateModelKernel kernel.";
  if (inputs.size() != 1 || outputs.size() != 1) {
    std::string reason = "inputs or outputs size is invalid.";
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, reason.c_str(), reason.size());
    return true;
  }

  void *req_data = inputs[0]->addr;
  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, reason.c_str(), reason.size());
    return true;
  }

  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t *>(req_data), inputs[0]->size);
  if (!verifier.VerifyBuffer<schema::RequestUpdateModel>()) {
    std::string reason = "The schema of RequestUpdateModel is invalid.";
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  ResultCode result_code = ReachThresholdForUpdateModel(fbb);
  if (result_code != ResultCode::kSuccess) {
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return ConvertResultCode(result_code);
  }

  const schema::RequestUpdateModel *update_model_req = flatbuffers::GetRoot<schema::RequestUpdateModel>(req_data);
  if (update_model_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestUpdateModel.";
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  result_code = UpdateModel(update_model_req, fbb);
  if (result_code != ResultCode::kSuccess) {
    MS_LOG(ERROR) << "Updating model failed.";
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return ConvertResultCode(result_code);
  }

  result_code = CountForUpdateModel(fbb, update_model_req);
  if (result_code != ResultCode::kSuccess) {
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return ConvertResultCode(result_code);
  }
  GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
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
    while (!executor_->IsAllWeightAggregationDone()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    size_t total_data_size = LocalMetaStore::GetInstance().value<size_t>(kCtxFedAvgTotalDataSize);
    MS_LOG(INFO) << "Total data size for iteration " << LocalMetaStore::GetInstance().curr_iter_num() << " is "
                 << total_data_size;
    if (ps::PSContext::instance()->encrypt_type() != ps::kPWEncryptType) {
      FinishIteration();
    }
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

ResultCode UpdateModelKernel::UpdateModel(const schema::RequestUpdateModel *update_model_req,
                                          const std::shared_ptr<FBBuilder> &fbb) {
  MS_ERROR_IF_NULL_W_RET_VAL(update_model_req, ResultCode::kSuccessAndReturn);
  size_t iteration = IntToSize(update_model_req->iteration());
  if (iteration != LocalMetaStore::GetInstance().curr_iter_num()) {
    std::string reason = "UpdateModel iteration number is invalid:" + std::to_string(iteration) +
                         ", current iteration:" + std::to_string(LocalMetaStore::GetInstance().curr_iter_num()) +
                         ". Retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return ResultCode::kSuccessAndReturn;
  }

  PBMetadata device_metas = DistributedMetadataStore::GetInstance().GetMetadata(kCtxDeviceMetas);
  FLIdToDeviceMeta fl_id_to_meta = device_metas.device_metas();
  std::string update_model_fl_id = update_model_req->fl_id()->str();
  MS_LOG(INFO) << "UpdateModel for fl id " << update_model_fl_id;
  if (ps::PSContext::instance()->encrypt_type() != ps::kPWEncryptType) {
    if (fl_id_to_meta.fl_id_to_meta().count(update_model_fl_id) == 0) {
      std::string reason = "devices_meta for " + update_model_fl_id + " is not set. Please retry later.";
      BuildUpdateModelRsp(
        fbb, schema::ResponseCode_OutOfTime, reason,
        std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
      MS_LOG(ERROR) << reason;
      return ResultCode::kSuccessAndReturn;
    }
  } else {
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
      MS_LOG(ERROR) << reason;
      return ResultCode::kSuccessAndReturn;
    }
  }

  size_t data_size = fl_id_to_meta.fl_id_to_meta().at(update_model_fl_id).data_size();
  auto feature_map = ParseFeatureMap(update_model_req);
  if (feature_map.empty()) {
    std::string reason = "Feature map is empty.";
    BuildUpdateModelRsp(fbb, schema::ResponseCode_RequestError, reason, "");
    MS_LOG(ERROR) << reason;
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
      MS_LOG(ERROR) << reason;
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
    MS_LOG(ERROR) << reason;
    return update_reason == kNetworkError ? ResultCode::kFail : ResultCode::kSuccessAndReturn;
  }

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
    MS_LOG(ERROR) << reason;
    return count_reason == kNetworkError ? ResultCode::kFail : ResultCode::kSuccessAndReturn;
  }
  return ResultCode::kSuccess;
}

void UpdateModelKernel::BuildUpdateModelRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                            const std::string &reason, const std::string &next_req_time) {
  if (fbb == nullptr) {
    MS_LOG(ERROR) << "Input fbb is nullptr.";
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
