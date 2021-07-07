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
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(ERROR) << "inputs or outputs size is invalid.";
    return false;
  }
  void *req_data = inputs[0]->addr;
  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    MS_LOG(ERROR) << "FBBuilder builder or req_data is nullptr.";
    return false;
  }

  MS_LOG(INFO) << "Launching UpdateModelKernel kernel.";
  if (ReachThresholdForUpdateModel(fbb)) {
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  const schema::RequestUpdateModel *update_model_req = flatbuffers::GetRoot<schema::RequestUpdateModel>(req_data);
  if (!UpdateModel(update_model_req, fbb)) {
    MS_LOG(ERROR) << "Updating model failed.";
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  if (!CountForUpdateModel(fbb, update_model_req)) {
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
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

void UpdateModelKernel::OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &message) {
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

bool UpdateModelKernel::ReachThresholdForUpdateModel(const std::shared_ptr<FBBuilder> &fbb) {
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    std::string reason = "Current amount for updateModel is enough. Please retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return true;
  }
  return false;
}

bool UpdateModelKernel::UpdateModel(const schema::RequestUpdateModel *update_model_req,
                                    const std::shared_ptr<FBBuilder> &fbb) {
  RETURN_IF_NULL(update_model_req, false);
  size_t iteration = static_cast<size_t>(update_model_req->iteration());
  if (iteration != LocalMetaStore::GetInstance().curr_iter_num()) {
    std::string reason = "UpdateModel iteration number is invalid:" + std::to_string(iteration) +
                         ", current iteration:" + std::to_string(LocalMetaStore::GetInstance().curr_iter_num()) +
                         ". Retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return true;
  }

  PBMetadata device_metas = DistributedMetadataStore::GetInstance().GetMetadata(kCtxDeviceMetas);
  FLIdToDeviceMeta fl_id_to_meta = device_metas.device_metas();
  std::string update_model_fl_id = update_model_req->fl_id()->str();
  MS_LOG(INFO) << "Update model for fl id " << update_model_fl_id;
  if (fl_id_to_meta.fl_id_to_meta().count(update_model_fl_id) == 0) {
    std::string reason = "devices_meta for " + update_model_fl_id + " is not set. Please retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(ERROR) << reason;
    return false;
  }

  size_t data_size = fl_id_to_meta.fl_id_to_meta().at(update_model_fl_id).data_size();
  auto feature_map = ParseFeatureMap(update_model_req);
  if (feature_map.empty()) {
    std::string reason = "Feature map is empty.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_RequestError, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(ERROR) << reason;
    return false;
  }

  for (auto weight : feature_map) {
    weight.second[kNewDataSize].addr = &data_size;
    weight.second[kNewDataSize].size = sizeof(size_t);
    executor_->HandleModelUpdate(weight.first, weight.second);
  }

  FLId fl_id;
  fl_id.set_fl_id(update_model_fl_id);
  PBMetadata comm_value;
  *comm_value.mutable_fl_id() = fl_id;
  if (!DistributedMetadataStore::GetInstance().UpdateMetadata(kCtxUpdateModelClientList, comm_value)) {
    std::string reason = "Updating metadata of UpdateModelClientList failed.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(ERROR) << reason;
    return false;
  }

  BuildUpdateModelRsp(fbb, schema::ResponseCode_SUCCEED, "success not ready",
                      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
  return true;
}

std::map<std::string, UploadData> UpdateModelKernel::ParseFeatureMap(
  const schema::RequestUpdateModel *update_model_req) {
  RETURN_IF_NULL(update_model_req, {});
  std::map<std::string, UploadData> feature_map;
  auto fbs_feature_map = update_model_req->feature_map();
  RETURN_IF_NULL(fbs_feature_map, feature_map);
  for (size_t i = 0; i < fbs_feature_map->size(); i++) {
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

bool UpdateModelKernel::CountForUpdateModel(const std::shared_ptr<FBBuilder> &fbb,
                                            const schema::RequestUpdateModel *update_model_req) {
  RETURN_IF_NULL(update_model_req, false);
  if (!DistributedCountService::GetInstance().Count(name_, update_model_req->fl_id()->str())) {
    std::string reason = "Counting for update model request failed. Please retry later.";
    BuildUpdateModelRsp(
      fbb, schema::ResponseCode_OutOfTime, reason,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(ERROR) << reason;
    return false;
  }
  return true;
}

void UpdateModelKernel::BuildUpdateModelRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                            const std::string &reason, const std::string &next_req_time) {
  auto fbs_reason = fbb->CreateString(reason);
  auto fbs_next_req_time = fbb->CreateString(next_req_time);

  schema::ResponseUpdateModelBuilder rsp_update_model_builder(*(fbb.get()));
  rsp_update_model_builder.add_retcode(retcode);
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
