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

#include "fl/server/kernel/round/start_fl_job_kernel.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "fl/server/model_store.h"
#include "fl/server/iteration.h"
#ifdef ENABLE_ARMOUR
#include "fl/armour/cipher/cipher_init.h"
#endif

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void StartFLJobKernel::InitKernel(size_t) {
  // The time window of one iteration should be started at the first message of startFLJob round.
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }
  iter_next_req_timestamp_ = LongToUlong(CURRENT_TIME_MILLI.count()) + iteration_time_window_;
  LocalMetaStore::GetInstance().put_value(kCtxIterationNextRequestTimestamp, iter_next_req_timestamp_);

  executor_ = &Executor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor_);
  if (!executor_->initialized()) {
    MS_LOG(EXCEPTION) << "Executor must be initialized in server pipeline.";
    return;
  }

  PBMetadata devices_metas;
  DistributedMetadataStore::GetInstance().RegisterMetadata(kCtxDeviceMetas, devices_metas);
  return;
}

bool StartFLJobKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &outputs) {
  MS_LOG(INFO) << "Launching StartFLJobKernel kernel.";
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
  if (!verifier.VerifyBuffer<schema::RequestFLJob>()) {
    std::string reason = "The schema of RequestFLJob is invalid.";
    BuildStartFLJobRsp(fbb, schema::ResponseCode_RequestError, reason, false, "");
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  ResultCode result_code = ReachThresholdForStartFLJob(fbb);
  if (result_code != ResultCode::kSuccess) {
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return ConvertResultCode(result_code);
  }

  const schema::RequestFLJob *start_fl_job_req = flatbuffers::GetRoot<schema::RequestFLJob>(req_data);
  if (start_fl_job_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestFLJob.";
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_RequestError, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, reason.c_str(), reason.size());
    return true;
  }

  DeviceMeta device_meta = CreateDeviceMetadata(start_fl_job_req);
  result_code = ReadyForStartFLJob(fbb, device_meta);
  if (result_code != ResultCode::kSuccess) {
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return ConvertResultCode(result_code);
  }
  PBMetadata metadata;
  *metadata.mutable_device_meta() = device_meta;
  std::string update_reason = "";
  if (!DistributedMetadataStore::GetInstance().UpdateMetadata(kCtxDeviceMetas, metadata, &update_reason)) {
    std::string reason = "Updating device metadata failed. " + update_reason;
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_OutOfTime, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return update_reason == kNetworkError ? false : true;
  }

  StartFLJob(fbb, device_meta);
  // If calling ReportCount before ReadyForStartFLJob, the result will be inconsistent if the device is not selected.
  result_code = CountForStartFLJob(fbb, start_fl_job_req);
  if (result_code != ResultCode::kSuccess) {
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return ConvertResultCode(result_code);
  }
  GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
  return true;
}

bool StartFLJobKernel::Reset() {
  MS_LOG(INFO) << "Starting fl job kernel reset!";
  StopTimer();
  DistributedCountService::GetInstance().ResetCounter(name_);
  DistributedMetadataStore::GetInstance().ResetMetadata(kCtxDeviceMetas);
  return true;
}

void StartFLJobKernel::OnFirstCountEvent(const std::shared_ptr<ps::core::MessageHandler> &) {
  iter_next_req_timestamp_ = LongToSize(CURRENT_TIME_MILLI.count()) + iteration_time_window_;
  LocalMetaStore::GetInstance().put_value(kCtxIterationNextRequestTimestamp, iter_next_req_timestamp_);
  // The first startFLJob request means a new iteration starts running.
  Iteration::GetInstance().SetIterationRunning();
}

ResultCode StartFLJobKernel::ReachThresholdForStartFLJob(const std::shared_ptr<FBBuilder> &fbb) {
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    std::string reason = "Current amount for startFLJob has reached the threshold. Please startFLJob later.";
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_OutOfTime, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return ResultCode::kSuccessAndReturn;
  }
  return ResultCode::kSuccess;
}

DeviceMeta StartFLJobKernel::CreateDeviceMetadata(const schema::RequestFLJob *start_fl_job_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req, {});
  std::string fl_name = start_fl_job_req->fl_name()->str();
  std::string fl_id = start_fl_job_req->fl_id()->str();
  int data_size = start_fl_job_req->data_size();
  MS_LOG(INFO) << "DeviceMeta fl_name:" << fl_name << ", fl_id:" << fl_id << ", data_size:" << data_size;

  DeviceMeta device_meta;
  device_meta.set_fl_name(fl_name);
  device_meta.set_fl_id(fl_id);
  device_meta.set_data_size(IntToSize(data_size));
  return device_meta;
}

ResultCode StartFLJobKernel::ReadyForStartFLJob(const std::shared_ptr<FBBuilder> &fbb, const DeviceMeta &device_meta) {
  ResultCode ret = ResultCode::kSuccess;
  std::string reason = "";
  if (device_meta.data_size() < 1) {
    reason = "FL job data size is not enough.";
    ret = ResultCode::kSuccessAndReturn;
  }
  if (ret != ResultCode::kSuccess) {
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_OutOfTime, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
  }
  return ret;
}

ResultCode StartFLJobKernel::CountForStartFLJob(const std::shared_ptr<FBBuilder> &fbb,
                                                const schema::RequestFLJob *start_fl_job_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req, ResultCode::kSuccessAndReturn);
  std::string count_reason = "";
  if (!DistributedCountService::GetInstance().Count(name_, start_fl_job_req->fl_id()->str(), &count_reason)) {
    std::string reason = "Counting start fl job request failed. Please retry later.";
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_OutOfTime, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(ERROR) << reason;
    return count_reason == kNetworkError ? ResultCode::kFail : ResultCode::kSuccessAndReturn;
  }
  return ResultCode::kSuccess;
}

void StartFLJobKernel::StartFLJob(const std::shared_ptr<FBBuilder> &fbb, const DeviceMeta &) {
  size_t last_iteration = LocalMetaStore::GetInstance().curr_iter_num() - 1;
  auto feature_maps = ModelStore::GetInstance().GetModelByIterNum(last_iteration);
  if (feature_maps.empty()) {
    MS_LOG(WARNING) << "The feature map for startFLJob is empty.";
  }
  BuildStartFLJobRsp(fbb, schema::ResponseCode_SUCCEED, "success", true,
                     std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)),
                     feature_maps);
  return;
}

void StartFLJobKernel::BuildStartFLJobRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                          const std::string &reason, const bool is_selected,
                                          const std::string &next_req_time,
                                          std::map<std::string, AddressPtr> feature_maps) {
  if (fbb == nullptr) {
    MS_LOG(ERROR) << "Input fbb is nullptr.";
    return;
  }
  auto fbs_reason = fbb->CreateString(reason);
  auto fbs_next_req_time = fbb->CreateString(next_req_time);
  auto fbs_server_mode = fbb->CreateString(ps::PSContext::instance()->server_mode());
  auto fbs_fl_name = fbb->CreateString(ps::PSContext::instance()->fl_name());

#ifdef ENABLE_ARMOUR
  auto *param = armour::CipherInit::GetInstance().GetPublicParams();
  auto prime = fbb->CreateVector(param->prime, PRIME_MAX_LEN);
  auto p = fbb->CreateVector(param->p, SECRET_MAX_LEN);
  int32_t t = param->t;
  int32_t g = param->g;
  float dp_eps = param->dp_eps;
  float dp_delta = param->dp_delta;
  float dp_norm_clip = param->dp_norm_clip;
  auto encrypt_type = fbb->CreateString(ps::PSContext::instance()->encrypt_type());

  auto cipher_public_params =
    schema::CreateCipherPublicParams(*fbb.get(), t, p, g, prime, dp_eps, dp_delta, dp_norm_clip, encrypt_type);
#endif

  schema::FLPlanBuilder fl_plan_builder(*(fbb.get()));
  fl_plan_builder.add_fl_name(fbs_fl_name);
  fl_plan_builder.add_server_mode(fbs_server_mode);
  fl_plan_builder.add_iterations(SizeToInt(ps::PSContext::instance()->fl_iteration_num()));
  fl_plan_builder.add_epochs(SizeToInt(ps::PSContext::instance()->client_epoch_num()));
  fl_plan_builder.add_mini_batch(SizeToInt(ps::PSContext::instance()->client_batch_size()));
  fl_plan_builder.add_lr(ps::PSContext::instance()->client_learning_rate());
#ifdef ENABLE_ARMOUR
  fl_plan_builder.add_cipher(cipher_public_params);
#endif
  auto fbs_fl_plan = fl_plan_builder.Finish();

  std::vector<flatbuffers::Offset<schema::FeatureMap>> fbs_feature_maps;
  for (auto feature_map : feature_maps) {
    auto fbs_weight_fullname = fbb->CreateString(feature_map.first);
    auto fbs_weight_data =
      fbb->CreateVector(reinterpret_cast<float *>(feature_map.second->addr), feature_map.second->size / sizeof(float));
    auto fbs_feature_map = schema::CreateFeatureMap(*(fbb.get()), fbs_weight_fullname, fbs_weight_data);
    fbs_feature_maps.push_back(fbs_feature_map);
  }
  auto fbs_feature_maps_vector = fbb->CreateVector(fbs_feature_maps);

  schema::ResponseFLJobBuilder rsp_fl_job_builder(*(fbb.get()));
  rsp_fl_job_builder.add_retcode(static_cast<int>(retcode));
  rsp_fl_job_builder.add_reason(fbs_reason);
  rsp_fl_job_builder.add_iteration(SizeToInt(LocalMetaStore::GetInstance().curr_iter_num()));
  rsp_fl_job_builder.add_is_selected(is_selected);
  rsp_fl_job_builder.add_next_req_time(fbs_next_req_time);
  rsp_fl_job_builder.add_fl_plan_config(fbs_fl_plan);
  rsp_fl_job_builder.add_feature_map(fbs_feature_maps_vector);
  auto rsp_fl_job = rsp_fl_job_builder.Finish();
  fbb->Finish(rsp_fl_job);
  return;
}

REG_ROUND_KERNEL(startFLJob, StartFLJobKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
