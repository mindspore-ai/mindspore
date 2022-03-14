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
#ifdef ENABLE_ARMOUR
#include "fl/armour/cipher/cipher_init.h"
#endif
#include "fl/server/model_store.h"
#include "fl/server/iteration.h"

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
  InitClientVisitedNum();
  executor_ = &Executor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor_);
  if (!executor_->initialized()) {
    MS_LOG(EXCEPTION) << "Executor must be initialized in server pipeline.";
    return;
  }
  PBMetadata devices_metas;
  DistributedMetadataStore::GetInstance().RegisterMetadata(kCtxDeviceMetas, devices_metas);

  PBMetadata client_key_attestation;
  DistributedMetadataStore::GetInstance().RegisterMetadata(kCtxClientKeyAttestation, client_key_attestation);
  return;
}

bool StartFLJobKernel::Launch(const uint8_t *req_data, size_t len,
                              const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_LOG(DEBUG) << "Launching StartFLJobKernel kernel.";
  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(WARNING) << reason;
    GenerateOutput(message, reason.c_str(), reason.size());
    return false;
  }

  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::RequestFLJob>()) {
    std::string reason = "The schema of RequestFLJob is invalid.";
    BuildStartFLJobRsp(fbb, schema::ResponseCode_RequestError, reason, false, "");
    MS_LOG(WARNING) << reason;
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  const schema::RequestFLJob *start_fl_job_req = flatbuffers::GetRoot<schema::RequestFLJob>(req_data);
  if (start_fl_job_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestFLJob.";
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_RequestError, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    GenerateOutput(message, reason.c_str(), reason.size());
    return false;
  }

  ResultCode result_code = ReachThresholdForStartFLJob(fbb, start_fl_job_req);
  if (result_code != ResultCode::kSuccess) {
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  if (ps::PSContext::instance()->pki_verify()) {
    if (!JudgeFLJobCert(fbb, start_fl_job_req)) {
      GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
      return false;
    }
    if (!StoreKeyAttestation(fbb, start_fl_job_req)) {
      GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
      return false;
    }
  }

  DeviceMeta device_meta = CreateDeviceMetadata(start_fl_job_req);
  result_code = ReadyForStartFLJob(fbb, device_meta);
  if (result_code != ResultCode::kSuccess) {
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }
  PBMetadata metadata;
  *metadata.mutable_device_meta() = device_meta;
  std::string update_reason = "";
  if (!DistributedMetadataStore::GetInstance().UpdateMetadata(kCtxDeviceMetas, metadata, &update_reason)) {
    std::string reason = "Updating device metadata failed for fl id " + device_meta.fl_id();
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_OutOfTime, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  // If calling ReportCount before ReadyForStartFLJob, the result will be inconsistent if the device is not selected.
  result_code = CountForStartFLJob(fbb, start_fl_job_req);
  if (result_code != ResultCode::kSuccess) {
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }
  IncreaseAcceptClientNum();
  auto curr_iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  auto last_iteration = curr_iter_num - 1;
  auto cache = ModelStore::GetInstance().GetModelResponseCache(name_, curr_iter_num, last_iteration);
  if (cache == nullptr) {
    StartFLJob(fbb);
    cache = ModelStore::GetInstance().StoreModelResponseCache(name_, curr_iter_num, last_iteration,
                                                              fbb->GetBufferPointer(), fbb->GetSize());
    if (cache == nullptr) {
      GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
  }
  GenerateOutputInference(message, cache->data(), cache->size(), ModelStore::GetInstance().RelModelResponseCache);
  return true;
}

bool StartFLJobKernel::JudgeFLJobCert(const std::shared_ptr<FBBuilder> &fbb,
                                      const schema::RequestFLJob *start_fl_job_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req, false);
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req->fl_id(), false);
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req->timestamp(), false);

  std::string fl_id = start_fl_job_req->fl_id()->str();
  std::string timestamp = start_fl_job_req->timestamp()->str();
  auto sign_data_vector = start_fl_job_req->sign_data();
  if (sign_data_vector == nullptr || sign_data_vector->size() == 0) {
    std::string reason = "sign data is empty.";
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_RequestError, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return false;
  }
  unsigned char sign_data[sign_data_vector->size()];

  for (unsigned int i = 0; i < sign_data_vector->size(); i++) {
    sign_data[i] = sign_data_vector->Get(i);
  }

  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req->key_attestation(), false);
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req->equip_cert(), false);
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req->equip_ca_cert(), false);

  std::string key_attestation = start_fl_job_req->key_attestation()->str();
  std::string equip_cert = start_fl_job_req->equip_cert()->str();
  std::string equip_ca_cert = start_fl_job_req->equip_ca_cert()->str();
  std::string root_first_ca_path = ps::PSContext::instance()->root_first_ca_path();
  std::string root_second_ca_path = ps::PSContext::instance()->root_second_ca_path();
  std::string equip_crl_path = ps::PSContext::instance()->equip_crl_path();

  auto certVerify = mindspore::ps::server::CertVerify::GetInstance();
  bool ret =
    certVerify.verifyCertAndSign(fl_id, timestamp, (const unsigned char *)sign_data, key_attestation, equip_cert,
                                 equip_ca_cert, root_first_ca_path, root_second_ca_path, equip_crl_path);
  if (!ret) {
    std::string reason = "startFLJob sign and certificate verify failed.";
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_RequestError, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
  } else {
    MS_LOG(DEBUG) << "JudgeFLJobVerify success." << ret;
  }

  return ret;
}

bool StartFLJobKernel::StoreKeyAttestation(const std::shared_ptr<FBBuilder> &fbb,
                                           const schema::RequestFLJob *start_fl_job_req) {
  // update key attestation
  if (start_fl_job_req == nullptr) {
    return false;
  }
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req->fl_id(), false);
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req->key_attestation(), false);

  std::string fl_id = start_fl_job_req->fl_id()->str();
  std::string key_attestation = start_fl_job_req->key_attestation()->str();

  fl::PairKeyAttestation pair_key_attestation_pb;
  pair_key_attestation_pb.set_fl_id(fl_id);
  pair_key_attestation_pb.set_certificate(key_attestation);

  fl::PBMetadata pb_data;
  pb_data.mutable_pair_key_attestation()->MergeFrom(pair_key_attestation_pb);
  bool ret = fl::server::DistributedMetadataStore::GetInstance().UpdateMetadata(kCtxClientKeyAttestation, pb_data);
  if (!ret) {
    std::string reason = "startFLJob: store key attestation failed";
    MS_LOG(WARNING) << reason;
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_OutOfTime, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    return false;
  }
  return true;
}

bool StartFLJobKernel::Reset() {
  MS_LOG(INFO) << "Starting fl job kernel reset!";
  StopTimer();
  DistributedCountService::GetInstance().ResetCounter(name_);
  DistributedMetadataStore::GetInstance().ResetMetadata(kCtxDeviceMetas);
  DistributedMetadataStore::GetInstance().ResetMetadata(kCtxClientKeyAttestation);
  return true;
}

void StartFLJobKernel::OnFirstCountEvent(const std::shared_ptr<ps::core::MessageHandler> &) {
  iter_next_req_timestamp_ = LongToUlong(CURRENT_TIME_MILLI.count()) + iteration_time_window_;
  LocalMetaStore::GetInstance().put_value(kCtxIterationNextRequestTimestamp, iter_next_req_timestamp_);
  // The first startFLJob request means a new iteration starts running.
  Iteration::GetInstance().SetIterationRunning();
}

ResultCode StartFLJobKernel::ReachThresholdForStartFLJob(const std::shared_ptr<FBBuilder> &fbb,
                                                         const schema::RequestFLJob *start_fl_job_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req, ResultCode::kFail);
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req->fl_id(), ResultCode::kFail);
  if (DistributedCountService::GetInstance().CountReachThreshold(name_, start_fl_job_req->fl_id()->str())) {
    std::string reason = "Current amount for startFLJob has reached the threshold. Please startFLJob later.";
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_OutOfTime, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(DEBUG) << reason;
    return ResultCode::kFail;
  }
  return ResultCode::kSuccess;
}

DeviceMeta StartFLJobKernel::CreateDeviceMetadata(const schema::RequestFLJob *start_fl_job_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req, {});
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req->fl_name(), {});
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req->fl_id(), {});

  std::string fl_name = start_fl_job_req->fl_name()->str();
  std::string fl_id = start_fl_job_req->fl_id()->str();
  int data_size = start_fl_job_req->data_size();
  MS_LOG(DEBUG) << "DeviceMeta fl_name:" << fl_name << ", fl_id:" << fl_id << ", data_size:" << data_size;

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
    ret = ResultCode::kFail;
  }
  if (ret != ResultCode::kSuccess) {
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_OutOfTime, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(DEBUG) << reason;
  }
  return ret;
}

ResultCode StartFLJobKernel::CountForStartFLJob(const std::shared_ptr<FBBuilder> &fbb,
                                                const schema::RequestFLJob *start_fl_job_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req, ResultCode::kFail);
  MS_ERROR_IF_NULL_W_RET_VAL(start_fl_job_req->fl_id(), ResultCode::kFail);

  std::string count_reason = "";
  if (!DistributedCountService::GetInstance().Count(name_, start_fl_job_req->fl_id()->str(), &count_reason)) {
    std::string reason =
      "Counting start fl job request failed for fl id " + start_fl_job_req->fl_id()->str() + ", Please retry later.";
    BuildStartFLJobRsp(
      fbb, schema::ResponseCode_OutOfTime, reason, false,
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)));
    MS_LOG(WARNING) << reason;
    return ResultCode::kFail;
  }
  return ResultCode::kSuccess;
}

void StartFLJobKernel::StartFLJob(const std::shared_ptr<FBBuilder> &fbb) {
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
    MS_LOG(WARNING) << "Input fbb is nullptr.";
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
  float sign_k = param->sign_k;
  float sign_eps = param->sign_eps;
  float sign_thr_ratio = param->sign_thr_ratio;
  float sign_global_lr = param->sign_global_lr;
  int sign_dim_out = param->sign_dim_out;

  auto pw_params = schema::CreatePWParams(*fbb.get(), t, p, g, prime);
  auto dp_params = schema::CreateDPParams(*fbb.get(), dp_eps, dp_delta, dp_norm_clip);
  auto ds_params = schema::CreateDSParams(*fbb.get(), sign_k, sign_eps, sign_thr_ratio, sign_global_lr, sign_dim_out);
  auto cipher_public_params =
    schema::CreateCipherPublicParams(*fbb.get(), encrypt_type, pw_params, dp_params, ds_params);
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
