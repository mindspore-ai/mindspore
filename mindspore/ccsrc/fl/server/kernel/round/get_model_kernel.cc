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

#include "fl/server/kernel/round/get_model_kernel.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "fl/server/iteration.h"
#include "fl/server/model_store.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void GetModelKernel::InitKernel(size_t) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }

  executor_ = &Executor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor_);
  if (!executor_->initialized()) {
    MS_LOG(EXCEPTION) << "Executor must be initialized in server pipeline.";
    return;
  }
}

bool GetModelKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                            const std::vector<AddressPtr> &outputs) {
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
  if (!verifier.VerifyBuffer<schema::RequestGetModel>()) {
    std::string reason = "The schema of RequestGetModel is invalid.";
    BuildGetModelRsp(fbb, schema::ResponseCode_RequestError, reason, LocalMetaStore::GetInstance().curr_iter_num(), {},
                     "");
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  (void)++retry_count_;
  if (retry_count_.load() % kPrintGetModelForEveryRetryTime == 1) {
    MS_LOG(INFO) << "Launching GetModelKernel retry count is " << retry_count_.load();
  }

  const schema::RequestGetModel *get_model_req = flatbuffers::GetRoot<schema::RequestGetModel>(req_data);
  if (get_model_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestGetModel.";
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, reason.c_str(), reason.size());
    return true;
  }
  GetModel(get_model_req, fbb);
  GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
  return true;
}

bool GetModelKernel::Reset() {
  MS_LOG(INFO) << "Get model kernel reset!";
  StopTimer();
  retry_count_ = 0;
  return true;
}

void GetModelKernel::GetModel(const schema::RequestGetModel *get_model_req, const std::shared_ptr<FBBuilder> &fbb) {
  auto next_req_time = LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp);
  std::map<std::string, AddressPtr> feature_maps;
  size_t current_iter = LocalMetaStore::GetInstance().curr_iter_num();
  size_t get_model_iter = IntToSize(get_model_req->iteration());
  const auto &iter_to_model = ModelStore::GetInstance().iteration_to_model();
  size_t latest_iter_num = iter_to_model.rbegin()->first;
  // If this iteration is not finished yet, return ResponseCode_SucNotReady so that clients could get model later.
  if ((current_iter == get_model_iter && latest_iter_num != current_iter)) {
    std::string reason = "The model is not ready yet for iteration " + std::to_string(get_model_iter) +
                         ". Maybe this is because\n" + "1.Client doesn't send enough update model requests.\n" +
                         "2. Worker has not push all the weights to servers.";
    BuildGetModelRsp(fbb, schema::ResponseCode_SucNotReady, reason, current_iter, feature_maps,
                     std::to_string(next_req_time));
    if (retry_count_.load() % kPrintGetModelForEveryRetryTime == 1) {
      MS_LOG(WARNING) << reason;
    }
    return;
  }

  if (iter_to_model.count(get_model_iter) == 0) {
    // If the model of get_model_iter is not stored, return the latest version of model and current iteration number.
    MS_LOG(WARNING) << "The iteration of GetModel request " << std::to_string(get_model_iter)
                    << " is invalid. Current iteration is " << std::to_string(current_iter);
    feature_maps = ModelStore::GetInstance().GetModelByIterNum(latest_iter_num);
  } else {
    feature_maps = ModelStore::GetInstance().GetModelByIterNum(get_model_iter);
  }

  MS_LOG(INFO) << "GetModel last iteration is valid or not: " << Iteration::GetInstance().is_last_iteration_valid()
               << ", next request time is " << next_req_time << ", current iteration is " << current_iter;
  BuildGetModelRsp(fbb, schema::ResponseCode_SUCCEED, "Get model for iteration " + std::to_string(get_model_iter),
                   current_iter, feature_maps, std::to_string(next_req_time));
  return;
}

void GetModelKernel::BuildGetModelRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                      const std::string &reason, const size_t iter,
                                      const std::map<std::string, AddressPtr> &feature_maps,
                                      const std::string &timestamp) {
  if (fbb == nullptr) {
    MS_LOG(ERROR) << "Input fbb is nullptr.";
    return;
  }
  auto fbs_reason = fbb->CreateString(reason);
  auto fbs_timestamp = fbb->CreateString(timestamp);
  std::vector<flatbuffers::Offset<schema::FeatureMap>> fbs_feature_maps;
  for (const auto &feature_map : feature_maps) {
    auto fbs_weight_fullname = fbb->CreateString(feature_map.first);
    auto fbs_weight_data =
      fbb->CreateVector(reinterpret_cast<float *>(feature_map.second->addr), feature_map.second->size / sizeof(float));
    auto fbs_feature_map = schema::CreateFeatureMap(*(fbb.get()), fbs_weight_fullname, fbs_weight_data);
    fbs_feature_maps.push_back(fbs_feature_map);
  }
  auto fbs_feature_maps_vector = fbb->CreateVector(fbs_feature_maps);

  schema::ResponseGetModelBuilder rsp_get_model_builder(*(fbb.get()));
  rsp_get_model_builder.add_retcode(static_cast<int>(retcode));
  rsp_get_model_builder.add_reason(fbs_reason);
  rsp_get_model_builder.add_iteration(static_cast<int>(iter));
  rsp_get_model_builder.add_feature_map(fbs_feature_maps_vector);
  rsp_get_model_builder.add_timestamp(fbs_timestamp);
  auto rsp_get_model = rsp_get_model_builder.Finish();
  fbb->Finish(rsp_get_model);
  return;
}

REG_ROUND_KERNEL(getModel, GetModelKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
