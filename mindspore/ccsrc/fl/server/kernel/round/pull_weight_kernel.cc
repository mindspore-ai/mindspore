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

#include "fl/server/kernel/round/pull_weight_kernel.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "fl/server/model_store.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void PullWeightKernel::InitKernel(size_t) {
  executor_ = &Executor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor_);
  if (!executor_->initialized()) {
    MS_LOG(EXCEPTION) << "Executor must be initialized in server pipeline.";
    return;
  }
}

bool PullWeightKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &outputs) {
  MS_LOG(DEBUG) << "Launching PullWeightKernel kernel.";
  void *req_data = inputs[0]->addr;
  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    MS_LOG(ERROR) << "FBBuilder builder or req_data is nullptr.";
    return false;
  }

  const schema::RequestPullWeight *pull_weight_req = flatbuffers::GetRoot<schema::RequestPullWeight>(req_data);
  if (pull_weight_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestPullWeight";
    BuildPullWeightRsp(fbb, schema::ResponseCode_RequestError, reason, LocalMetaStore::GetInstance().curr_iter_num(),
                       {});
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  PullWeight(fbb, pull_weight_req);
  GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
  return true;
}

bool PullWeightKernel::Reset() {
  retry_count_ = 0;
  return true;
}

void PullWeightKernel::PullWeight(const std::shared_ptr<FBBuilder> &fbb,
                                  const schema::RequestPullWeight *pull_weight_req) {
  if (fbb == nullptr || pull_weight_req == nullptr) {
    MS_LOG(ERROR) << "fbb or pull_weight_req is nullptr.";
    return;
  }
  std::map<std::string, AddressPtr> feature_maps = {};
  size_t current_iter = LocalMetaStore::GetInstance().curr_iter_num();
  size_t pull_weight_iter = IntToSize(pull_weight_req->iteration());
  // The iteration from worker should be the same as server's, otherwise return SucNotReady so that worker could retry.
  if (pull_weight_iter != current_iter) {
    std::string reason = "PullWeight iteration " + std::to_string(pull_weight_iter) +
                         " is invalid. Server current iteration: " + std::to_string(current_iter);
    BuildPullWeightRsp(fbb, schema::ResponseCode_SucNotReady, reason, current_iter, feature_maps);
    MS_LOG(WARNING) << reason;
    return;
  }

  std::vector<std::string> weight_names = {};
  auto weights_names_fbs = pull_weight_req->weight_names();
  if (weights_names_fbs == nullptr) {
    MS_LOG(ERROR) << "weights_names_fbs is nullptr.";
    return;
  }
  for (uint32_t i = 0; i < weights_names_fbs->size(); i++) {
    weight_names.push_back(weights_names_fbs->Get(i)->str());
  }
  if (!executor_->IsWeightAggrDone(weight_names) || !executor_->unmasked()) {
    (void)++retry_count_;
    std::string reason = "The aggregation for the weights is not done yet.";
    BuildPullWeightRsp(fbb, schema::ResponseCode_SucNotReady, reason, current_iter, feature_maps);
    if (retry_count_.load() % kPrintPullWeightForEveryRetryTime == 1) {
      MS_LOG(WARNING) << reason << " Retry count is " << retry_count_.load();
    }
    return;
  }

  feature_maps = executor_->HandlePullWeight(weight_names);
  if (feature_maps.empty()) {
    std::string reason = "The feature_map is empty for the given weight names.";
    BuildPullWeightRsp(fbb, schema::ResponseCode_RequestError, reason, current_iter, feature_maps);
    MS_LOG(WARNING) << reason;
    return;
  }
  MS_LOG(INFO) << "Pulling weight for iteration " << current_iter << " succeeds.";

  BuildPullWeightRsp(fbb, schema::ResponseCode_SUCCEED,
                     "Pulling weight by weight names for iteration " + std::to_string(pull_weight_iter) + " success.",
                     current_iter, feature_maps);
  return;
}

void PullWeightKernel::BuildPullWeightRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode,
                                          const std::string &reason, size_t iteration,
                                          const std::map<std::string, AddressPtr> &feature_maps) {
  if (fbb == nullptr) {
    MS_LOG(ERROR) << "fbb is nullptr.";
    return;
  }
  auto fbs_reason = fbb->CreateString(reason);
  std::vector<flatbuffers::Offset<schema::FeatureMap>> fbs_feature_maps;
  for (auto feature_map : feature_maps) {
    auto fbs_weight_fullname = fbb->CreateString(feature_map.first);
    auto fbs_weight_data =
      fbb->CreateVector(reinterpret_cast<float *>(feature_map.second->addr), feature_map.second->size / sizeof(float));
    auto fbs_feature_map = schema::CreateFeatureMap(*(fbb.get()), fbs_weight_fullname, fbs_weight_data);
    fbs_feature_maps.push_back(fbs_feature_map);
  }
  auto fbs_feature_maps_vector = fbb->CreateVector(fbs_feature_maps);

  schema::ResponsePullWeightBuilder rsp_pull_weight_builder(*(fbb.get()));
  rsp_pull_weight_builder.add_retcode(static_cast<int>(retcode));
  rsp_pull_weight_builder.add_reason(fbs_reason);
  rsp_pull_weight_builder.add_iteration(SizeToInt(iteration));
  rsp_pull_weight_builder.add_feature_map(fbs_feature_maps_vector);
  auto rsp_pull_weight = rsp_pull_weight_builder.Finish();
  fbb->Finish(rsp_pull_weight);
  return;
}

REG_ROUND_KERNEL(pullWeight, PullWeightKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
