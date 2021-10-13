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

#include "fl/server/kernel/round/reconstruct_secrets_kernel.h"
#include <string>
#include <vector>
#include <memory>

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void ReconstructSecretsKernel::InitKernel(size_t) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }

  auto last_cnt_handler = [&](std::shared_ptr<ps::core::MessageHandler>) {
    if (ps::PSContext::instance()->resetter_round() == ps::ResetterRound::kReconstructSeccrets) {
      MS_LOG(INFO) << "start FinishIteration";
      FinishIteration();
      MS_LOG(INFO) << "end FinishIteration";
    }
    return;
  };
  auto first_cnt_handler = [&](std::shared_ptr<ps::core::MessageHandler>) { return; };
  name_unmask_ = "UnMaskKernel";
  MS_LOG(INFO) << "ReconstructSecretsKernel Init, ITERATION NUMBER IS : "
               << LocalMetaStore::GetInstance().curr_iter_num();
  DistributedCountService::GetInstance().RegisterCounter(name_unmask_, ps::PSContext::instance()->initial_server_num(),
                                                         {first_cnt_handler, last_cnt_handler});
}

bool ReconstructSecretsKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                      const std::vector<AddressPtr> &outputs) {
  bool response = false;
  size_t iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  size_t total_duration = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  MS_LOG(INFO) << "Iteration number is " << iter_num << ", ReconstructSecretsKernel total duration is "
               << total_duration;
  clock_t start_time = clock();

  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(ERROR) << "ReconstructSecretsKernel needs 1 input, but got " << inputs.size();
    return false;
  }

  std::shared_ptr<server::FBBuilder> fbb = std::make_shared<server::FBBuilder>();
  void *req_data = inputs[0]->addr;

  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return false;
  }

  // get client list from memory server.
  std::vector<string> update_model_clients;
  const PBMetadata update_model_clients_pb_out =
    DistributedMetadataStore::GetInstance().GetMetadata(kCtxUpdateModelClientList);
  const UpdateModelClientList &update_model_clients_pb = update_model_clients_pb_out.client_list();

  for (int i = 0; i < update_model_clients_pb.fl_id_size(); ++i) {
    update_model_clients.push_back(update_model_clients_pb.fl_id(i));
  }

  const schema::SendReconstructSecret *reconstruct_secret_req =
    flatbuffers::GetRoot<schema::SendReconstructSecret>(req_data);
  std::string fl_id = reconstruct_secret_req->fl_id()->str();

  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(ERROR) << "Current amount for ReconstructSecretsKernel is enough.";
    if (find(update_model_clients.begin(), update_model_clients.end(), fl_id) != update_model_clients.end()) {
      // client in get update model client list.
      cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_SUCCEED,
                                                     "Current amount for ReconstructSecretsKernel is enough.",
                                                     SizeToInt(iter_num), std::to_string(CURRENT_TIME_MILLI.count()));
    } else {
      cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_OutOfTime,
                                                     "Current amount for ReconstructSecretsKernel is enough.",
                                                     SizeToInt(iter_num), std::to_string(CURRENT_TIME_MILLI.count()));
    }
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  response = cipher_reconstruct_.ReconstructSecrets(SizeToInt(iter_num), std::to_string(CURRENT_TIME_MILLI.count()),
                                                    reconstruct_secret_req, fbb, update_model_clients);
  if (response) {
    (void)DistributedCountService::GetInstance().Count(name_, reconstruct_secret_req->fl_id()->str());
  }
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(INFO) << "Current amount for ReconstructSecretsKernel is enough.";
  }
  GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
  clock_t end_time = clock();
  double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
  MS_LOG(INFO) << "reconstruct_secrets_kernel success time is : " << duration;
  if (!response) {
    MS_LOG(INFO) << "reconstruct_secrets_kernel response is false.";
  }
  return true;
}

void ReconstructSecretsKernel::OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &) {
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << LocalMetaStore::GetInstance().curr_iter_num();
  if (ps::PSContext::instance()->encrypt_type() == ps::kPWEncryptType) {
    int sleep_time = 5;
    while (!Executor::GetInstance().IsAllWeightAggregationDone()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }
    MS_LOG(INFO) << "start unmask";
    while (!Executor::GetInstance().Unmask()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }
    MS_LOG(INFO) << "end unmask";
    Executor::GetInstance().set_unmasked(true);
    std::string worker_id = std::to_string(DistributedCountService::GetInstance().local_rank());
    (void)DistributedCountService::GetInstance().Count(name_unmask_, worker_id);
  }
}

bool ReconstructSecretsKernel::Reset() {
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "reconstruct secrets kernel reset!";
  DistributedCountService::GetInstance().ResetCounter(name_);
  DistributedCountService::GetInstance().ResetCounter(name_unmask_);
  StopTimer();
  Executor::GetInstance().set_unmasked(false);
  cipher_reconstruct_.ClearReconstructSecrets();
  return true;
}

REG_ROUND_KERNEL(reconstructSecrets, ReconstructSecretsKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
