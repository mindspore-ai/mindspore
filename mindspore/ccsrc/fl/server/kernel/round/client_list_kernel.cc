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

#include "fl/server/kernel/round/client_list_kernel.h"
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include "schema/cipher_generated.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void ClientListKernel::InitKernel(size_t) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }

  executor_ = &Executor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor_);
  if (!executor_->initialized()) {
    MS_LOG(EXCEPTION) << "Executor must be initialized in server pipeline.";
    return;
  }
  cipher_init_ = &armour::CipherInit::GetInstance();
}

bool ClientListKernel::DealClient(const size_t iter_num, const schema::GetClientList *get_clients_req,
                                  std::shared_ptr<server::FBBuilder> fbb) {
  bool response = false;
  std::vector<string> client_list;
  std::string fl_id = get_clients_req->fl_id()->str();
  int32_t iter_client = (size_t)get_clients_req->iteration();
  if (iter_num != (size_t)iter_client) {
    MS_LOG(ERROR) << "ClientListKernel iteration invalid. servertime is " << iter_num;
    MS_LOG(ERROR) << "ClientListKernel iteration invalid. clienttime is " << iter_client;
    BuildClientListRsp(fbb, schema::ResponseCode_OutOfTime, "iter num is error.", client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
  } else {
    if (LocalMetaStore::GetInstance().has_value(kCtxUpdateModelThld)) {
      uint64_t update_model_client_num = LocalMetaStore::GetInstance().value<uint64_t>(kCtxUpdateModelThld);
      PBMetadata client_list_pb_out = DistributedMetadataStore::GetInstance().GetMetadata(kCtxUpdateModelClientList);
      const UpdateModelClientList &client_list_pb = client_list_pb_out.client_list();
      for (int i = 0; i < client_list_pb.fl_id_size(); ++i) {
        client_list.push_back(client_list_pb.fl_id(i));
      }
      if (find(client_list.begin(), client_list.end(), fl_id) != client_list.end()) {  // client in client_list.
        if (static_cast<uint64_t>(client_list_pb.fl_id_size()) >= update_model_client_num) {
          MS_LOG(INFO) << "send clients_list succeed!";
          MS_LOG(INFO) << "UpdateModel client list: ";
          for (size_t i = 0; i < client_list.size(); ++i) {
            MS_LOG(INFO) << " fl_id : " << client_list[i];
          }
          MS_LOG(INFO) << "update_model_client_num: " << update_model_client_num;
          BuildClientListRsp(fbb, schema::ResponseCode_SUCCEED, "send clients_list succeed!", client_list,
                             std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
          response = true;
        } else {
          MS_LOG(INFO) << "The server is not ready. update_model_client_need_num: " << update_model_client_num;
          MS_LOG(INFO) << "now update_model_client_num: " << client_list_pb.fl_id_size();
          /*for (size_t i = 0; i < std::min(client_list.size(), size_t(2)); ++i) {
            MS_LOG(INFO) << " client_list fl_id : " << client_list[i];
          }
          for (size_t i = client_list.size() - size_t(1); i > std::max(client_list.size() - size_t(2), size_t(0));
               --i) {
            MS_LOG(INFO) << " client_list fl_id : " << client_list[i];
          }*/
          int count_tmp = 0;
          for (size_t i = 0; i < cipher_init_->get_model_num_need_; ++i) {
            size_t j = 0;
            for (; j < client_list.size(); ++j) {
              if (("f" + std::to_string(i)) == client_list[j]) break;
            }
            if (j >= client_list.size()) {
              count_tmp++;
              MS_LOG(INFO) << " no client_list fl_id : " << i;
              if (count_tmp > 3) break;
            }
          }
          BuildClientListRsp(fbb, schema::ResponseCode_SucNotReady, "The server is not ready.", client_list,
                             std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
        }
      }
      if (response) {
        DistributedCountService::GetInstance().Count(name_, get_clients_req->fl_id()->str());
      }
    } else {
      MS_LOG(ERROR) << "update_model_client_num is zero.";
      BuildClientListRsp(fbb, schema::ResponseCode_SystemError, "update_model_client_num is zero.", client_list,
                         std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    }
  }
  return response;
}
bool ClientListKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                              const std::vector<AddressPtr> &outputs) {
  std::shared_ptr<server::FBBuilder> fbb = std::make_shared<server::FBBuilder>();
  size_t iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  size_t total_duration = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  MS_LOG(INFO) << "Iteration number is " << iter_num << ", ClientListKernel total duration is " << total_duration;
  clock_t start_time = clock();

  std::vector<string> client_list;
  if (inputs.size() != 1) {
    MS_LOG(ERROR) << "ClientListKernel needs 1 input,but got " << inputs.size();
    BuildClientListRsp(fbb, schema::ResponseCode_SystemError, "ClientListKernel input num not match", client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
  } else if (outputs.size() != 1) {
    MS_LOG(ERROR) << "ClientListKernel needs 1 output,but got " << outputs.size();
    BuildClientListRsp(fbb, schema::ResponseCode_SystemError, "ClientListKernel output num not match", client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
  } else {
    if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
      MS_LOG(ERROR) << "Current amount for GetClientList is enough.";
      BuildClientListRsp(fbb, schema::ResponseCode_OutOfTime, "ClientListKernel num is enough", client_list,
                         std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    } else {
      void *req_data = inputs[0]->addr;
      const schema::GetClientList *get_clients_req = flatbuffers::GetRoot<schema::GetClientList>(req_data);

      if (get_clients_req == nullptr || fbb == nullptr) {
        MS_LOG(ERROR) << "GetClientList is nullptr or ClientListRsp builder is nullptr.";
        BuildClientListRsp(fbb, schema::ResponseCode_RequestError,
                           "GetClientList is nullptr or ClientListRsp builder is nullptr.", client_list,
                           std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
      } else {
        DealClient(iter_num, get_clients_req, fbb);
      }
    }
  }

  GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
  clock_t end_time = clock();
  double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
  MS_LOG(INFO) << "client_list_kernel success time is : " << duration;
  return true;
}  // namespace fl

bool ClientListKernel::Reset() {
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "Get Client list kernel reset!";
  DistributedCountService::GetInstance().ResetCounter(name_);
  StopTimer();
  return true;
}

void ClientListKernel::BuildClientListRsp(std::shared_ptr<server::FBBuilder> client_list_resp_builder,
                                          const schema::ResponseCode retcode, const string &reason,
                                          std::vector<std::string> clients, const string &next_req_time,
                                          const int iteration) {
  auto rsp_reason = client_list_resp_builder->CreateString(reason);
  auto rsp_next_req_time = client_list_resp_builder->CreateString(next_req_time);
  if (clients.size() > 0) {
    std::vector<flatbuffers::Offset<flatbuffers::String>> clients_vector;
    for (auto client : clients) {
      auto client_fb = client_list_resp_builder->CreateString(client);
      clients_vector.push_back(client_fb);
    }
    auto clients_fb = client_list_resp_builder->CreateVector(clients_vector);
    schema::ReturnClientListBuilder rsp_builder(*(client_list_resp_builder.get()));
    rsp_builder.add_retcode(retcode);
    rsp_builder.add_reason(rsp_reason);
    rsp_builder.add_clients(clients_fb);
    rsp_builder.add_iteration(iteration);
    rsp_builder.add_next_req_time(rsp_next_req_time);
    auto rsp_exchange_keys = rsp_builder.Finish();
    client_list_resp_builder->Finish(rsp_exchange_keys);
  } else {
    schema::ReturnClientListBuilder rsp_builder(*(client_list_resp_builder.get()));
    rsp_builder.add_retcode(retcode);
    rsp_builder.add_reason(rsp_reason);
    rsp_builder.add_iteration(iteration);
    rsp_builder.add_next_req_time(rsp_next_req_time);
    auto rsp_exchange_keys = rsp_builder.Finish();
    client_list_resp_builder->Finish(rsp_exchange_keys);
  }
  return;
}

REG_ROUND_KERNEL(getClientList, ClientListKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
