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

#include <string>
#include "fl/server/kernel/round/push_metrics_kernel.h"
#include "fl/server/iteration.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void PushMetricsKernel::InitKernel(size_t) { local_rank_ = DistributedCountService::GetInstance().local_rank(); }

bool PushMetricsKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                               const std::vector<AddressPtr> &outputs) {
  MS_LOG(INFO) << "Launching PushMetricsKernel kernel.";
  void *req_data = inputs[0]->addr;
  std::shared_ptr<FBBuilder> fbb = std::make_shared<FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, reason.c_str(), reason.size());
    return true;
  }

  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t *>(req_data), inputs[0]->size);
  if (!verifier.VerifyBuffer<schema::RequestPushMetrics>()) {
    std::string reason = "The schema of RequestPushMetrics is invalid.";
    BuildPushMetricsRsp(fbb, schema::ResponseCode_RequestError);
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  const schema::RequestPushMetrics *push_metrics_req = flatbuffers::GetRoot<schema::RequestPushMetrics>(req_data);
  if (push_metrics_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestPushMetrics";
    BuildPushMetricsRsp(fbb, schema::ResponseCode_RequestError);
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return false;
  }

  ResultCode result_code = PushMetrics(fbb, push_metrics_req);
  GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
  return ConvertResultCode(result_code);
}

bool PushMetricsKernel::Reset() {
  MS_LOG(INFO) << "PushMetricsKernel reset!";
  StopTimer();
  DistributedCountService::GetInstance().ResetCounter(name_);
  return true;
}

void PushMetricsKernel::OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &) {
  if (ps::PSContext::instance()->resetter_round() == ps::ResetterRound::kPushMetrics) {
    FinishIteration();
  }
  return;
}

ResultCode PushMetricsKernel::PushMetrics(const std::shared_ptr<FBBuilder> &fbb,
                                          const schema::RequestPushMetrics *push_metrics_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(fbb, ResultCode::kSuccessAndReturn);
  MS_ERROR_IF_NULL_W_RET_VAL(push_metrics_req, ResultCode::kSuccessAndReturn);

  float loss = push_metrics_req->loss();
  float accuracy = push_metrics_req->accuracy();
  Iteration::GetInstance().set_loss(loss);
  Iteration::GetInstance().set_accuracy(accuracy);

  std::string count_reason = "";
  if (!DistributedCountService::GetInstance().Count(name_, std::to_string(local_rank_), &count_reason)) {
    std::string reason = "Count for push metrics request failed.";
    BuildPushMetricsRsp(fbb, schema::ResponseCode_SystemError);
    MS_LOG(ERROR) << reason;
    return count_reason == kNetworkError ? ResultCode::kFail : ResultCode::kSuccessAndReturn;
  }

  BuildPushMetricsRsp(fbb, schema::ResponseCode_SUCCEED);
  return ResultCode::kSuccess;
}

void PushMetricsKernel::BuildPushMetricsRsp(const std::shared_ptr<FBBuilder> &fbb, const schema::ResponseCode retcode) {
  MS_ERROR_IF_NULL_WO_RET_VAL(fbb);
  schema::ResponsePushMetricsBuilder rsp_push_metrics_builder(*(fbb.get()));
  rsp_push_metrics_builder.add_retcode(static_cast<int>(retcode));
  auto rsp_push_metrics = rsp_push_metrics_builder.Finish();
  fbb->Finish(rsp_push_metrics);
}

REG_ROUND_KERNEL(pushMetrics, PushMetricsKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
