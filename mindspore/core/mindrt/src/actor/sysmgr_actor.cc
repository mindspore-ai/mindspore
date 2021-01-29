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

#include <actor/sysmgr_actor.h>
#include "actor/actormgr.h"
#include "actor/iomgr.h"
#include "utils/log_adapter.h"

namespace mindspore {

Duration SysMgrActor::linkRecycleDuration = 10000;

void MetricsMessage::PrintMetrics() {
  // print sendMetrics by default, in the future we can add more metrics format
  std::ostringstream out;

  while (!intTypeMetrics.empty()) {
    out << intTypeMetrics.front() << "-";
    intTypeMetrics.pop();
  }

  out << "|";

  while (!stringTypeMetrics.empty()) {
    std::string stringMetric = stringTypeMetrics.front();
    if (stringMetric.empty()) {
      out << "null"
          << "-";
    } else {
      out << stringMetric << "-";
    }

    stringTypeMetrics.pop();
  }

  MS_LOG(INFO) << "[format:fd-err-sum-size|to-okmsg-failmsg], value:" << out.str().c_str();
}

void SysMgrActor::SendMetricsDurationCallback() {
  std::string protocol = "tcp";
  std::shared_ptr<mindspore::IOMgr> ioMgrRef = ActorMgr::GetIOMgrRef(protocol);
  if (ioMgrRef == nullptr) {
    MS_LOG(INFO) << "tcp protocol is not exist.";
  } else {
    ioMgrRef->CollectMetrics();
  }

  (void)AsyncAfter(printSendMetricsDuration, GetAID(), &SysMgrActor::SendMetricsDurationCallback);
}

void SysMgrActor::HandleSendMetricsCallback(const AID &from, std::unique_ptr<MetricsMessage> message) {
  if (message == nullptr) {
    MS_LOG(WARNING) << "Can't transform to MetricsMessage.";
    return;
  }
  message->PrintMetrics();
  return;
}

void SysMgrActor::LinkRecycleDurationCallback() {
  std::string protocol = "tcp";
  std::shared_ptr<mindspore::IOMgr> ioMgrRef = ActorMgr::GetIOMgrRef(protocol);
  if (ioMgrRef == nullptr) {
    MS_LOG(INFO) << "tcp protocol is not exist.";
  } else {
    ioMgrRef->LinkRecycleCheck(linkRecyclePeriod);
  }

  (void)AsyncAfter(linkRecycleDuration, GetAID(), &SysMgrActor::LinkRecycleDurationCallback);
}

}  // namespace mindspore
