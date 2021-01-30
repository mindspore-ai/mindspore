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

#ifndef MINDSPORE_CORE_MINDRT_SRC_ACTOR_SYSMGR_ACTOR_H
#define MINDSPORE_CORE_MINDRT_SRC_ACTOR_SYSMGR_ACTOR_H

#include <queue>

#include "async/async.h"
#include "async/asyncafter.h"
#include "actor/actorapp.h"
#include "utils/log_adapter.h"

namespace mindspore {

const std::string SYSMGR_ACTOR_NAME = "SysMgrActor";
const std::string METRICS_SEND_MSGNAME = "SendMetrics";
const int LINK_RECYCLE_PERIOD_MIN = 20;
const int LINK_RECYCLE_PERIOD_MAX = 360;

using IntTypeMetrics = std::queue<int>;
using StringTypeMetrics = std::queue<std::string>;

class MetricsMessage : public MessageBase {
 public:
  explicit MetricsMessage(const std::string &tfrom, const std::string &tTo, const std::string &tName,
                          const IntTypeMetrics &tInts = IntTypeMetrics(),
                          const StringTypeMetrics &tStrings = StringTypeMetrics())
      : MessageBase(tfrom, tTo, tName), intTypeMetrics(tInts), stringTypeMetrics(tStrings) {}

  ~MetricsMessage() override {}

  void PrintMetrics();

 private:
  IntTypeMetrics intTypeMetrics;
  StringTypeMetrics stringTypeMetrics;
};

class SysMgrActor : public mindspore::AppActor {
 public:
  explicit SysMgrActor(const std::string &name, const Duration &duration)
      : mindspore::AppActor(name), printSendMetricsDuration(duration) {
    linkRecyclePeriod = 0;
  }

  ~SysMgrActor() override {}

 protected:
  virtual void Init() override {
    MS_LOG(INFO) << "Initiaize SysMgrActor";
    // register receive handle
    Receive("SendMetrics", &SysMgrActor::HandleSendMetricsCallback);

    // start sys manager timers
    (void)AsyncAfter(printSendMetricsDuration, GetAID(), &SysMgrActor::SendMetricsDurationCallback);

    char *linkRecycleEnv = getenv("LITEBUS_LINK_RECYCLE_PERIOD");
    if (linkRecycleEnv != nullptr) {
      int period = 0;
      period = std::stoi(linkRecycleEnv);
      if (period >= LINK_RECYCLE_PERIOD_MIN && period <= LINK_RECYCLE_PERIOD_MAX) {
        MS_LOG(INFO) << "link recycle set:" << period;
        linkRecyclePeriod = period;
        (void)AsyncAfter(linkRecycleDuration, GetAID(), &SysMgrActor::LinkRecycleDurationCallback);
      }
    }
  }

 private:
  void SendMetricsDurationCallback();
  void HandleSendMetricsCallback(const AID &from, std::unique_ptr<MetricsMessage> message);
  void LinkRecycleDurationCallback();
  Duration printSendMetricsDuration;
  static Duration linkRecycleDuration;
  int linkRecyclePeriod;
};

}  // namespace mindspore
#endif
