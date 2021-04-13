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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_ACTOR_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_ACTOR_H

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include "actor/msg.h"

namespace mindspore {
class ActorBase;
class ActorMgr;
class ActorPolicy;

using ActorReference = std::shared_ptr<ActorBase>;

// should be at least greater than 1
constexpr uint32_t MAX_ACTOR_RECORD_SIZE = 3;

class ActorBase {
 public:
  inline const AID &GetAID() const { return id; }

  inline void AddMsgRecord(const std::string &msgName) {
    recordNextPoint++;
    uint32_t startPoint = recordNextPoint % MAX_ACTOR_RECORD_SIZE;
    msgRecords[startPoint] = msgName;
  }

  inline void PrintMsgRecord() {
    uint32_t startPoint = recordNextPoint % MAX_ACTOR_RECORD_SIZE;
    for (uint32_t i = 0; i < MAX_ACTOR_RECORD_SIZE; i++) {
      ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "Actor message dumps:%s",
                          "actor:%s,msg:%s", id.Name().c_str(), msgRecords[startPoint].c_str());
      startPoint = (startPoint + MAX_ACTOR_RECORD_SIZE - 1) % MAX_ACTOR_RECORD_SIZE;
    }
  }

  explicit ActorBase(const std::string &name);
  virtual ~ActorBase();

  // send  MessageBase message to the  actor.
  int Send(const AID &to, std::unique_ptr<MessageBase> msg);

  // send string message to the actor
  int Send(const AID &to, std::string &&name, std::string &&msg, bool remoteLink = false,
           bool isExactNotRemote = false);

  // get output buffer size for flow control
  uint64_t GetOutBufSize(const AID &to);

  // get input buffer size for flow control
  uint64_t GetInBufSize(const AID &to);

  // set record send/receive message package size
  int AddRuleUdp(const std::string &peer, int recordNum);

  // delete the send/receive message package size
  void DelRuleUdp(const std::string &peer, bool outputLog);

 protected:
  using ActorFunction = std::function<void(std::unique_ptr<MessageBase> &msg)>;

  // install KMSG handler . This method will be called before the actor start to run.
  virtual void Init() {}

  // This method will be called before the actor start to terminate.
  virtual void Finalize() {}

  // KHTTPMsg handler
  virtual void HandleHttp(std::unique_ptr<MessageBase> msg) {
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG,
                        "ACTOR (%s) HandleHttp() is not implemented", "a=%s", id.Name().c_str());
  }

  // KLOCALMsg handler
  virtual void HandleLocalMsg(std::unique_ptr<MessageBase> msg) {
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG,
                        "ACTOR (%s) HandleLocalMsg() is not implemented.", "a=%s", id.Name().c_str());
  }

  // The link is closed.
  virtual void Exited(const AID &actor) {
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG,
                        "ACTOR (%s) Exited() is not implemented. ", "a=%s", id.Name().c_str());
  }

  // Filter the KMSG
  virtual bool Filter(const std::unique_ptr<MessageBase> &msg) { return false; }

  // register the message handle
  void Receive(const std::string &msgName, ActorFunction &&func);

  // register the message handle. It will be discarded.
  template <typename T>
  void Receive(const std::string &msgName, void (T::*method)(mindspore::AID, std::string &&, std::string &&)) {
    ActorFunction func = std::bind(&BehaviorBase1<T>, static_cast<T *>(this), method, std::placeholders::_1);
    Receive(msgName, std::move(func));
  }

  // register the message handle
  template <typename T>
  void Receive(const std::string &msgName, void (T::*method)(const mindspore::AID &, std::string &&, std::string &&)) {
    ActorFunction func = std::bind(&BehaviorBase<T>, static_cast<T *>(this), method, std::placeholders::_1);
    Receive(msgName, std::move(func));
    return;
  }

  // register the message handle, for kmsg-udp message
  template <typename T>
  void ReceiveUdp(const std::string &msgName,
                  void (T::*method)(const mindspore::AID &, std::string &&, std::string &&)) {
    ActorFunction func = std::bind(&BehaviorBaseForUdp<T>, static_cast<T *>(this), method, std::placeholders::_1);
    Receive(msgName, std::move(func));
    return;
  }

  // Link the remote actor
  int Link(const AID &to);

  // Unlink the remote actor
  int UnLink(const AID &to);

  // Reconnect to the remote actor
  int Reconnect(const AID &to);

  void Terminate();
  void Await();

 private:
  friend class ActorMgr;
  friend class ActorThread;

  // KMSG Msg Handler
  virtual void HandlekMsg(std::unique_ptr<MessageBase> &msg);

  template <typename T>
  static void BehaviorBase(T *t, void (T::*method)(const mindspore::AID &, std::string &&, std::string &&),
                           std::unique_ptr<MessageBase> &msg) {
    if (msg->type != MessageBase::Type::KMSG) {
      ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "Drop non-tcp message: %s",
                          "from:%s,to:%s,name:%s", std::string(msg->from).c_str(), std::string(msg->to).c_str(),
                          msg->name.c_str());
      return;
    }
    (t->*method)(msg->from, std::move(msg->name), std::move(msg->body));
  }

  // register the message handle. It will be discarded.
  template <typename T>
  static void BehaviorBase1(T *t, void (T::*method)(mindspore::AID, std::string &&, std::string &&),
                            std::unique_ptr<MessageBase> &msg) {
    if (msg->type != MessageBase::Type::KMSG) {
      ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "Drop non-tcp message: %s",
                          "from:%s,to:%s,name:%s", std::string(msg->from).c_str(), std::string(msg->to).c_str(),
                          msg->name.c_str());
      return;
    }
    (t->*method)(msg->from, std::move(msg->name), std::move(msg->body));
  }

  // register the udp message handle. Use this closure function to drop non-udp messages
  template <typename T>
  static void BehaviorBaseForUdp(T *t, void (T::*method)(const mindspore::AID &, std::string &&, std::string &&),
                                 std::unique_ptr<MessageBase> &msg) {
    if (msg->type != MessageBase::Type::KUDP) {
      ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "Drop non-udp message: %s",
                          "from:%s,to:%s,name:%s", std::string(msg->from).c_str(), std::string(msg->to).c_str(),
                          msg->name.c_str());
      return;
    }
    (t->*method)(msg->from, std::move(msg->name), std::move(msg->body));
  }

  void Run();
  void Quit();
  int EnqueMessage(std::unique_ptr<MessageBase> msg);

  void Spawn(std::shared_ptr<ActorBase> &actor, std::unique_ptr<ActorPolicy> actorThread);
  void SetRunningStatus(bool start);

  std::unique_ptr<ActorPolicy> actorThread;

  AID id;
  std::map<std::string, ActorFunction> actionFunctions;
  std::mutex waiterLock;

  std::string msgRecords[MAX_ACTOR_RECORD_SIZE];
  uint32_t recordNextPoint = 0;
};
};  // namespace mindspore
#endif
