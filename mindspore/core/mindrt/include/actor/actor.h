/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "thread/hqueue.h"
#include "actor/msg.h"
#include "actor/mailbox.h"

namespace mindspore {
class ActorMgr;
class ActorWorker;
class ActorThreadPool;

// should be at least greater than 1
constexpr uint32_t MAX_ACTOR_RECORD_SIZE = 3;

class MS_CORE_API ActorBase {
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
      MS_LOG(DEBUG) << "Actor message dumps:"
                    << "actor:" << id.Name().c_str() << " msg:" << msgRecords[startPoint].c_str();
      startPoint = (startPoint + MAX_ACTOR_RECORD_SIZE - 1) % MAX_ACTOR_RECORD_SIZE;
    }
  }

  ActorBase();
  explicit ActorBase(const std::string &name);
  explicit ActorBase(const std::string &name, ActorThreadPool *pool);
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

  void set_thread_pool(ActorThreadPool *pool) { pool_ = pool; }

  // Judge if actor running by the received message number, the default is true.
  virtual bool IsActive(int msg_num) { return true; }

  inline void set_actor_mgr(const std::shared_ptr<ActorMgr> &mgr) { actor_mgr_ = mgr; }
  inline std::shared_ptr<ActorMgr> get_actor_mgr() const { return actor_mgr_; }

 protected:
  using ActorFunction = std::function<void(const std::unique_ptr<MessageBase> &msg)>;

  // install KMSG handler . This method will be called before the actor start to run.
  virtual void Init() {}

  // This method will be called before the actor start to terminate.
  virtual void Finalize() {}

  // KHTTPMsg handler
  virtual void HandleHttp(const std::unique_ptr<MessageBase> &msg) {
    MS_LOG(ERROR) << "ACTOR (" << id.Name().c_str() << ") HandleHttp() is not implemented";
  }

  // KLOCALMsg handler
  virtual void HandleLocalMsg(const std::unique_ptr<MessageBase> &msg) {
    MS_LOG(ERROR) << "ACTOR (" << id.Name().c_str() << ") HandleLocalMsg() is not implemented.";
  }

  // The link is closed.
  virtual void Exited(const AID &actor) {
    MS_LOG(ERROR) << "ACTOR (" << id.Name().c_str() << ") Exited() is not implemented. ";
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
  friend class ActorWorker;
  friend class ParallelWorker;

  // KMSG Msg Handler
  virtual void HandlekMsg(const std::unique_ptr<MessageBase> &msg);

  template <typename T>
  static void BehaviorBase(T *t, void (T::*method)(const mindspore::AID &, std::string &&, std::string &&),
                           const std::unique_ptr<MessageBase> &msg) {
    MINDRT_OOM_EXIT(msg);
    if (msg->type != MessageBase::Type::KMSG) {
      MS_LOG(ERROR) << "Drop non-tcp message: from:" << std::string(msg->from).c_str()
                    << ",to:" << std::string(msg->to).c_str() << ",name:" << msg->name.c_str();
      return;
    }
    (t->*method)(msg->from, std::move(msg->name), std::move(msg->body));
  }

  // register the message handle. It will be discarded.
  template <typename T>
  static void BehaviorBase1(T *t, void (T::*method)(mindspore::AID, std::string &&, std::string &&),
                            const std::unique_ptr<MessageBase> &msg) {
    MINDRT_OOM_EXIT(msg);
    if (msg->type != MessageBase::Type::KMSG) {
      MS_LOG(ERROR) << "Drop non-tcp message:  from:" << std::string(msg->from).c_str()
                    << ",to:" << std::string(msg->to).c_str() << ",name:" << msg->name.c_str();
      return;
    }
    (t->*method)(msg->from, std::move(msg->name), std::move(msg->body));
  }

  // register the udp message handle. Use this closure function to drop non-udp messages
  template <typename T>
  static void BehaviorBaseForUdp(T *t, void (T::*method)(const mindspore::AID &, std::string &&, std::string &&),
                                 const std::unique_ptr<MessageBase> &msg) {
    MINDRT_OOM_EXIT(msg);
    if (msg->type != MessageBase::Type::KUDP) {
      MS_LOG(ERROR) << "Drop non-udp message:  from:" << std::string(msg->from).c_str()
                    << ",to:" << std::string(msg->to).c_str() << ",name:" << msg->name.c_str();
      return;
    }
    (t->*method)(msg->from, std::move(msg->name), std::move(msg->body));
  }

  void Run();
  void Quit();
  int EnqueMessage(std::unique_ptr<MessageBase> msg) const;

  void Spawn(const std::shared_ptr<ActorBase>, std::unique_ptr<MailBox> mailbox);

  std::unique_ptr<MailBox> mailbox;
  std::atomic_bool terminating_{false};

  AID id;
  std::map<std::string, ActorFunction> actionFunctions;
  std::mutex waiterLock;
  std::string msgRecords[MAX_ACTOR_RECORD_SIZE];
  uint32_t recordNextPoint = 0;

  ActorThreadPool *pool_{nullptr};
  std::shared_ptr<ActorMgr> actor_mgr_;
};
using ActorReference = std::shared_ptr<ActorBase>;
};  // namespace mindspore
#endif
