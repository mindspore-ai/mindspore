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

#ifndef MINDSPORE_MAILBOX_H
#define MINDSPORE_MAILBOX_H
#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <utility>
#include "actor/msg.h"
#include "thread/hqueue.h"

namespace mindspore {
class MailBox {
 public:
  virtual ~MailBox() = default;
  virtual int EnqueueMessage(std::unique_ptr<MessageBase> msg) = 0;
  virtual std::list<std::unique_ptr<MessageBase>> *GetMsgs() = 0;
  virtual std::unique_ptr<MessageBase> GetMsg() = 0;
  inline void SetNotifyHook(std::unique_ptr<std::function<void()>> &&hook) { notifyHook = std::move(hook); }
  inline bool TakeAllMsgsEachTime() const { return takeAllMsgsEachTime; }

 protected:
  // if this flag is true, GetMsgs() should be invoked to take all enqueued msgs each time, otherwise we can only get
  // one msg by GetMsg() each time.
  bool takeAllMsgsEachTime = true;
  std::unique_ptr<std::function<void()>> notifyHook;
};

class BlockingMailBox : public MailBox {
 public:
  BlockingMailBox() : mailbox1(), mailbox2(), enqueMailBox(&mailbox1), dequeMailBox(&mailbox2) {}
  virtual ~BlockingMailBox() {
    mailbox1.clear();
    mailbox2.clear();
  }
  int EnqueueMessage(std::unique_ptr<MessageBase> msg) override;
  std::list<std::unique_ptr<MessageBase>> *GetMsgs() override;
  std::unique_ptr<MessageBase> GetMsg() override { return nullptr; }

 private:
  std::list<std::unique_ptr<MessageBase>> mailbox1;
  std::list<std::unique_ptr<MessageBase>> mailbox2;
  std::list<std::unique_ptr<MessageBase>> *enqueMailBox;
  std::list<std::unique_ptr<MessageBase>> *dequeMailBox;
  std::mutex lock;
  std::condition_variable cond;
};

class NonblockingMailBox : public MailBox {
 public:
  NonblockingMailBox() : mailbox1(), mailbox2(), enqueMailBox(&mailbox1), dequeMailBox(&mailbox2) {}
  virtual ~NonblockingMailBox() {
    mailbox1.clear();
    mailbox2.clear();
  }
  int EnqueueMessage(std::unique_ptr<MessageBase> msg) override;
  std::list<std::unique_ptr<MessageBase>> *GetMsgs() override;
  std::unique_ptr<MessageBase> GetMsg() override { return nullptr; }

 private:
  std::list<std::unique_ptr<MessageBase>> mailbox1;
  std::list<std::unique_ptr<MessageBase>> mailbox2;
  std::list<std::unique_ptr<MessageBase>> *enqueMailBox;
  std::list<std::unique_ptr<MessageBase>> *dequeMailBox;
  std::mutex lock;
  bool released_ = true;
};

class HQueMailBox : public MailBox {
 public:
  HQueMailBox() { takeAllMsgsEachTime = false; }
  inline bool Init() { return mailbox.Init(MAX_MSG_QUE_SIZE); }
  int EnqueueMessage(std::unique_ptr<MessageBase> msg) override;
  std::list<std::unique_ptr<MessageBase>> *GetMsgs() override { return nullptr; }
  std::unique_ptr<MessageBase> GetMsg() override;

 private:
  HQueue<MessageBase> mailbox;
  static const int32_t MAX_MSG_QUE_SIZE = 4096;
};
}  // namespace mindspore

#endif  // MINDSPORE_MAILBOX_H
