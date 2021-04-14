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

#include "async/async.h"
#include "actor/actormgr.h"

namespace mindspore {
class MessageAsync : public MessageBase {
 public:
  explicit MessageAsync(std::unique_ptr<MessageHandler> h)
      : MessageBase("Async", Type::KASYNC), handler(std::move(h)) {}

  ~MessageAsync() override {}

  void Run(ActorBase *actor) override { (*handler)(actor); }

 private:
  std::unique_ptr<MessageHandler> handler;
};

void Async(const AID &aid, std::unique_ptr<std::function<void(ActorBase *)>> handler) {
  std::unique_ptr<MessageAsync> msg(new (std::nothrow) MessageAsync(std::move(handler)));
  BUS_OOM_EXIT(msg);
  (void)ActorMgr::GetActorMgrRef()->Send(aid, std::move(msg));
}
}  // namespace mindspore
