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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_MSG_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_MSG_H

#include <utility>
#include <string>

#include "actor/aid.h"

namespace mindspore {
class ActorBase;
class MessageBase {
 public:
  enum class Type : char {
    KMSG = 1,
    KUDP,
    KHTTP,
    KASYNC,
    KLOCAL,
    KEXIT,
    KTERMINATE,
  };

  explicit MessageBase(Type eType = Type::KMSG) : from(), name(), data(nullptr), size(0), type(eType), func_id_(0) {}

  explicit MessageBase(const std::string &sName, Type eType = Type::KMSG)
      : from(), name(sName), data(nullptr), size(0), type(eType), func_id_(0) {}

  explicit MessageBase(const AID &aFrom, const AID &aTo, Type eType = Type::KMSG)
      : from(aFrom), to(aTo), name(), body(), data(nullptr), size(0), type(eType), func_id_(0) {}

  explicit MessageBase(const AID &aFrom, const AID &aTo, const std::string &sName, Type eType = Type::KMSG)
      : from(aFrom), to(aTo), name(sName), body(), data(nullptr), size(0), type(eType), func_id_(0) {}

  explicit MessageBase(const AID &aFrom, const AID &aTo, const std::string &sName, std::string &&sBody,
                       Type eType = Type::KMSG)
      : from(aFrom), to(aTo), name(sName), body(std::move(sBody)), data(nullptr), size(0), type(eType), func_id_(0) {}

  virtual ~MessageBase() {}

  inline std::string &Name() { return name; }

  inline void SetName(const std::string &aName) { this->name = aName; }

  inline AID &From() { return from; }

  inline std::string &Body() { return body; }

  inline void SetFrom(const AID &aFrom) { from = aFrom; }

  inline AID &To() { return to; }

  inline void SetTo(const AID &aTo) { to = aTo; }

  inline Type GetType() const { return type; }

  inline void SetType(Type eType) { type = eType; }

  virtual void Run(ActorBase *actor) {}

  friend class ActorBase;
  friend class TCPMgr;
  AID from;
  AID to;
  std::string name;
  std::string body;

  // The raw bytes of data to be sent and the length of data.
  void *data;
  size_t size;

  Type type;

  // The id of remote function to call.
  uint32_t func_id_;
};
}  // namespace mindspore

#endif
