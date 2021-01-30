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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_LITEBUS_HPP_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_LITEBUS_HPP_H

#include <string>
#include "mindrt/include/actor/actor.h"

// brief provide an asynchronous programming framework as Actor model
namespace mindspore {

struct LitebusAddress {
    std::string scheme;
    std::string ip;
    uint16_t port;
};

// brief initialize the library
int Initialize(const std::string &tcpUrl, const std::string &tcpUrlAdv = "", const std::string &udpUrl = "",
               const std::string &udpUrlAdv = "", int threadCount = 0);

// brief spawn a process to run an actor
AID Spawn(ActorReference actor, bool sharedThread = true, bool start = true);

// brief wait for the actor process to exit . It will be discarded
void Await(const ActorReference &actor);

// brief get actor with aid
ActorReference GetActor(const AID &actor);

// brief wait for the actor process to exit
void Await(const AID &actor);

// brief  Terminate the actor  to exit
void Terminate(const AID &actor);

// brief  set the actor's  running status
void SetActorStatus(const AID &actor, bool start);

// brief  Terminate all actors
void TerminateAll();

// brief terminate the process. It will be discarded
void Finalize();

// brief set the delegate of restful
void SetDelegate(const std::string &delegate);

// set log pid of the process use litebus
void SetLogPID(HARES_LOG_PID pid);

// get global litebus address
const LitebusAddress &GetLitebusAddress();

// get flag of http message format
int GetHttpKmsgFlag();

// brief set flag of http message format
void SetHttpKmsgFlag(int flag);

}    // namespace mindspore
#endif
