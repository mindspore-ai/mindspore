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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_BUSERRCODE_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_BUSERRCODE_H

// common err code  -1 ~ -100
constexpr int BUS_ERROR = -1;
constexpr int BUS_OK = 0;
constexpr int COMM_NULL_PTR = -1;
constexpr int ERRORCODE_SUCCESS = 1;

// actor module  err code   -101 ~ -200
constexpr int ACTOR_PARAMER_ERR = -101;
constexpr int ACTOR_NOT_FIND = -102;
constexpr int IO_NOT_FIND = -103;

// TCP module err code -301 ~ -400
// Null
// UDP IO err code  -401 ~ -500
constexpr int UDP_MSG_TOO_BIG = -401;
constexpr int UDP_MSG_WRITE_ERR = -402;
constexpr int UDP_MSG_SEND_ERR = -403;
constexpr int UDP_MSG_ADDR_ERR = -404;
constexpr int UDP_MSG_SEND_SUCCESS = 1;

// Protocol module err code -501 ~ -600
constexpr int PB_MSG_NO_NAME = -501;

#endif
