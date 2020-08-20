/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_COMMON_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_COMMON_H_

/// \note This header file contains common header files and some inlines used by
/// both client and server side codes. Do not put code that is not common here.
/// There are client and server specific header files.

// On platform like Windows, we may support only tcp/ip clients
#if !defined(_WIN32) && !defined(_WIN64)
#define CACHE_LOCAL_CLIENT 1
#endif

#ifdef ENABLE_CACHE
#include <grpcpp/grpcpp.h>
#endif
#include <string>
#ifdef ENABLE_CACHE
#include "proto/cache_grpc.grpc.pb.h"
#endif
#include "proto/cache_grpc.pb.h"
#include "minddata/dataset/engine/cache/cache_request.h"
#include "minddata/dataset/engine/cache/de_tensor_generated.h"
namespace mindspore {
namespace dataset {
/// \brief CacheRow and BatchFetch requests will switch to use shared memory method (if supported
/// on the platform) when the amount of bytes sent is greater than the following number.
/// For too small amount, we won't get any benefit using shared memory method because we need
/// two rpc requests to use shared memory method.
constexpr static int32_t kLocalByPassThreshold = 64 * 1024;
/// \brief A flag used by the BatchFetch request (client side) if it can support local bypass
constexpr static uint32_t kLocalClientSupport = 1;
/// \brief A flag used by CacheRow request (client side) and BatchFetch (server side) reply to indicate if the data is
/// inline in the protobuf. This also implies kLocalClientSupport is also true.
constexpr static uint32_t kDataIsInSharedMemory = 2;
/// \brief Size of each message used in message queue.
constexpr static int32_t kSharedMessageSize = 2048;

/// \brief Convert a Status object into a protobuf
/// \param rc[in] Status object
/// \param reply[in/out] pointer to pre-allocated protobuf object
inline void Status2CacheReply(const Status &rc, CacheReply *reply) {
  reply->set_rc(static_cast<int32_t>(rc.get_code()));
  reply->set_msg(rc.ToString());
}
/// \brief Generate the unix socket file we use on both client/server side given a tcp/ip port number
/// \param port
/// \return unix socket url
inline std::string PortToUnixSocketPath(int port) { return "/tmp/cache_server_p" + std::to_string(port); }
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_COMMON_H_
