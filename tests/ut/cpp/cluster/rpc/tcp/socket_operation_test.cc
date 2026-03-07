/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include <gtest/gtest.h>
#include "common/common_test.h"
#include "cluster/rpc/tcp/socket_operation.h"

namespace mindspore {
namespace distributed {
namespace rpc {
class TestSocketOperation : public UT::Common {};

/// Feature: socket address parser supports ipv6 loopback.
/// Description: parse a valid ipv6 loopback url with port.
/// Expectation: parsed address family and port are correct.
TEST_F(TestSocketOperation, GetSockAddrWithIPv6Loopback) {
  SocketAddress addr;
  ASSERT_TRUE(SocketOperation::GetSockAddr("::1:8080", &addr));
  ASSERT_EQ(addr.saIn6.sin6_family, AF_INET6);
  ASSERT_EQ(ntohs(addr.saIn6.sin6_port), 8080);
}

/// Feature: socket address parser supports ipv6 link-local address.
/// Description: parse a valid ipv6 link-local url with port.
/// Expectation: parsed address family and port are correct.
TEST_F(TestSocketOperation, GetSockAddrWithIPv6LinkLocal) {
  SocketAddress addr;
  ASSERT_TRUE(SocketOperation::GetSockAddr("fe80::1:9090", &addr));
  ASSERT_EQ(addr.saIn6.sin6_family, AF_INET6);
  ASSERT_EQ(ntohs(addr.saIn6.sin6_port), 9090);
}

/// Feature: socket address parser validates input and null pointer.
/// Description: parse invalid address and null output pointer.
/// Expectation: parser returns false.
TEST_F(TestSocketOperation, GetSockAddrInvalidAddress) {
  SocketAddress addr;
  ASSERT_FALSE(SocketOperation::GetSockAddr("invalid-address:8080", &addr));
  ASSERT_FALSE(SocketOperation::GetSockAddr("127.0.0.1:8080", nullptr));
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
