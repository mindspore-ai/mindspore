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
#include <string>
#include "common/common_test.h"
#include "cluster/rpc/core/comm_util.h"
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

/// Feature: socket address parser supports IPv4 localhost.
/// Description: parse a valid IPv4 localhost url with port.
/// Expectation: parsed address family and port are correct.
TEST_F(TestSocketOperation, GetSockAddrWithIPv4Localhost) {
  SocketAddress addr;
  ASSERT_TRUE(SocketOperation::GetSockAddr("127.0.0.1:8080", &addr));
  ASSERT_EQ(addr.saIn.sin_family, AF_INET);
  ASSERT_EQ(ntohs(addr.saIn.sin_port), 8080);
}

/// Feature: socket address parser validates input and null pointer.
/// Description: parse invalid address and null output pointer.
/// Expectation: parser returns false.
TEST_F(TestSocketOperation, GetSockAddrInvalidAddress) {
  SocketAddress addr;
  ASSERT_FALSE(SocketOperation::GetSockAddr("invalid-address:8080", &addr));
  ASSERT_FALSE(SocketOperation::GetSockAddr("127.0.0.1:8080", nullptr));
}

class TestCommUtilIPv6 : public UT::Common {};

/// Feature: GetAvailableInterfaceAndIP prefers IPv6 when ip is empty.
/// Description: Call with empty ip, function tries IPv6 first then falls back to IPv4.
/// Expectation: Returns non-empty interface and ip.
TEST_F(TestCommUtilIPv6, GetAvailableInterfaceAndIPWithEmptyIp) {
  std::string interface;
  std::string ip;
  mindspore::ps::core::CommUtil::GetAvailableInterfaceAndIP(&interface, &ip);
  EXPECT_FALSE(interface.empty());
  EXPECT_FALSE(ip.empty());
}

/// Feature: GetAvailableInterfaceAndIP prefers IPv6 when ip contains ':' (IPv6 format).
/// Description: Call with IPv6-style hint (e.g. "::1"), function prefers IPv6 interface.
/// Expectation: If system has IPv6, returns IPv6 address (contains ':'); else returns IPv4.
TEST_F(TestCommUtilIPv6, GetAvailableInterfaceAndIPWithIPv6Hint) {
  std::string interface;
  std::string ip = "::1";
  mindspore::ps::core::CommUtil::GetAvailableInterfaceAndIP(&interface, &ip);
  EXPECT_FALSE(interface.empty());
  EXPECT_FALSE(ip.empty());
  EXPECT_TRUE(ip.find('.') != std::string::npos || ip.find(':') != std::string::npos);
}

/// Feature: GetAvailableInterfaceAndIP prefers IPv4 when ip is IPv4 format.
/// Description: Call with IPv4-style hint (e.g. "192.168.1.1"), function prefers IPv4 interface only.
/// Expectation: Returns IPv4 address (dotted decimal format).
TEST_F(TestCommUtilIPv6, GetAvailableInterfaceAndIPWithIPv4Hint) {
  std::string interface;
  std::string ip = "192.168.1.1";
  mindspore::ps::core::CommUtil::GetAvailableInterfaceAndIP(&interface, &ip);
  EXPECT_FALSE(interface.empty());
  EXPECT_FALSE(ip.empty());
  EXPECT_NE(ip.find('.'), std::string::npos) << "Expected IPv4 dotted decimal format, got: " << ip;
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
