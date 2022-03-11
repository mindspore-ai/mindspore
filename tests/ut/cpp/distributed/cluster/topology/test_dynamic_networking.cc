/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "distributed/cluster/topology/compute_graph_node.h"
#include "distributed/cluster/topology/meta_server_node.h"
#include "utils/ms_utils.h"
#include "common/common_test.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
// Test the dynamic networking for distributed computation graph execution.
class TestDynamicNetworking : public UT::Common {
 protected:
  void SetUp() {}
  void TearDown() {}
};

/// Feature: test the normal node registration from compute graph node to meta server node.
/// Description: start a compute graph node and meta server node and send a register message.
/// Expectation: the register message is received by meta server node successfully.
TEST_F(TestDynamicNetworking, NodeRegister) {
  std::string server_host = "127.0.0.1";
  std::string server_port = "8090";
  common::SetEnv(kEnvMetaServerHost, server_host.c_str());
  common::SetEnv(kEnvMetaServerPort, server_port.c_str());

  MetaServerNode msn("meta_server_node");
  ASSERT_TRUE(msn.Initialize());

  ComputeGraphNode cgn("compute_graph_node");
  ASSERT_TRUE(cgn.Initialize());

  cgn.Finalize();
  msn.Finalize();
}
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
