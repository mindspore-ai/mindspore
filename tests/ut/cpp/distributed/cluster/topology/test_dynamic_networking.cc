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
#include "distributed/recovery/recovery_context.h"
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

/// Feature: test the normal node registration from compute graph nodes to meta server node.
/// Description: start some compute graph nodes and meta server node and send a register message.
/// Expectation: these register messages are received by meta server node successfully.
TEST_F(TestDynamicNetworking, NodeRegister) {
  std::string server_host = "127.0.0.1";
  std::string server_port = "8090";
  common::SetEnv(kEnvMetaServerHost, server_host.c_str());
  common::SetEnv(kEnvMetaServerPort, server_port.c_str());

  size_t total_node_num = 8;
  std::vector<std::shared_ptr<ComputeGraphNode>> cgns;
  MetaServerNode msn("meta_server_node", total_node_num);
  ASSERT_TRUE(msn.Initialize());

  for (size_t i = 0; i < total_node_num; ++i) {
    auto cgn = std::make_shared<ComputeGraphNode>("compute_graph_node_" + std::to_string(i + 1));
    ASSERT_TRUE(cgn->Initialize());
    cgns.push_back(cgn);
  }

  size_t interval = 1;
  size_t retry = 30;
  while (((msn.GetAliveNodeNum() != total_node_num) || (msn.TopologyState() != TopoState::kInitialized)) &&
         (retry-- > 0)) {
    sleep(interval);
  }

  ASSERT_EQ(total_node_num, msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kInitialized, msn.TopologyState());

  for (int i = 0; i < total_node_num; ++i) {
    ASSERT_EQ(i, cgns[i]->rank_id());
  }
  ASSERT_EQ(-1, msn.rank_id());

  for (auto &cgn : cgns) {
    cgn->Finalize();
  }

  retry = 30;
  while ((msn.GetAliveNodeNum() > 0 || msn.TopologyState() != TopoState::kFinished) && retry-- > 0) {
    sleep(interval);
  }
  ASSERT_EQ(0, msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kFinished, msn.TopologyState());

  msn.Finalize();
}

/// Feature: test sending message through compute graph node to meta server node.
/// Description: send a special kind of message to msn and register the corresponding message handler.
/// Expectation: the registered handler received the sent message successfully.
TEST_F(TestDynamicNetworking, AddMessageHandler) {
  std::string server_host = "127.0.0.1";
  std::string server_port = "8090";
  common::SetEnv(kEnvMetaServerHost, server_host.c_str());
  common::SetEnv(kEnvMetaServerPort, server_port.c_str());

  size_t total_node_num = 1;
  MetaServerNode msn("meta_server_node", total_node_num);
  ASSERT_TRUE(msn.Initialize());

  std::string message_name = "route";
  static std::string received_message;
  auto func =
    std::make_shared<std::function<std::string(const std::string &)>>([](const std::string &message) -> std::string {
      received_message = message;
      return "";
    });
  msn.RegisterMessageHandler(message_name, func);

  ComputeGraphNode cgn("compute_graph_node");
  ASSERT_TRUE(cgn.Initialize());

  size_t interval = 1;
  size_t retry = 30;
  while (((msn.GetAliveNodeNum() != total_node_num) || (msn.TopologyState() != TopoState::kInitialized)) &&
         (retry-- > 0)) {
    sleep(interval);
  }

  ASSERT_EQ(total_node_num, msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kInitialized, msn.TopologyState());

  std::string message_body = "127.0.0.1:8080";
  ASSERT_TRUE(cgn.SendMessageToMSN(message_name, message_body));

  cgn.Finalize();

  retry = 30;
  while ((msn.GetAliveNodeNum() > 0 || msn.TopologyState() != TopoState::kFinished) && retry-- > 0) {
    sleep(interval);
  }
  ASSERT_EQ(0, msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kFinished, msn.TopologyState());
  ASSERT_EQ(message_body, received_message);

  msn.Finalize();
}

/// Feature: test retrieve message from the meta server node.
/// Description: send a retrieve request to msn.
/// Expectation: get message from msn successfully.
TEST_F(TestDynamicNetworking, RetrieveMessageFromMSN) {
  std::string server_host = "127.0.0.1";
  std::string server_port = "8090";
  common::SetEnv(kEnvMetaServerHost, server_host.c_str());
  common::SetEnv(kEnvMetaServerPort, server_port.c_str());

  size_t total_node_num = 1;
  MetaServerNode msn("meta_server_node", total_node_num);
  ASSERT_TRUE(msn.Initialize());

  std::string message_name = "get_route";
  static std::string received_message = "127.0.0.1::1234";
  auto func = std::make_shared<std::function<std::string(const std::string &)>>(
    [](const std::string &) -> std::string { return received_message; });
  msn.RegisterMessageHandler(message_name, func);

  ComputeGraphNode cgn("compute_graph_node");
  ASSERT_TRUE(cgn.Initialize());

  size_t interval = 1;
  size_t retry = 30;
  while (((msn.GetAliveNodeNum() != total_node_num) || (msn.TopologyState() != TopoState::kInitialized)) &&
         (retry-- > 0)) {
    sleep(interval);
  }

  ASSERT_EQ(total_node_num, msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kInitialized, msn.TopologyState());

  std::shared_ptr<std::string> ret_msg = cgn.RetrieveMessageFromMSN(message_name);

  cgn.Finalize();

  retry = 30;
  while ((msn.GetAliveNodeNum() > 0 || msn.TopologyState() != TopoState::kFinished) && retry-- > 0) {
    sleep(interval);
  }
  ASSERT_EQ(*ret_msg, received_message);
  ASSERT_EQ(TopoState::kFinished, msn.TopologyState());
  ASSERT_EQ(0, msn.GetAliveNodeNum());

  msn.Finalize();
}

/// Feature: test the recovery of meta server node.
/// Description: construct a cluster and restart the meta server node under recovery mode.
/// Expectation: the meta server node is restarted successfully and all the metadata is restored.
TEST_F(TestDynamicNetworking, MetaServerNodeRecovery) {
  // Prepare the environment.
  std::string local_file = "recovery.dat";
  char *dir = getcwd(nullptr, 0);
  EXPECT_NE(nullptr, dir);

  std::string path = dir;
  free(dir);
  dir = nullptr;

  std::string full_file_path = path + "/" + local_file;
  if (storage::FileIOUtils::IsFileOrDirExist(full_file_path)) {
    remove(full_file_path.c_str());
  }
  EXPECT_TRUE(!storage::FileIOUtils::IsFileOrDirExist(full_file_path));
  common::SetEnv(recovery::kEnvEnableRecovery, "1");
  common::SetEnv(recovery::kEnvRecoveryPath, path.c_str());

  // Construct the cluster(meta server node and compute graph node).
  std::string server_host = "127.0.0.1";
  std::string server_port = "8090";
  common::SetEnv(kEnvMetaServerHost, server_host.c_str());
  common::SetEnv(kEnvMetaServerPort, server_port.c_str());

  constexpr char kEnvMSRole[] = "MS_ROLE";
  common::SetEnv(kEnvMSRole, "MS_SCHED");
  size_t total_node_num = 8;
  MetaServerNode msn("meta_server_node", total_node_num);
  ASSERT_TRUE(msn.Initialize());

  common::SetEnv(kEnvMSRole, "MS_WORKER");
  std::vector<std::shared_ptr<ComputeGraphNode>> cgns;
  for (size_t i = 0; i < total_node_num; ++i) {
    auto cgn = std::make_shared<ComputeGraphNode>("compute_graph_node_" + std::to_string(i + 1));
    ASSERT_TRUE(cgn->Initialize());
    cgns.push_back(cgn);
  }

  size_t interval = 1;
  size_t retry = 30;
  while (((msn.GetAliveNodeNum() != total_node_num) || (msn.TopologyState() != TopoState::kInitialized)) &&
         (retry-- > 0)) {
    sleep(interval);
  }

  ASSERT_EQ(total_node_num, msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kInitialized, msn.TopologyState());

  for (int i = 0; i < total_node_num; ++i) {
    ASSERT_EQ(i, cgns[i]->rank_id());
  }
  ASSERT_EQ(-1, msn.rank_id());

  for (auto &cgn : cgns) {
    cgn->Finalize();
  }

  retry = 30;
  while ((msn.GetAliveNodeNum() > 0 || msn.TopologyState() != TopoState::kFinished) && retry-- > 0) {
    sleep(interval);
  }
  ASSERT_EQ(0, msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kFinished, msn.TopologyState());

  msn.Finalize();

  // Restart the meta server node and check if the node is restored successfully.
  common::SetEnv(kEnvMSRole, "MS_SCHED");
  MetaServerNode restored_msn("meta_server_node", total_node_num);
  ASSERT_TRUE(restored_msn.Initialize());

  ASSERT_EQ(total_node_num, restored_msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kInitialized, restored_msn.TopologyState());

  restored_msn.Finalize(true);
  remove(full_file_path.c_str());
}
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
