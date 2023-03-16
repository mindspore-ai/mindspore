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

#include <vector>
#include <gtest/gtest.h>
#include "include/backend/distributed/cluster/topology/compute_graph_node.h"
#include "distributed/cluster/topology/meta_server_node.h"
#include "distributed/persistent/storage/file_io_utils.h"
#include "include/backend/distributed/recovery/recovery_context.h"
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

  size_t total_node_num = 2;
  std::vector<std::shared_ptr<ComputeGraphNode>> cgns;
  MetaServerNode msn("meta_server_node", "scheduler", total_node_num);
  ASSERT_TRUE(msn.Initialize());

  for (size_t i = 0; i < total_node_num; ++i) {
    auto cgn = std::make_shared<ComputeGraphNode>("compute_graph_node_" + std::to_string(i + 1), "worker");
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
  MetaServerNode msn("meta_server_node", "scheduler", total_node_num);
  ASSERT_TRUE(msn.Initialize());

  std::string message_name = "route";
  static std::string received_message;
  auto func =
    std::make_shared<std::function<std::string(const std::string &)>>([](const std::string &message) -> std::string {
      received_message = message;
      return "";
    });
  msn.RegisterMessageHandler(message_name, func);

  ComputeGraphNode cgn("compute_graph_node", "worker");
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
  MetaServerNode msn("meta_server_node", "scheduler", total_node_num);
  ASSERT_TRUE(msn.Initialize());

  std::string message_name = "get_route";
  static std::string received_message = "127.0.0.1::1234";
  auto func = std::make_shared<std::function<std::string(const std::string &)>>(
    [](const std::string &) -> std::string { return received_message; });
  msn.RegisterMessageHandler(message_name, func);

  ComputeGraphNode cgn("compute_graph_node", "worker");
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
  size_t total_node_num = 2;
  MetaServerNode msn("meta_server_node", "scheduler", total_node_num);
  ASSERT_TRUE(msn.Initialize());

  common::SetEnv(kEnvMSRole, "MS_WORKER");
  std::vector<std::shared_ptr<ComputeGraphNode>> cgns;
  for (size_t i = 0; i < total_node_num; ++i) {
    auto cgn = std::make_shared<ComputeGraphNode>("compute_graph_node_" + std::to_string(i + 1), "worker");
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
  MetaServerNode restored_msn("meta_server_node", "scheduler", total_node_num);
  ASSERT_TRUE(restored_msn.Initialize());

  ASSERT_EQ(total_node_num, restored_msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kInitialized, restored_msn.TopologyState());

  restored_msn.Finalize(true);
  remove(full_file_path.c_str());
}

/// Feature: test heartbeat from compute graph node to meta server node is timed out.
/// Description: start a cluster with one meta server node and three compute graph nodes, and then kill one of the
/// compute graph node.
/// Expectation: the number of alive compute graph node is equal to two.
TEST_F(TestDynamicNetworking, HeartbeatTimeout) {
  // Start the meta server node in the parent process.
  std::string server_host = "127.0.0.1";
  std::string server_port = "8090";
  common::SetEnv(kEnvMetaServerHost, server_host.c_str());
  common::SetEnv(kEnvMetaServerPort, server_port.c_str());
  constexpr char kEnvMSRole[] = "MS_ROLE";
  common::SetEnv(kEnvMSRole, "MS_SCHED");

  size_t total_node_num = 2;
  uint64_t timeout = 4;
  MetaServerNode msn("meta_server_node", "scheduler", total_node_num, timeout);
  ASSERT_TRUE(msn.Initialize());

  // Start compute graph nodes in separate sub processes.
  std::vector<pid_t> cgns;
  for (size_t i = 0; i < total_node_num; ++i) {
    pid_t pid = fork();
    EXPECT_LE(0, pid);
    if (pid == 0) {
      // Start the compute graph node in the sub process.
      common::SetEnv(kEnvMSRole, "MS_WORKER");
      common::SetEnv(kEnvMetaServerHost, server_host.c_str());
      common::SetEnv(kEnvMetaServerPort, server_port.c_str());

      auto cgn = std::make_shared<ComputeGraphNode>("compute_graph_node_" + std::to_string(i + 1), "worker");
      ASSERT_TRUE(cgn->Initialize());
      size_t time = 3600;
      sleep(time);
    } else {
      cgns.push_back(pid);
    }
  }

  size_t interval = 1;
  size_t retry = 30;
  while (((msn.GetAliveNodeNum() != total_node_num) || (msn.TopologyState() != TopoState::kInitialized)) &&
         (retry-- > 0)) {
    sleep(interval);
  }

  ASSERT_EQ(total_node_num, msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kInitialized, msn.TopologyState());

  for (size_t i = 0; i < (total_node_num / 2); ++i) {
    kill(cgns[i], 9);
  }
  sleep(timeout + 6);
  ASSERT_EQ(total_node_num - (total_node_num / 2), msn.GetAliveNodeNum());

  // Kill all the processes of compute graph nodes.
  for (size_t i = total_node_num / 2; i < total_node_num; ++i) {
    kill(cgns[i], 9);
  }
  msn.Finalize(true);
}

/// Feature: test reconnect to meta server node if needed during node registration period.
/// Description: first start the compute graph node and then start the meta server node.
/// Expectation: the cluster topology is constructed successfully.
TEST_F(TestDynamicNetworking, ReconnectToMetaServerDuringReg) {
  // Init the environment variables.
  std::string server_host = "127.0.0.1";
  std::string server_port = "8090";
  common::SetEnv(kEnvMetaServerHost, server_host.c_str());
  common::SetEnv(kEnvMetaServerPort, server_port.c_str());

  size_t total_node_num = 2;
  std::vector<pid_t> cgns;

  // Start the compute graph nodes firstly.
  for (size_t i = 0; i < total_node_num; ++i) {
    pid_t pid = fork();
    EXPECT_LE(0, pid);

    if (pid == 0) {
      common::SetEnv(kEnvMetaServerHost, server_host.c_str());
      common::SetEnv(kEnvMetaServerPort, server_port.c_str());
      auto cgn = std::make_shared<ComputeGraphNode>("compute_graph_node_" + std::to_string(i + 1), "worker");
      ASSERT_TRUE(cgn->Initialize());
      while (!cgn->Initialized()) {
        sleep(1);
      }
      sleep(1);
      cgn->Finalize(true);
      return;
    } else {
      cgns.push_back(pid);
    }
  }

  size_t interval = 6;
  sleep(interval);

  // Start the meta server node.
  MetaServerNode msn("meta_server_node", "scheduler", total_node_num);
  ASSERT_TRUE(msn.Initialize());

  // Wait for the cluster to be ready.
  interval = 1;
  size_t retry = 30;
  while (((msn.GetAliveNodeNum() != total_node_num) || (msn.TopologyState() != TopoState::kInitialized)) &&
         (retry-- > 0)) {
    sleep(interval);
  }

  // Validate the state of the cluster.
  ASSERT_EQ(total_node_num, msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kInitialized, msn.TopologyState());

  // Destroy the cluster.
  for (size_t i = 0; i < total_node_num; ++i) {
    kill(cgns[i], 9);
  }
  msn.Finalize(true);
}

/// Feature: test reconnect to meta server node if needed during node unregistration period.
/// Description: start the meta server node and several compute graph nodes, then restart the meta server node after the
/// cluster is initialized successfully.
/// Expectation: the cluster topology is shutdown finally.
TEST_F(TestDynamicNetworking, ReconnectToMetaServerDuringUnreg) {
  // Init the environment variables.
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

  std::string server_host = "127.0.0.1";
  std::string server_port = "8090";
  common::SetEnv(kEnvMetaServerHost, server_host.c_str());
  common::SetEnv(kEnvMetaServerPort, server_port.c_str());

  // Start the meta server node.
  constexpr char kEnvMSRole[] = "MS_ROLE";
  common::SetEnv(kEnvMSRole, "MS_SCHED");
  size_t total_node_num = 1;
  MetaServerNode msn("meta_server_node", "scheduler", total_node_num);
  ASSERT_TRUE(msn.Initialize());

  // Start the compute graph nodes.
  common::SetEnv(kEnvMSRole, "MS_WORKER");
  std::vector<std::shared_ptr<ComputeGraphNode>> cgns;
  for (size_t i = 0; i < total_node_num; ++i) {
    auto cgn = std::make_shared<ComputeGraphNode>("compute_graph_node_" + std::to_string(i + 1), "worker");
    ASSERT_TRUE(cgn->Initialize());
    cgns.push_back(cgn);
  }

  // Wait for the cluster to be initialized.
  size_t interval = 1;
  size_t retry = 30;
  while (((msn.GetAliveNodeNum() != total_node_num) || (msn.TopologyState() != TopoState::kInitialized)) &&
         (retry-- > 0)) {
    sleep(interval);
  }
  ASSERT_EQ(total_node_num, msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kInitialized, msn.TopologyState());

  // Stop the meta server node.
  msn.Finalize(true);

  // Restart the meta server node.
  common::SetEnv(kEnvMSRole, "MS_SCHED");
  MetaServerNode restarted_msn("meta_server_node", "scheduler", total_node_num);
  ASSERT_TRUE(restarted_msn.Initialize());

  // Check if the cluster is recovered successfully.
  while (((restarted_msn.GetAliveNodeNum() != total_node_num) ||
          (restarted_msn.TopologyState() != TopoState::kInitialized)) &&
         (retry-- > 0)) {
    sleep(interval);
  }
  ASSERT_EQ(total_node_num, restarted_msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kInitialized, restarted_msn.TopologyState());

  // Destroy the cluster peacefully.
  for (auto &cgn : cgns) {
    cgn->Finalize();
  }
  retry = 30;
  while ((restarted_msn.GetAliveNodeNum() > 0 || restarted_msn.TopologyState() != TopoState::kFinished) &&
         retry-- > 0) {
    sleep(interval);
  }
  ASSERT_EQ(0, restarted_msn.GetAliveNodeNum());
  ASSERT_EQ(TopoState::kFinished, restarted_msn.TopologyState());
  restarted_msn.Finalize();
}

/// Feature: test get hostnames from meta server node from compute graph node.
/// Description: build a cluster and call the gethostname of compute graph node.
/// Expectation: the hostnames of specified compute graph node are returned.
TEST_F(TestDynamicNetworking, GetHostNames) {
  std::string server_host = "127.0.0.1";
  std::string server_port = "8090";
  common::SetEnv(kEnvMetaServerHost, server_host.c_str());
  common::SetEnv(kEnvMetaServerPort, server_port.c_str());
  common::SetEnv(recovery::kEnvEnableRecovery, "0");

  size_t total_node_num = 3;
  size_t total_node_group_0_num = 2;
  size_t total_node_group_1_num = 1;
  std::string worker_group_0 = "worker_group_0";
  std::string worker_group_1 = "worker_group_1";

  std::vector<std::shared_ptr<ComputeGraphNode>> cgns;
  MetaServerNode msn("meta_server_node", "scheduler", total_node_num);
  ASSERT_TRUE(msn.Initialize());

  constexpr char kEnvMSRole[] = "MS_ROLE";
  common::SetEnv(kEnvMSRole, worker_group_0.c_str());
  for (size_t i = 0; i < total_node_group_0_num; ++i) {
    auto cgn = std::make_shared<ComputeGraphNode>("compute_graph_node_" + std::to_string(i + 1), worker_group_0);
    cgn->set_rank_id(i);
    ASSERT_TRUE(cgn->Initialize());
    cgns.push_back(cgn);
  }

  common::SetEnv(kEnvMSRole, worker_group_1.c_str());
  for (size_t i = total_node_group_0_num; i < total_node_num; ++i) {
    auto cgn = std::make_shared<ComputeGraphNode>("compute_graph_node_" + std::to_string(i + 1), worker_group_1);
    cgn->set_rank_id(i);
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

  auto hostnames_0 = cgns[0]->GetHostNames(worker_group_0);
  auto hostnames_1 = cgns[0]->GetHostNames(worker_group_1);

  ASSERT_EQ(total_node_group_0_num, hostnames_0.size());
  ASSERT_EQ(total_node_group_1_num, hostnames_1.size());

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
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
