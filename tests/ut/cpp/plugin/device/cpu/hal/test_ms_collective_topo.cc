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
#include "include/backend/distributed/cluster/topology/compute_graph_node.h"
#include "distributed/cluster/topology/meta_server_node.h"
#include "plugin/device/cpu/hal/hardware/ms_collective_topo.h"
#include "utils/ms_utils.h"
#include "common/common_test.h"

namespace mindspore {
namespace device {
namespace cpu {
class TestMSCollectiveTopo : public UT::Common {
 protected:
  void SetUp() {}
  void TearDown() {}
};

/// Feature: test create cpu collective topology node.
/// Description: create the topology node.
/// Expectation: the topology node is created successfully.
TEST_F(TestMSCollectiveTopo, InitCollectiveTopoNode) {
  std::string server_host = "127.0.0.1";
  std::string server_port = "8090";
  common::SetEnv(distributed::cluster::topology::kEnvMetaServerHost, server_host.c_str());
  common::SetEnv(distributed::cluster::topology::kEnvMetaServerPort, server_port.c_str());

  size_t total_node_num = 8;
  std::vector<std::shared_ptr<distributed::cluster::topology::ComputeGraphNode>> cgns;
  distributed::cluster::topology::MetaServerNode msn("meta_server_node", "scheduler", total_node_num);
  ASSERT_TRUE(msn.Initialize());

  for (size_t i = 0; i < total_node_num; ++i) {
    auto cgn = std::make_shared<distributed::cluster::topology::ComputeGraphNode>(
      "compute_graph_node_" + std::to_string(i + 1), "worker");
    ASSERT_TRUE(cgn->Initialize());
    cgns.push_back(cgn);
  }

  size_t interval = 1;
  size_t retry = 30;
  while (((msn.GetAliveNodeNum() != total_node_num) ||
          (msn.TopologyState() != distributed::cluster::topology::TopoState::kInitialized)) &&
         (retry-- > 0)) {
    sleep(interval);
  }

  ASSERT_EQ(total_node_num, msn.GetAliveNodeNum());
  ASSERT_EQ(distributed::cluster::topology::TopoState::kInitialized, msn.TopologyState());

  // Create the topo nodes.
  std::vector<std::shared_ptr<TopologyNode>> topo_nodes;
  for (size_t i = 0; i < total_node_num; ++i) {
    auto node = std::make_shared<TopologyNode>(total_node_num, cgns[i]);
    topo_nodes.push_back(node);
    node->Initialize();
  }
  for (size_t i = 0; i < total_node_num; ++i) {
    ASSERT_TRUE(topo_nodes[i]->Initialized());
  }

  // Check the rank id of topo node.
  for (size_t i = 0; i < total_node_num; ++i) {
    ASSERT_EQ(i, topo_nodes[i]->rank_id());
  }

  // Test data communication.
  for (size_t i = 0; i < total_node_num; ++i) {
    auto node = topo_nodes[i];
    auto rank_id = node->rank_id();
    auto next_rank_id = (rank_id + 1) % total_node_num;

    std::string data = "model gradients " + std::to_string(rank_id);
    node->SendAsync(next_rank_id, data.data(), data.length());
  }

  // Flush all the sending data.
  for (size_t i = 0; i < total_node_num; ++i) {
    auto node = topo_nodes[i];
    auto rank_id = node->rank_id();
    auto next_rank_id = (rank_id + 1) % total_node_num;
    node->WaitForSend(next_rank_id);
  }

  // Receive data from other rank nodes.
  for (size_t i = 0; i < total_node_num; ++i) {
    auto node = topo_nodes[i];
    auto rank_id = node->rank_id();
    auto upstream_rank_id = (rank_id > 0) ? (rank_id - 1) : (total_node_num - 1);
    MessageBase *message = nullptr;
    node->Receive(upstream_rank_id, &message);

    ASSERT_NE(nullptr, message);
    ASSERT_EQ(std::to_string(upstream_rank_id), message->name);
    ASSERT_EQ("model gradients " + std::to_string(upstream_rank_id), message->body);

    delete message;
    message = nullptr;
  }

  // Destroy the topo nodes.
  for (size_t i = 0; i < total_node_num; ++i) {
    topo_nodes[i]->Finalize();
  }

  for (auto &cgn : cgns) {
    cgn->Finalize();
  }

  retry = 30;
  while ((msn.GetAliveNodeNum() > 0 || msn.TopologyState() != distributed::cluster::topology::TopoState::kFinished) &&
         retry-- > 0) {
    sleep(interval);
  }
  ASSERT_EQ(0, msn.GetAliveNodeNum());
  ASSERT_EQ(distributed::cluster::topology::TopoState::kFinished, msn.TopologyState());

  msn.Finalize();
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
