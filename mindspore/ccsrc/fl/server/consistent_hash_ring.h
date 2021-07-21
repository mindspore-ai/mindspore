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

#ifndef MINDSPORE_CCSRC_FL_SERVER_CONSISTENT_HASH_RING_H_
#define MINDSPORE_CCSRC_FL_SERVER_CONSISTENT_HASH_RING_H_

#include <map>
#include <string>
#include "utils/log_adapter.h"

namespace mindspore {
namespace fl {
namespace server {
constexpr uint32_t kDefaultVirtualNodeNum = 32;
// To support distributed storage and make servers easy to scale-out and scale-in for a large load of metadata in
// server, we use class ConsistentHashRing to help servers find out which metadata is stored in which server node.

// Class ConsistentHashRing implements the algorithm described in the paper
// <https://dl.acm.org/doi/pdf/10.1145/258533.258660>.

// This class will create a ring for hash values of metadata and server nodes. Each server could use this ring to
// retrieve data stored in other servers according to the hash keys. The time complexity for adding/deleting/searching
// of this algorithm is basically O(log n).
class ConsistentHashRing {
 public:
  // The parameter virtual_node_num for constructor means the virtual node number to be created for each physical server
  // node. According to the paper, these virtual nodes could help spread data to all the servers and ensuring balancing
  // at the same time. And when we say "adding/deleting/searching", we are talking about operations on thease virtual
  // nodes instead of the physical nodes.
  explicit ConsistentHashRing(uint32_t virtual_node_num = 128) : virtual_node_num_(virtual_node_num) {}
  ~ConsistentHashRing() = default;

  // Insert several virtual nodes for a server into this ring according to its rank id.
  bool Insert(uint32_t rank);

  // Remove virtual nodes for a server according to its rank id.
  bool Erase(uint32_t rank);

  // Find the physical server node's rank according to the metadata's key.
  uint32_t Find(const std::string &key);

 private:
  uint32_t virtual_node_num_;
  // The hash ring for the server nodes.
  // Key is the hash value of the virtual node.
  // Value is the physical node' rank id.
  std::map<size_t, uint32_t> ring_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_CONSISTENT_HASH_RING_H_
