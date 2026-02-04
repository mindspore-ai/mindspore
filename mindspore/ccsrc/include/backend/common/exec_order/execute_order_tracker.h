/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_EXEC_ORDER_EXECUTE_ORDER_TRACKER_H_
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_EXEC_ORDER_EXECUTE_ORDER_TRACKER_H_

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <utility>
#include "include/backend/visible.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "ir/anf.h"

namespace mindspore {

struct OrderInfo {
  std::string index;
  std::string node_name;
  std::string logic_id;
  std::string stream_id;
  std::string node_info;
  std::string event_id;
  std::string group;

  OrderInfo() = default;
};

using OrderInfoPtr = std::shared_ptr<OrderInfo>;

struct CommOrderInfo {
  // Same as index in OrderInfo
  std::string index;
  std::string group;
  std::string comm_rank;
  std::string primitive;
  std::string src_rank;
  std::string dest_rank;
  std::string root_rank;
  std::string input_shape;
  std::string input_type;
  std::string output_shape;
  std::string output_type;
  std::string input_size;
  std::string output_size;

  CommOrderInfo() = default;
};

using CommOrderInfoPtr = std::shared_ptr<CommOrderInfo>;

BACKEND_COMMON_EXPORT bool EnableExecuteOrderDump();

class BACKEND_COMMON_EXPORT ExecuteOrderTracker {
 public:
  static ExecuteOrderTracker &GetInstance();

  void ProcessNode(const CNodePtr &cnode);

  void ProcessPyboostCommOp(const std::shared_ptr<kernel::pyboost::OpRunner> &op, const std::string &group,
                            size_t comm_stream_id, const tensor::TensorPtr &input_tensor,
                            const tensor::TensorPtr &output_tensor, int64_t rank);

  // Data is written to disk and cleaned up
  void Clear();

 private:
  ExecuteOrderTracker() = default;
  ~ExecuteOrderTracker() = default;
  ExecuteOrderTracker(const ExecuteOrderTracker &) = delete;
  ExecuteOrderTracker &operator=(const ExecuteOrderTracker &) = delete;

  void AddOrderInfo(const OrderInfoPtr &order_info);

  void AddCommOrderInfo(const CommOrderInfoPtr &comm_info);

  std::vector<uint32_t> GetCommRanks(const std::string &group_name);

  bool IsCommunicationOp(const CNodePtr &cnode) const;

  CommOrderInfoPtr CreateCommOrderInfo(const std::string &index, const std::string &group,
                                       const std::string &primitive_str, const CNodePtr &cnode,
                                       const tensor::TensorPtr &input_tensor = nullptr,
                                       const tensor::TensorPtr &output_tensor = nullptr, int64_t direct_rank = -1);

  void GetInputOutputShapeAndType(const CNodePtr &cnode, const CommOrderInfoPtr &comm_info) const;

  std::string GetCommunicationRanks(const std::variant<int64_t, std::pair<const CNodePtr &, const char *>> &input,
                                    const std::vector<uint32_t> &comm_ranks) const;

  std::mutex mutex_;
  std::string order_path_;
  std::string comm_order_path_;
  // Operator launch order
  size_t index_counter_{1};
  std::vector<OrderInfoPtr> order_info_list_;
  std::vector<CommOrderInfoPtr> comm_order_info_list_;

  // comm_ranks cache
  std::unordered_map<std::string, std::vector<uint32_t>> comm_ranks_cache_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_EXEC_ORDER_EXECUTE_ORDER_TRACKER_H_
