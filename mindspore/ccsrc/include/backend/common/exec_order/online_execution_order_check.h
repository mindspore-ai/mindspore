/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_EXEC_ORDER_ONLINE_EXECUTION_ORDER_CHECK_H_
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_EXEC_ORDER_ONLINE_EXECUTION_ORDER_CHECK_H_

#include <memory>
#include <map>
#include <vector>
#include <future>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include "ir/anf.h"
#include "include/backend/visible.h"
#include "include/backend/common/exec_order/kernel_cache.h"

namespace mindspore {
namespace runtime {
using KernelVariant = std::variant<CNodePtr, CommPyboostKernelPtr>;
class BACKEND_COMMON_EXPORT Process {
 public:
  static Process &GetInstance() {
    static Process instance;
    return instance;
  }

  static const int kPynativeFlag = -1;
  static const size_t kMaxAllGatherBuffSize;
  static const uint64_t kFnvOffsetBasis;
  static const char kCommGroupName[];
  static const char kSendReceive[];

  void CheckCommOrderIteration(size_t total_running_count);

  inline static uint64_t fnv1a_hash_update(uint64_t hash, char c) {
    const uint64_t FNV_PRIME = 1099511628211ULL;
    hash ^= static_cast<uint64_t>(c);
    hash *= FNV_PRIME;
    return hash;
  }

  struct ProcessResult {
    std::unordered_map<std::string, uint64_t> group_hashes;
  };

  void StartCollectExecOrder();

  void StopCollectExecOrder();

 private:
  std::unordered_map<int, std::future<ProcessResult>> async_futures_;
  std::unordered_map<std::string, std::vector<uint32_t>> comm_rank_cache_;
  std::mutex cache_mutex_;

  Process() = default;

  DISABLE_COPY_AND_ASSIGN(Process);

  std::unordered_map<int, ProcessResult> latest_results_;
  ProcessResult pynative_results_;

  uint32_t GetRankSize();
  std::string GetRankID();
  std::string GetGroupFromPrim(const PrimitivePtr &prim);
  std::pair<std::string, std::string> GetKernelShapes(const CNodePtr &kernel);

  void ProcessKernels(int step = kPynativeFlag);

  void ValidateCommGroupExecuteOrders(int step = kPynativeFlag);
  void AllGatherExecuteOrderHash(std::unique_ptr<char[]> *output_host_buffer, int step = kPynativeFlag);
  void ValidateExecuteOrders(const std::map<std::string, std::map<uint64_t, size_t>> &group_execute_order_hash);

  uint64_t accumulate_hash(uint64_t current_hash, const std::string &str);
  void FetchCommRanksCache(const std::string &group_name);
  void ProcessSendReceive(ProcessResult *result, const std::string &group, const KernelVariant &kernel,
                          const std::string &primitive_str, const std::string &inputShape,
                          const std::string &outputShape);
  void ProcessNormalGroupHash(ProcessResult *result, const std::string &group, const std::string &primitive_str,
                              const std::string &inputShape);
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_EXEC_ORDER_ONLINE_EXECUTION_ORDER_CHECK_H_
