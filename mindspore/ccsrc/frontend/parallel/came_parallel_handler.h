/**
 * Copyright 2023-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_CAME_PARALLEL_HANDLER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_CAME_PARALLEL_HANDLER_H_

#include <string>
#include <vector>

#include <set>
#include <utility>
#include <memory>
#include <unordered_map>
#include "base/base.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"

namespace mindspore {
namespace parallel {
using TensorLayoutPtr = std::shared_ptr<TensorLayout>;

constexpr size_t kFirstCameReduceMean = 1;
constexpr size_t kSecondCameReduceMean = 2;
constexpr size_t kThirdCameReduceMean = 3;
constexpr size_t kForthCameReduceMean = 4;
constexpr size_t kFifthCameReduceMean = 5;
constexpr size_t kSixthCameReduceMean = 6;
constexpr size_t kSeventhCameReduceMean = 7;
constexpr size_t kParameterDimTwo = 2;

constexpr char EXP_AVG[] = "exp_avg";
constexpr char EXP_AVG_SQ_ROW[] = "exp_avg_sq_row_";
constexpr char EXP_AVG_SQ_COL[] = "exp_avg_sq_col_";
constexpr char EXP_AVG_INSTA_ROW[] = "exp_avg_insta_row_";
constexpr char EXP_AVG_INSTA_COL[] = "exp_avg_insta_col_";
constexpr char EXP_AVG_SQ[] = "exp_avg_sq_";

class CameCommHandler {
 public:
  CameCommHandler(ParameterPtr origin, const std::vector<AnfNodePtr> &all_parameters,
                  const NodeUsersMap &node_user_map);
  void Process();

 private:
  ParameterPtr origin;
  const std::vector<AnfNodePtr> &all_parameters;
  TensorLayoutPtr tensor_layout;
  const NodeUsersMap &node_user_map;

  int64_t cur_rank = -1;
  DeviceMatrix dev_matrix;
  RankList full_rank_list;

  bool is_opt_shard = false;

  ParameterPtr exp_avg_sq_row = nullptr;
  ParameterPtr exp_avg_sq_col = nullptr;
  ParameterPtr exp_avg = nullptr;
  ParameterPtr exp_avg_insta_row = nullptr;
  ParameterPtr exp_avg_insta_col = nullptr;

  std::set<size_t> reduce_mean_numbers = {kFirstCameReduceMean,  kSecondCameReduceMean, kThirdCameReduceMean,
                                          kForthCameReduceMean,  kFifthCameReduceMean,  kSixthCameReduceMean,
                                          kSeventhCameReduceMean};

  void FindCameParams();

  CNodePtr FindReduceMean(size_t number);
  CNodePtr FindReduceMean1256(const ParameterPtr &param);
  CNodePtr FindReduceMean37(const ParameterPtr &param);
  CNodePtr FindReduceMean4();

  std::pair<Status, RankList> GetOptShardRankList(const int64_t rank);
  std::pair<Status, RankList> GetDimRankList(const int64_t rank, const int64_t dim);

  RankList ExpandRankListWithOptShard(const RankList &rank_list);
  RankList ExpandRankListWithDim(const RankList &base, const int64_t dim);

  std::string CreateCommGroupFromRankList(const RankList &rank_list);
  void InsertAllReduceAndRealDivToReduceMeanInput(CNodePtr reduce_mean, const RankList &comm_rank_list);
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_CAME_PARALLEL_HANDLER_H_
