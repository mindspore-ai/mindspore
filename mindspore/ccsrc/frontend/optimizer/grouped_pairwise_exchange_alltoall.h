/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_GROUPED_PAIRWISE_EXCHANGE_ALLTOALL_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_GROUPED_PAIRWISE_EXCHANGE_ALLTOALL_H_

#include <vector>
#include <string>
#include <algorithm>

#include "ir/anf.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/parallel_context.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace opt {
void SetGroupedPairwiseExchangeAllToAll(const pipeline::ResourcePtr &resource);
size_t GetDeviceNum();
size_t GetGlobalRankID();

class GroupedPairwiseExchangeAllToAllInfo {
 public:
  GroupedPairwiseExchangeAllToAllInfo() {
    SetGroupNum();
    SetRanksPerGroup();
    SetGroupRank();
    SetSendGroupRanks();
    SetSortedInputsIdx();
    SetTotalSendRankIds();
    SetTotalRecvRankIds();
    SetReshapeScaleAxisVec();
  }
  ~GroupedPairwiseExchangeAllToAllInfo() = default;
  int64_t GetGroupNum() const { return gpea_num_; }
  int64_t GetRanksPerGroup() const { return ranks_per_group_; }
  int64_t GetGroupRank() const { return group_rank_; }
  std::vector<int64_t> GetSendRankIds(int64_t step_id) { return total_send_rank_ids_[step_id]; }
  std::vector<int64_t> GetRecvRankIds(int64_t step_id) { return total_recv_rank_ids_[step_id]; }
  std::vector<int64_t> GetSendGroupRanks() { return send_group_ranks_; }
  std::vector<int64_t> GetSortedInputsIdx() { return sorted_inputs_idx_; }
  std::vector<uint32_t> GetReshapeScaleAxisVec() { return reshape_scale_axis_vec_; }

  void DisplayInfo() {
    MS_LOG(DEBUG) << "gpea_num_ " << GetGroupNum();
    MS_LOG(DEBUG) << "ranks_per_group_ " << GetRanksPerGroup();
    MS_LOG(DEBUG) << "group_rank_ " << GetGroupRank();
    MS_LOG(DEBUG) << "send_group_ranks_ " << GetSendGroupRanks();
    MS_LOG(DEBUG) << "sorted_inputs_idx_ " << GetSortedInputsIdx();
    for (int64_t step = 0; step < GetGroupNum(); step++) {
      MS_LOG(DEBUG) << "step " << step << " recv_rank_ids " << GetRecvRankIds(step);
      MS_LOG(DEBUG) << "step " << step << " send_rank_ids " << GetSendRankIds(step);
    }
    MS_LOG(DEBUG) << "reshape_scale_axis_vec_ " << GetReshapeScaleAxisVec();
  }

 private:
  int64_t gpea_num_;
  int64_t ranks_per_group_;
  int64_t group_rank_;
  std::vector<std::vector<int64_t>> total_send_rank_ids_;
  std::vector<std::vector<int64_t>> total_recv_rank_ids_;
  std::vector<int64_t> send_group_ranks_;
  std::vector<int64_t> sorted_inputs_idx_;
  std::vector<uint32_t> reshape_scale_axis_vec_;

  void SetGroupNum() {
    // for example, env['GPEA_NUM'] = "1"
    std::string gpea_num_str = common::GetEnv("GPEA_NUM");
    gpea_num_ = 1;
    if (!gpea_num_str.empty()) {
      const int decimal = 10;
      gpea_num_ = std::strtol(gpea_num_str.c_str(), nullptr, decimal);
    }
  }

  void SetRanksPerGroup() { ranks_per_group_ = SizeToLong(GetDeviceNum()) / gpea_num_; }

  void SetGroupRank() { group_rank_ = SizeToLong(GetGlobalRankID()) / ranks_per_group_; }

  void SetTotalSendRankIds() {
    for (int64_t step = 0; step < gpea_num_; step++) {
      std::vector<int64_t> curr_rank_ids;
      int64_t send_group_id = (group_rank_ + step) % gpea_num_;
      for (int64_t i = 0; i < ranks_per_group_; i++) {
        curr_rank_ids.push_back(send_group_id * ranks_per_group_ + i);
      }
      total_send_rank_ids_.push_back(curr_rank_ids);
    }
  }

  void SetTotalRecvRankIds() {
    for (int64_t step = 0; step < gpea_num_; step++) {
      std::vector<int64_t> curr_rank_ids;
      int64_t recv_group_id = (group_rank_ - step + gpea_num_) % gpea_num_;
      for (int64_t i = 0; i < ranks_per_group_; i++) {
        curr_rank_ids.push_back(recv_group_id * ranks_per_group_ + i);
      }
      total_recv_rank_ids_.push_back(curr_rank_ids);
    }
  }

  void SetSendGroupRanks() {
    for (int64_t step = 0; step < gpea_num_; step++) {
      int64_t curr_group_rank = (group_rank_ + step) % gpea_num_;
      send_group_ranks_.push_back(curr_group_rank);
    }
  }

  template <typename T>
  std::vector<int64_t> sort_indexes(const std::vector<T> &v) const {
    std::vector<int64_t> idx(v.size());
    for (size_t i = 0; i < idx.size(); ++i) {
      idx[i] = SizeToLong(i);
    }
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
  }

  void SetSortedInputsIdx() { sorted_inputs_idx_ = sort_indexes(send_group_ranks_); }

  void SetReshapeScaleAxisVec() {
    // for example, env['GPEA_RESHAPE_SCALE_AXIS'] = "2,1"
    std::string reshape_scale_axis_str = common::GetEnv("GPEA_RESHAPE_SCALE_AXIS");
    if (reshape_scale_axis_str.empty()) {
      reshape_scale_axis_vec_ = {kIndex2, kIndex1};
      return;
    }

    std::string value_str;
    std::vector<std::string> result_str_vec;
    for (size_t i = 0; i < reshape_scale_axis_str.size(); ++i) {
      if (reshape_scale_axis_str[i] == ',') {
        result_str_vec.push_back(value_str);
        value_str.clear();
      } else {
        value_str += reshape_scale_axis_str[i];
      }
    }
    if (reshape_scale_axis_str.back() != ',') {
      result_str_vec.push_back(value_str);
    }

    for (size_t i = 0; i < result_str_vec.size(); i++) {
      reshape_scale_axis_vec_.push_back(atoi(result_str_vec[i].c_str()));
    }
    return;
  }
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_GROUPED_PAIRWISE_EXCHANGE_ALLTOALL_H_
