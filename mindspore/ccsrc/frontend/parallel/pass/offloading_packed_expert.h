/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_OFFLOADING_PACKED_EXPERT_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_OFFLOADING_PACKED_EXPERT_H_

#include <vector>
#include <string>
#include <algorithm>

#include "ir/anf.h"
#include "include/utils/utils.h"
#include "include/utils/parallel_context.h"
#include "frontend/jit/ps/resource.h"
#include "primitive/math_ops.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace parallel {
bool SetOffloadingPackedExpert(const FuncGraphPtr &func_graph);

class OffloadingPackedExpertInfo {
 public:
  OffloadingPackedExpertInfo() {}
  ~OffloadingPackedExpertInfo() = default;
  int64_t GetExpertNum() const { return expert_num_; }
  int64_t GetPackedExpertNum() const { return pe_num_; }
  std::vector<int64_t> GetFrontReorderExpertsIdx() { return front_reorder_experts_idx_; }
  std::vector<int64_t> GetBackReorderExpertsIdx() { return back_reorder_experts_idx_; }

  void DisplayInfo() {
    MS_LOG(INFO) << "expert_num_ " << GetExpertNum();
    MS_LOG(INFO) << "pe_num_ " << GetPackedExpertNum();
    MS_LOG(INFO) << "front_reorder_experts_idx_ " << GetFrontReorderExpertsIdx();
    MS_LOG(INFO) << "back_reorder_experts_idx_ " << GetBackReorderExpertsIdx();
  }

  void SetExpertNumAndPeNum(int64_t expert_num, int64_t pe_num) {
    expert_num_ = expert_num;
    pe_num_ = pe_num;
    front_reorder_experts_idx_.clear();
    back_reorder_experts_idx_.clear();
    SetFrontReorderExpertsIdx();
    SetBackReorderExpertsIdx();
    DisplayInfo();
  }

 private:
  int64_t expert_num_ = 0;
  int64_t pe_num_ = 0;
  std::vector<int64_t> front_reorder_experts_idx_;
  std::vector<int64_t> back_reorder_experts_idx_;

  void FrontReorderIdx(std::vector<int64_t> *v) {
    for (int64_t i = 0; i < pe_num_; i++) {
      for (int64_t j = 0; j < expert_num_ / pe_num_; j++) {
        v->push_back(i + j * pe_num_);
      }
    }
  }
  void SetFrontReorderExpertsIdx() { FrontReorderIdx(&front_reorder_experts_idx_); }

  void BackReorderIdx(std::vector<int64_t> *v) {
    for (int64_t i = 0; i < expert_num_ / pe_num_; i++) {
      for (int64_t j = 0; j < pe_num_; j++) {
        v->push_back(i + j * (expert_num_ / pe_num_));
      }
    }
  }
  void SetBackReorderExpertsIdx() { BackReorderIdx(&back_reorder_experts_idx_); }
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_OFFLOADING_PACKED_EXPERT_H_
