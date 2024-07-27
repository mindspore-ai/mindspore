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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_FLASH_SP_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_FLASH_SP_H_

#include <vector>
#include <string>
#include <algorithm>

#include "ir/anf.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/parallel_context.h"
#include "include/backend/optimizer/optimizer.h"
#include "pipeline/jit/ps/resource.h"

namespace mindspore {
namespace parallel {
bool SetFlashSP(const FuncGraphPtr &func_graph);

class FlashSPInfo {
 public:
  explicit FlashSPInfo(CNodePtr fa_score_node);
  ~FlashSPInfo() = default;
  size_t GetSPNum() const { return flashsp_num_; }
  size_t GetRankId() const { return dev_rank_id_; }
  int64_t GetSendRankId() const { return send_rank_id_; }
  int64_t GetRecvRankId() const { return recv_rank_id_; }
  int64_t actual_seq_length_size() const { return actual_seq_length_size_; }

  void DisplayInfo() {
    MS_LOG(DEBUG) << "sp_num_ " << flashsp_num_;
    MS_LOG(DEBUG) << "dev_rank_id_ " << dev_rank_id_;
    MS_LOG(DEBUG) << "send_rank_id_ " << send_rank_id_;
    MS_LOG(DEBUG) << "recv_rank_id_ " << recv_rank_id_;
  }

 private:
  size_t flashsp_num_;
  int64_t send_rank_id_;
  int64_t recv_rank_id_;
  size_t dev_rank_id_;
  int64_t actual_seq_length_size_;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_FLASH_SP_H_
