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

#include "include/common/utils/comm_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "hccl/hcom.h"
#include "utils/ms_context.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/utils.h"
#include "include/backend/distributed/init.h"

namespace mindspore {
namespace {
class AscendCommManager : public CommManager {
 public:
  AscendCommManager() : CommManager("hccl") {}
  ~AscendCommManager() override = default;

  bool CreateGroupSync(const string &group, const std::vector<unsigned int> &rank_id_list) const override {
    return distributed::collective::CollectiveManager::instance()->CreateCommunicationGroup(group, rank_id_list);
  }

  bool GetRankID(const string &group, unsigned int *rank_id) const override {
    MS_EXCEPTION_IF_NULL(rank_id);
    *rank_id = distributed::collective::CollectiveManager::instance()->GetRankId(group);
    return true;
  }

  bool GetRankSize(const string &group, unsigned int *rank_size) const override {
    MS_EXCEPTION_IF_NULL(rank_size);
    *rank_size = distributed::collective::CollectiveManager::instance()->GetGroupSize(group);
    return true;
  }

  bool DestroyGroup(const string &group) const override {
    return distributed::collective::CollectiveManager::instance()->DestroyCommunicationGroup(group);
  }

  uint32_t GetRank() override {
    uint32_t rank_id = 0;
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto parallel_context = parallel::ParallelContext::GetInstance();
    MS_EXCEPTION_IF_NULL(parallel_context);
    if (parallel_context->parallel_mode() != parallel::kStandalone) {
      // Check HCCL inited.
      if (!hccl::HcclAdapter::GetInstance().Inited()) {
        MS_LOG(DEBUG) << "HCCL not inited, return rank_id = 0";
        return rank_id;
      }
      if (!GetRankID(kHcclWorldGroup, &rank_id)) {
        MS_LOG(EXCEPTION) << "Get rank id failed.";
      }
    }
    return rank_id;
  }
};
COMM_MANAGER_REG(kAscendDevice, std::make_shared<AscendCommManager>());
}  // namespace
}  // namespace mindspore
