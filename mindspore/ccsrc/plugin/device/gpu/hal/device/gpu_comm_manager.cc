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
#include "include/common/utils/convert_utils.h"
#include "utils/ms_context.h"
#include "include/common/utils/parallel_context.h"
#include "plugin/device/gpu/hal/device/distribution/collective_init.h"

using CollectiveInitializer = mindspore::device::gpu::CollectiveInitializer;
using CreateCommGroupFunc = mindspore::device::gpu::CreateCommGroupFunc;
using GetRankIDByGroupFunc = mindspore::device::gpu::GetRankIDByGroupFunc;
using GetGroupSizeFunc = mindspore::device::gpu::GetGroupSizeFunc;
using DestroyGroupFunc = mindspore::device::gpu::DestroyGroupFunc;

namespace mindspore {
namespace {
constexpr char kNcclWorldGroup[] = "nccl_world_group";

class GpuCommManager : public CommManager {
 public:
  GpuCommManager() : CommManager("nccl") {}
  ~GpuCommManager() override = default;

  bool CreateGroupSync(const std::string &group, const std::vector<unsigned int> &rank_id_list) const override {
    bool ret = CollectiveInitializer::instance().CreateCommunicationGroup(group, rank_id_list);
    if (!ret) {
      MS_LOG(ERROR) << "Failed to create group " << group << " for rank id list " << rank_id_list;
      return ret;
    }

    MS_LOG(INFO) << "Successfully create group " << group << " for rank id list " << rank_id_list;
    return ret;
  }

  bool GetRankID(const std::string &group, unsigned int *rank_id) const override {
    *rank_id = CollectiveInitializer::instance().GetRankIDByGroup(group);
    MS_LOG(INFO) << "This process rank id is " << *rank_id << " in group " << group;
    return true;
  }

  bool GetRankSize(const std::string &group, unsigned int *rank_size) const override {
    *rank_size = CollectiveInitializer::instance().GetGroupSize(group);
    MS_LOG(INFO) << "Group " << group << " size is " << *rank_size;
    return true;
  }

  bool DestroyGroup(const std::string &group) const override {
    bool ret = CollectiveInitializer::instance().DestroyCommunicationGroup(group);
    if (!ret) {
      MS_LOG(ERROR) << "Failed to destroy group " << group;
      return ret;
    }

    MS_LOG(INFO) << "Successfully destroy group " << group;
    return ret;
  }

  uint32_t GetRank() override {
    uint32_t rank_id = 0;
    auto parallel_context = parallel::ParallelContext::GetInstance();
    MS_EXCEPTION_IF_NULL(parallel_context);
    if (parallel_context->parallel_mode() != parallel::kStandalone) {
      // Check NCCL inited.
      if (!CollectiveInitializer::instance().collective_inited()) {
        MS_LOG(DEBUG) << "NCLL not inited, return rank_id = 0";
        return rank_id;
      }
      if (!GetRankID(kNcclWorldGroup, &rank_id)) {
        MS_LOG(EXCEPTION) << "Get rank id failed.";
      }
    }
    return rank_id;
  }
};
COMM_MANAGER_REG(kGPUDevice, std::make_shared<GpuCommManager>());
}  // namespace
}  // namespace mindspore
