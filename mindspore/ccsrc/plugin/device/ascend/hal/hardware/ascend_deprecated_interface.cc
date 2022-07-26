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

#include "plugin/device/ascend/hal/hardware/ascend_deprecated_interface.h"
#include "plugin/device/ascend/hal/device/distribute/ascend_collective.h"
#include "plugin/device/ascend/hal/profiler/parallel_strategy_profiling.h"

namespace mindspore {
namespace device {
namespace ascend {
uint32_t AscendDeprecatedInterface::InitCollective() {
#ifdef WITH_BACKEND
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_LOG(INFO) << "The process are launched with OpenMPI, the environment variable for rank table will be ignored "
                  "even if set by users.";
  MS_LOG(INFO) << "mpi collective init.";
  if (!collective::HcclCollectiveGroup::instance().InitCollective()) {
    MS_LOG(EXCEPTION) << "Mpi init failed, please check if mpirun is used correctly.";
  }
  auto rank_id = collective::HcclCollectiveGroup::instance().GetRankId(kHcclWorldGroup);
  (void)common::SetEnv(kRankID, std::to_string(rank_id).c_str());
  uint32_t device_id = IntToUint(collective::HcclCollectiveGroup::instance().GetDeviceId());
  (void)common::SetEnv("DEVICE_ID", std::to_string(device_id).c_str());
  ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, device_id);
  return device_id;
#else
  return 0;
#endif
}

void AscendDeprecatedInterface::DumpProfileParallelStrategy(const FuncGraphPtr &func_graph) {
  return profiler::ascend::DumpProfileParallelStrategy(func_graph);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
