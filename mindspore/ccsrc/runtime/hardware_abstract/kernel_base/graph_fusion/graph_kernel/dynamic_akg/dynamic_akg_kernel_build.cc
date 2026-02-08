/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/dynamic_akg/dynamic_akg_kernel_build.h"

#include <vector>
#include <set>
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_pool.h"

namespace mindspore {
namespace kernel {
constexpr int32_t PROCESS_NUM = 16;
constexpr int32_t TIME_OUT = 300;
constexpr auto platform = "default";

bool DynamicAkgKernelBuilder::SingleOpParallelBuild(const std::vector<AnfNodePtr> &anf_nodes) { return true; }

bool DynamicAkgKernelBuilder::ParallelBuild(const std::vector<JsonNodePair> &build_args) {
  struct timeval start_time;
  struct timeval end_time;
  (void)gettimeofday(&start_time, nullptr);
  MS_LOG(INFO) << "AKG V2 start parallel build. kernel count: " << build_args.size();

  KernelPool kp;
  auto ret = kp.Init(build_args);
  if (ret != 0) {
    MS_LOG(ERROR) << "KernelPool for AKG V2 init failed.";
    return false;
  }

  std::set<size_t> fetched_ids;
  ret = kp.FetchKernels(&fetched_ids);
  if (ret != 0) {
    MS_LOG(ERROR) << "KernelPool for AKG V2 FetchKernels failed.";
    return false;
  }

  if (!fetched_ids.empty()) {
    auto jsons = GetKernelJsonsByHashId(build_args, fetched_ids);

    auto dyn_akg_client = GetClient();
    MS_EXCEPTION_IF_NULL(dyn_akg_client);
    if (!dyn_akg_client->CompilerStart(PROCESS_NUM, TIME_OUT, platform)) {
      MS_LOG(ERROR) << "AKG V2 start failed.";
      return false;
    }
    auto attrs = CollectBuildAttrs();
    if (!attrs.empty() && !dyn_akg_client->CompilerSendAttr(attrs)) {
      MS_LOG(ERROR) << "AKG V2 send attr failed.";
      return false;
    }
    if (!dyn_akg_client->CompilerSendData(jsons)) {
      MS_LOG(ERROR) << "AKG V2 send data failed.";
      return false;
    }
    if (!dyn_akg_client->CompilerWait()) {
      MS_LOG(ERROR) << "AKG V2 compile failed.";
      return false;
    }
  }

  ret = kp.UpdateAndWait(fetched_ids);
  if (ret != 0) {
    MS_LOG(ERROR) << "KernelPool for AKG V2 UpdateAndWait failed.";
    return false;
  }

  if (kp.Release() != 0) {
    MS_LOG(ERROR) << "KernelPool for AKG V2 release failed.";
    return false;
  }

  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "AKG V2 kernel build time: " << cost << " us.";

  return true;
}
}  // namespace kernel
}  // namespace mindspore
