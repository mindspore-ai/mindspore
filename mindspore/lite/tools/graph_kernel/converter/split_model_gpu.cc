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
#include "tools/graph_kernel/converter/split_model_gpu.h"
#include <memory>
#include "utils/ms_context.h"

namespace mindspore::graphkernel::inner {
SPLIT_MODEL_REGISTER(kGPUDevice, SplitModelGpu);
constexpr size_t kReduceFusionDepth = 20;
constexpr size_t kBroadcastFusionDepth = 20;

void SplitModelGpu::InitFusePatterns() {
  AddPattern(std::make_shared<FuseVirtualNode>(), true);
  AddPattern(std::make_shared<FuseReshape>(), true);
  AddPattern(FuseElemwiseFwd::CreateDepthMatcher(), true);
  AddPattern(FuseElemwiseFwd::CreateWidthMatcher(), true);
  AddPattern(FuseElemwiseBroadcastFwd::CreateDepthMatcher(), true);
  AddPattern(FuseElemwiseBroadcastFwd::CreateWidthMatcher(), true);
  AddPattern(FuseReduceFwd::CreateDepthMatcher(kReduceFusionDepth), true);
  AddPattern(FuseReduceFwd::CreateWidthMatcher(kReduceFusionDepth), true);
  AddPattern(FuseElemwiseBroadcastBwd::CreateDepthMatcher(kBroadcastFusionDepth), true);
  AddPattern(FuseElemwiseBroadcastBwd::CreateWidthMatcher(kBroadcastFusionDepth), true);
  AddPattern(std::make_shared<FuseIsolateReshape>(), true);
}

AreaMode SplitModelGpu::GetDefaultAreaMode(const PrimOpPtr &) const { return AreaMode::COMPOSITE; }
}  // namespace mindspore::graphkernel::inner
