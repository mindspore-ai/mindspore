/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_OTHER_OP_NAME_H_
#define MINDSPORE_CORE_BASE_OTHER_OP_NAME_H_

namespace mindspore {
// spectral
constexpr auto kBartlettWindowOpName = "BartlettWindow";

// Inner op for fall back
constexpr auto kInnerAbsOpName = "inner_abs";

constexpr auto kBlackmanWindowOpName = "BlackmanWindow";
constexpr auto kFusedPullWeightOpName = "FusedPullWeight";
constexpr auto kFusedPushWeightOpName = "FusedPushWeight";
constexpr auto kHammingWindowOpName = "HammingWindow";
constexpr auto kInitDatasetQueueOpName = "InitDataSetQueue";
constexpr auto kLabelGotoOpName = "LabelGoto";
constexpr auto kLabelSetOpName = "LabelSet";
constexpr auto kLabelSwitchOpName = "LabelSwitch";
constexpr auto kMirrorOperatorOpName = "_MirrorOperator";
constexpr auto kNPUAllocFloatStatusOpName = "NPUAllocFloatStatus";
constexpr auto kNPUClearFloatStatusOpName = "NPUClearFloatStatus";
constexpr auto kNPUGetFloatStatusOpName = "NPUGetFloatStatus";
constexpr auto kNPUClearFloatStatusV2OpName = "NPUClearFloatStatusV2";
constexpr auto kNPUGetFloatStatusV2OpName = "NPUGetFloatStatusV2";
constexpr auto kQueueDataOpName = "QueueData";
constexpr auto kReservoirReplayBufferCreateOpName = "ReservoirReplayBufferCreate";
constexpr auto kReservoirReplayBufferDestroyOpName = "ReservoirReplayBufferDestroy";
constexpr auto kReservoirReplayBufferPushOpName = "ReservoirReplayBufferPush";
constexpr auto kReservoirReplayBufferSampleOpName = "ReservoirReplayBufferSample";
constexpr auto kAllGatherOpName = "AllGather";
constexpr auto kAllReduceOpName = "AllReduce";
constexpr auto kReduceOpName = "Reduce";
constexpr auto kReduceScatterOpName = "ReduceScatter";
constexpr auto kAllToAllvOpName = "AllToAllv";
constexpr auto kBarrierOpName = "Barrier";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_OTHER_OP_NAME_H_
