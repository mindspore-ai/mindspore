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

#ifndef MINDSPORE_CORE_BASE_OTHER_OPS_H_
#define MINDSPORE_CORE_BASE_OTHER_OPS_H_

#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/other_op_name.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
GVAR_DEF(PrimitivePtr, kPrimInnerAbs, std::make_shared<Primitive>(kInnerAbsOpName));
GVAR_DEF(PrimitivePtr, kPrimInnerRound, std::make_shared<Primitive>("inner_round"));
GVAR_DEF(PrimitivePtr, kPrimDynamicLossScale, std::make_shared<Primitive>("_DynamicLossScale"));
GVAR_DEF(PrimitivePtr, kPrimScaleGrad, std::make_shared<Primitive>("ScaleGrad"));
GVAR_DEF(PrimitivePtr, kPrimPopulationCount, std::make_shared<Primitive>("PopulationCount"));
GVAR_DEF(PrimitivePtr, kPrimOpaquePredicate, std::make_shared<Primitive>("OpaquePredicate"));

// spectral
GVAR_DEF(PrimitivePtr, kPrimBartlettWindow, std::make_shared<Primitive>(kBartlettWindowOpName));
GVAR_DEF(PrimitivePtr, kPrimBlackmanWindow, std::make_shared<Primitive>("BlackmanWindow"));
GVAR_DEF(PrimitivePtr, kPrimHammingWindow, std::make_shared<Primitive>("HammingWindow"));

// Label
GVAR_DEF(PrimitivePtr, kPrimLabelGoto, std::make_shared<Primitive>("LabelGoto"));
GVAR_DEF(PrimitivePtr, kPrimLabelSwitch, std::make_shared<Primitive>("LabelSwitch"));
GVAR_DEF(PrimitivePtr, kPrimLabelSet, std::make_shared<Primitive>("LabelSet"));

// Comm ops
GVAR_DEF(PrimitivePtr, kPrimMirror, std::make_shared<Primitive>("_MirrorOperator"));
GVAR_DEF(PrimitivePtr, kPrimMirrorMiniStep, std::make_shared<Primitive>("_MirrorMiniStepOperator"));
GVAR_DEF(PrimitivePtr, kPrimMiniStepAllGather, std::make_shared<Primitive>("_MiniStepAllGather"));
GVAR_DEF(PrimitivePtr, kPrimMicroStepAllGather, std::make_shared<Primitive>("_MicroStepAllGather"));
GVAR_DEF(PrimitivePtr, kPrimVirtualDiv, std::make_shared<Primitive>("_VirtualDiv"));
GVAR_DEF(PrimitivePtr, kPrimVirtualAdd, std::make_shared<Primitive>("_VirtualAdd"));
GVAR_DEF(PrimitivePtr, kPrimVirtualDataset, std::make_shared<Primitive>("_VirtualDataset"));
GVAR_DEF(PrimitivePtr, kPrimVirtualOutput, std::make_shared<Primitive>("_VirtualOutput"));
GVAR_DEF(PrimitivePtr, kPrimAllReduce, std::make_shared<Primitive>("AllReduce"));
GVAR_DEF(PrimitivePtr, kPrimReduce, std::make_shared<Primitive>("Reduce"));
GVAR_DEF(PrimitivePtr, kPrimCollectiveScatter, std::make_shared<Primitive>("CollectiveScatter"));
GVAR_DEF(PrimitivePtr, kPrimCollectiveGather, std::make_shared<Primitive>("CollectiveGather"));
GVAR_DEF(PrimitivePtr, kPrimNeighborExchange, std::make_shared<Primitive>("NeighborExchange"));
GVAR_DEF(PrimitivePtr, kPrimNeighborExchangeV2, std::make_shared<Primitive>("NeighborExchangeV2"));
GVAR_DEF(PrimitivePtr, kPrimNeighborExchangeV2Grad, std::make_shared<Primitive>("NeighborExchangeV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimAllToAll, std::make_shared<Primitive>("AlltoAll"));
GVAR_DEF(PrimitivePtr, kPrimAllToAllv, std::make_shared<Primitive>("AllToAllv"));
GVAR_DEF(PrimitivePtr, kPrimAllGather, std::make_shared<Primitive>("AllGather"));
GVAR_DEF(PrimitivePtr, kPrimAllSwap, std::make_shared<Primitive>("_AllSwap"));
GVAR_DEF(PrimitivePtr, kPrimReduceScatter, std::make_shared<Primitive>("ReduceScatter"));
GVAR_DEF(PrimitivePtr, kPrimBarrier, std::make_shared<Primitive>("Barrier"));
GVAR_DEF(PrimitivePtr, kPrimFusedPushWeight, std::make_shared<Primitive>("FusedPushWeight"));
GVAR_DEF(PrimitivePtr, kPrimFusedPullWeight, std::make_shared<Primitive>("FusedPullWeight"));
GVAR_DEF(PrimitivePtr, kPrimInitDataSetQueue, std::make_shared<Primitive>("InitDataSetQueue"));
GVAR_DEF(PrimitivePtr, kPrimQueueData, std::make_shared<Primitive>("QueueData"));
GVAR_DEF(PrimitivePtr, kPrimVirtualAssignAdd, std::make_shared<Primitive>("_VirtualAssignAdd"));
GVAR_DEF(PrimitivePtr, kPrimVirtualAccuGrad, std::make_shared<Primitive>("_VirtualAccuGrad"));
GVAR_DEF(PrimitivePtr, kPrimVirtualPipelineEnd, std::make_shared<Primitive>("_VirtualPipelineEnd"));
GVAR_DEF(PrimitivePtr, kPrimMirrorMicroStep, std::make_shared<Primitive>("_MirrorMicroStepOperator"));

// Quant ops
GVAR_DEF(PrimitivePtr, kPrimBatchNormFold, std::make_shared<Primitive>("BatchNormFold"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantWithMinMaxVarsPerChannel,
         std::make_shared<Primitive>("FakeQuantWithMinMaxVarsPerChannel"));
GVAR_DEF(PrimitivePtr, kPrimQuant, std::make_shared<Primitive>("Quant"));

// RL Ops
GVAR_DEF(PrimitivePtr, kPrimTensorArrayStack, std::make_shared<Primitive>("TensorArrayStack"));
GVAR_DEF(PrimitivePtr, kPrimTensorArray, std::make_shared<Primitive>("TensorArray"));
GVAR_DEF(PrimitivePtr, kPrimTensorArrayWrite, std::make_shared<Primitive>("TensorArrayWrite"));
GVAR_DEF(PrimitivePtr, kPrimTensorArrayGather, std::make_shared<Primitive>("TensorArrayGather"));
GVAR_DEF(PrimitivePtr, kPrimPartitionedCall, std::make_shared<Primitive>("PartitionedCall"));
GVAR_DEF(PrimitivePtr, kPrimDecodeImage, std::make_shared<Primitive>("DecodeImage"));
GVAR_DEF(PrimitivePtr, kPrimStridedSliceV2, std::make_shared<Primitive>("StridedSliceV2"));
GVAR_DEF(PrimitivePtr, kPrimStridedSliceV2Grad, std::make_shared<Primitive>("StridedSliceV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimKMeansCentroids, std::make_shared<Primitive>("KMeansCentroids"));
GVAR_DEF(PrimitivePtr, kPrimReservoirReplayBufferCreate, std::make_shared<Primitive>("ReservoirReplayBufferCreate"));
GVAR_DEF(PrimitivePtr, kPrimReservoirReplayBufferPush, std::make_shared<Primitive>("ReservoirReplayBufferPush"));
GVAR_DEF(PrimitivePtr, kPrimReservoirReplayBufferSample, std::make_shared<Primitive>("ReservoirReplayBufferSample"));
GVAR_DEF(PrimitivePtr, kPrimReservoirReplayBufferDestroy, std::make_shared<Primitive>("ReservoirReplayBufferDestroy"));
GVAR_DEF(PrimitivePtr, kPrimOCRDetectionPreHandle, std::make_shared<Primitive>("OCRDetectionPreHandle"));
GVAR_DEF(PrimitivePtr, kPrimBufferAppend, std::make_shared<Primitive>("BufferAppend"));

// NPU
GVAR_DEF(PrimitivePtr, kPrimNPUGetFloatStatus, std::make_shared<Primitive>("NPUGetFloatStatus"));
GVAR_DEF(PrimitivePtr, kPrimNPUAllocFloatStatus, std::make_shared<Primitive>("NPUAllocFloatStatus"));
GVAR_DEF(PrimitivePtr, kPrimNPUClearFloatStatus, std::make_shared<Primitive>("NPUClearFloatStatus"));
GVAR_DEF(PrimitivePtr, kPrimAntiQuant, std::make_shared<Primitive>("AntiQuant"));

// Fusion Inference OP
GVAR_DEF(PrimitivePtr, kPrimFFN, std::make_shared<Primitive>("FFN"));

// ToEnum OP
GVAR_DEF(PrimitivePtr, kPrimStringToEnum, std::make_shared<Primitive>("StringToEnum"));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_OTHER_OPS_H_
