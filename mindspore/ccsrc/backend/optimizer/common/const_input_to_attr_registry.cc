/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/common/const_input_to_attr_registry.h"

#include <utility>

#include "utils/utils.h"
#include "utils/log_adapter.h"
#include "base/core_ops.h"

namespace mindspore {
namespace opt {
ConstInputToAttrInfoRegistry::ConstInputToAttrInfoRegistry() {
  Register(prim::kPrimCast->name(), {1});
  Register(prim::kPrimAvgPoolGradVm->name(), {0});
  Register(prim::kPrimConv2DBackpropInput->name(), {2});
  Register(prim::kPrimConv2DBackpropFilter->name(), {2});
  Register(prim::kPrimConv3DBackpropInput->name(), {2});
  Register(prim::kPrimConv3DBackpropFilter->name(), {2});
  Register(prim::kPrimDepthwiseConv2dNativeBackpropFilter->name(), {1});
  Register(prim::kPrimDepthwiseConv2dNativeBackpropInput->name(), {0});
  Register(prim::kPrimReshape->name(), {1});
  Register(prim::kPrimReduceMax->name(), {1});
  Register(prim::kPrimReduceMin->name(), {1});
  Register(prim::kPrimReduceSum->name(), {1});
  Register(prim::kPrimReduceMean->name(), {1});
  Register(prim::kPrimCentralization->name(), {1});
  Register(prim::kPrimGather->name(), {2});
  Register(prim::kPrimGatherD->name(), {1});
  Register(prim::kPrimEmbeddingLookup->name(), {2, 3, 4, 5});
  Register(prim::kPrimEmbeddingLookupCommGrad->name(), {1});
  Register(prim::kPrimSubscalar->name(), {1});
  Register(prim::kPrimTranspose->name(), {1});
  Register(prim::kPrimUnsortedSegmentSum->name(), {2});
  Register(prim::kPrimOneHot->name(), {1});
  Register(prim::kPrimConcat->name(), {0});
  Register(prim::kPrimCumSum->name(), {1});
  Register(prim::kPrimCumProd->name(), {1});
  Register(prim::kPrimReduceAll->name(), {1});
  Register(prim::kPrimReduceAny->name(), {1});
  Register(prim::kPrimUnsortedSegmentMin->name(), {2});
  Register(prim::kPrimUnsortedSegmentMax->name(), {2});
  Register(kSparseGatherV2OpName, {2});
  Register(kUnsortedSegmentProdOpName, {2});
  Register(kSimpleMeanGradOpName, {1});
  Register(kMeanGradOpName, {1});
  Register(kSliceOpName, {1, 2});
  Register(kSliceGradOpName, {2, 3});
  Register(kTileOpName, {1});
  Register(kScatterNdOpName, {2});
  Register(kStridedSliceAssignOpName, {1, 2, 3});
  Register(kStridedSliceOpName, {1, 2, 3});
  Register(kStridedSliceGradOpName, {1, 2, 3, 4});
  Register(kFlattenGradOpName, {1});
  Register(kExpandDimsOpName, {1});
  Register(kSplitOpName, {0});
  Register(kErfOpName, {1});
  Register(kSparseApplyAdagradOpName, {2});
  Register(kResizeNearestNeighborGradOpName, {1});
  Register(kResizeNearestNeighborV2OpName, {1});
  Register(kResizeNearestNeighborV2GradOpName, {1});
  Register(kApplyRMSPropOpname, {5, 6, 7});
  Register(kResizeBilinearV2OpName, {1});
  Register(kReduceProdOpName, {1});
  Register(kCumprodOpName, {1});
  Register(kSpaceToBatchOpName, {1});
  Register(kBatchToSpaceOpName, {1});
  Register(kPadOpName, {1});
  Register(kPushOpName, {1});
}

ConstInputToAttrInfoRegistry &ConstInputToAttrInfoRegistry::Instance() {
  static ConstInputToAttrInfoRegistry instance;
  return instance;
}

void ConstInputToAttrInfoRegistry::Register(const ConstInputToAttrInfoRegister &reg) {
  auto op_name = reg.GetOpName();
  if (op_input_to_attr_map_.find(op_name) == op_input_to_attr_map_.end()) {
    (void)op_input_to_attr_map_.insert(make_pair(op_name, reg));
    MS_LOG(DEBUG) << op_name << " const2attr register successfully!";
  }
}

void ConstInputToAttrInfoRegistry::Register(const std::string &op_name,
                                            const std::unordered_set<size_t> &input_attr_set) {
  if (op_input_to_attr_map_.find(op_name) == op_input_to_attr_map_.end()) {
    ConstInputToAttrInfoRegister reg(op_name);
    (void)reg.SetConstInputToAttr(input_attr_set);
    (void)op_input_to_attr_map_.insert(make_pair(op_name, reg));
    MS_LOG(DEBUG) << op_name << " const2attr register successfully!";
  }
}

bool ConstInputToAttrInfoRegistry::GetRegisterByOpName(const std::string &op_name,
                                                       ConstInputToAttrInfoRegister *reg) const {
  if (op_input_to_attr_map_.find(op_name) != op_input_to_attr_map_.end()) {
    *reg = op_input_to_attr_map_.at(op_name);
    MS_LOG(DEBUG) << op_name << " const2attr find in registry.";
    return true;
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
