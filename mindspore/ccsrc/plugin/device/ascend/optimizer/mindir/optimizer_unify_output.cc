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

#include "plugin/device/ascend/optimizer/mindir/optimizer_unify_output.h"

#include <vector>
#include <memory>

#include "abstract/abstract_value.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kFtrlOutputNum = 3;
constexpr size_t kMomentumOutputNum = 2;
constexpr size_t kRMSPropOutputNum = 3;
constexpr size_t kCenteredRMSPropOutputNum = 4;
constexpr auto kOptVar = "var";
constexpr auto kOptAccum = "accum";
constexpr auto kOptLinear = "linear";
constexpr auto kOptGrad = "grad";
constexpr auto kOptLr = "lr";
constexpr auto kOptL1 = "l1";
constexpr auto kOptL2 = "l2";
constexpr auto kOptLrPower = "lr_power";
constexpr auto kOptU = "u";
constexpr auto kOptIndex = "index";
constexpr auto kMomentum = "momentum";
constexpr auto kInputs = "inputs";
constexpr auto kMg = "mg";
constexpr auto kMs = "ms";
constexpr auto kMom = "mom";
constexpr auto kRho = "rho";
constexpr auto kEpsilon = "epsilon";
constexpr auto kMOptimizer = "m_optimizer";
constexpr auto kRTupleGet = "r_tuple_get";

bool CheckNode(const AnfNodePtr &node) {
  auto cnode_ptr = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode_ptr);

  auto abstract = cnode_ptr->abstract();
  MS_EXCEPTION_IF_NULL(abstract);

  if (common::AnfAlgo::HasNodeAttr("optim_output_passed", cnode_ptr) && abstract->isa<abstract::AbstractTuple>()) {
    return false;
  }
  return true;
}

AnfNodePtr BuildZero(const PatternMap &) { return NewValueNode(static_cast<int64_t>(0)); }
}  // namespace

AnfNodePtr BuildTupleGetFunc::operator()(const PatternMap &m, const AnfNodePtr &get_item) const {
  auto node = m.Get(kMOptimizer);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode_ptr = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode_ptr);

  auto abstract = cnode_ptr->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  common::AnfAlgo::SetNodeAttr("optim_output_passed", MakeValue(true), cnode_ptr);

  std::vector<AbstractBasePtr> abstract_list;
  for (size_t i = 0; i < output_size_; i++) {
    abstract_list.push_back(abstract->Clone());
  }
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  cnode_ptr->set_abstract(abstract_tuple);

  get_item->set_abstract(abstract->Clone());
  return get_item;
}

bool FtrlUnifyOutput::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &node) const {
  return CheckNode(node);
}

void FtrlUnifyOutput::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern)
    .AddVar(kOptVar)
    .AddVar(kOptAccum)
    .AddVar(kOptLinear)
    .AddVar(kOptGrad)
    .AddVar(kOptLr)
    .AddVar(kOptL1)
    .AddVar(kOptL2)
    .AddVar(kOptLrPower)
    .AddVar(kOptU)
    .AddCNode(kMOptimizer, {prim::kPrimApplyFtrl, kOptVar, kOptAccum, kOptLinear, kOptGrad, kOptLr, kOptL1, kOptL2,
                            kOptLrPower, kOptU});
}

void FtrlUnifyOutput::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern)
    .AddValueNode(kOptIndex, BuildZero)
    .AddCNode(kRTupleGet, {prim::kPrimTupleGetItem, kMOptimizer, kOptIndex}, BuildTupleGetFunc(kFtrlOutputNum));
}

bool MomentumUnifyOutput::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &node) const {
  return CheckNode(node);
}

void MomentumUnifyOutput::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern)
    .AddVar(kOptVar)
    .AddVar(kOptAccum)
    .AddVar(kOptLr)
    .AddVar(kOptGrad)
    .AddVar(kMomentum)
    .AddVar(kOptU)
    .AddCNode(kMOptimizer, {prim::kPrimApplyMomentum, kOptVar, kOptAccum, kOptLr, kOptGrad, kMomentum, kOptU});
}

void MomentumUnifyOutput::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern)
    .AddValueNode(kOptIndex, BuildZero)
    .AddCNode(kRTupleGet, {prim::kPrimTupleGetItem, kMOptimizer, kOptIndex}, BuildTupleGetFunc(kMomentumOutputNum));
}

bool RMSPropUnifyOutput::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &node) const {
  return CheckNode(node);
}

void RMSPropUnifyOutput::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern).AddSeqVar(kInputs).AddCNode(kMOptimizer, {prim::kPrimApplyRMSProp, kInputs});
}

void RMSPropUnifyOutput::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern)
    .AddValueNode(kOptIndex, BuildZero)
    .AddCNode(kRTupleGet, {prim::kPrimTupleGetItem, kMOptimizer, kOptIndex}, BuildTupleGetFunc(kRMSPropOutputNum));
}

bool CenteredRMSPropUnifyOutput::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &,
                                                 const AnfNodePtr &node) const {
  return CheckNode(node);
}

void CenteredRMSPropUnifyOutput::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern)
    .AddVar(kOptVar)
    .AddVar(kMg)
    .AddVar(kMs)
    .AddVar(kMom)
    .AddVar(kOptGrad)
    .AddVar(kOptLr)
    .AddVar(kRho)
    .AddVar(kMomentum)
    .AddVar(kEpsilon)
    .AddVar(kOptU)
    .AddCNode(kMOptimizer, {prim::kPrimApplyCenteredRMSProp, kOptVar, kMg, kMs, kMom, kOptGrad, kOptLr, kRho, kMomentum,
                            kEpsilon, kOptU});
}

void CenteredRMSPropUnifyOutput::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern)
    .AddValueNode(kOptIndex, BuildZero)
    .AddCNode(kRTupleGet, {prim::kPrimTupleGetItem, kMOptimizer, kOptIndex},
              BuildTupleGetFunc(kCenteredRMSPropOutputNum));
}
}  // namespace opt
}  // namespace mindspore
