/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_DYNAMIC_CREATOR_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_DYNAMIC_CREATOR_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "frontend/parallel/ops_info/ops_info_head_files.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {
#define REGISTER(className)                                                                                  \
  OperatorInfoPtr objectCreator##className(std::string name, Shapes in, Shapes out, PrimitiveAttrs &attrs) { \
    return std::make_shared<className>(name, in, out, attrs);                                                \
  }                                                                                                          \
  RegisterAction className##Register(#className, (CreatFn)objectCreator##className);

typedef OperatorInfoPtr (*CreatFn)(const std::string &name, const Shapes &shape_in, const Shapes shape_out,
                                   const PrimitiveAttrs &attrs);

class DynCreator {
 public:
  ~DynCreator() = default;

  // create static singleton dyn_creator instance
  static DynCreator &Instance() {
    static DynCreator fac = DynCreator();
    return fac;
  }
  // register
  void Register(std::string name, CreatFn func) { (void)Function_map_.insert(std::make_pair(name, func)); }
  // creator
  OperatorInfoPtr Create(const std::string &name, const Shapes &shape_in, const Shapes &shape_out,
                         const PrimitiveAttrs &attrs, size_t count) {
    std::string op_name = name + std::to_string(count);
    auto iter = Function_map_.find(name);
    if (iter == Function_map_.end()) {
      MS_LOG(INFO) << name << " is not register yet";
      return nullptr;
    }
    return iter->second(op_name, shape_in, shape_out, attrs);
  }

 private:
  DynCreator() = default;
  std::map<std::string, CreatFn> Function_map_;
};

class RegisterAction {
 public:
  RegisterAction(const std::string &name, CreatFn creatfn) : name_(name) {
    DynCreator::Instance().Register(name, creatfn);
  }
  ~RegisterAction() = default;

 private:
  std::string name_;
};

// operator register
REGISTER(MatMulInfo);
REGISTER(GeLUInfo);
REGISTER(VirtualDatasetInfo);
REGISTER(BatchParallelInfo);
REGISTER(TanhInfo);
REGISTER(SoftmaxInfo);
REGISTER(LogSoftmaxInfo);
REGISTER(ActivationInfo);
REGISTER(SoftmaxCrossEntropyWithLogitsInfo);
REGISTER(SubInfo);
REGISTER(AddInfo);
REGISTER(BiasAddInfo);
REGISTER(MulInfo);
REGISTER(DivInfo);
REGISTER(ModInfo);
REGISTER(RealDivInfo);
REGISTER(PowInfo);
REGISTER(ExpInfo);
REGISTER(OneHotInfo);
REGISTER(EqualInfo);
REGISTER(NotEqualInfo);
REGISTER(LogInfo);
REGISTER(CosInfo);
REGISTER(ACosInfo);
REGISTER(LogicalNotInfo);
REGISTER(L2NormalizeInfo);
REGISTER(LayerNormInfo);
REGISTER(ReduceMaxInfo);
REGISTER(ArgMaxWithValueInfo);
REGISTER(ArgMinWithValueInfo);
REGISTER(ReduceMeanInfo);
REGISTER(ReduceSumInfo);
REGISTER(ReduceMinInfo);
REGISTER(TransposeInfo);
REGISTER(PReLUInfo);
REGISTER(DropoutDoMaskInfo);
REGISTER(ReshapeInfo);
REGISTER(FloorDivInfo);
REGISTER(MaximumInfo);
REGISTER(MinimumInfo);
REGISTER(CastInfo);
REGISTER(GreaterInfo);
REGISTER(GreaterEqualInfo);
REGISTER(LessEqualInfo);
REGISTER(LessInfo);
REGISTER(ApproximateEqualInfo);
REGISTER(SparseSoftmaxCrossEntropyWithLogitsInfo);
REGISTER(AssignSubInfo);
REGISTER(FloorModInfo);
REGISTER(AssignInfo);
REGISTER(AssignAddInfo);
REGISTER(Atan2Info);
REGISTER(DivNoNanInfo);
REGISTER(LogicalAndInfo);
REGISTER(LogicalOrInfo);
REGISTER(EluInfo);
REGISTER(ReLUInfo);
REGISTER(RepeatElementsInfo);
REGISTER(TensorDotInfo);
REGISTER(RangeInfo);
REGISTER(ReLU6Info);
REGISTER(ReLUV2Info);
REGISTER(SoftplusInfo);
REGISTER(SoftsignInfo);
REGISTER(GatherInfo);
REGISTER(SparseGatherV2Info);
REGISTER(SqrtInfo);
REGISTER(SigmoidInfo);
REGISTER(GetNextInfo);
REGISTER(NegInfo);
REGISTER(AbsInfo);
REGISTER(AcoshInfo);
REGISTER(AsinInfo);
REGISTER(AsinhInfo);
REGISTER(AtanInfo);
REGISTER(AtanhInfo);
REGISTER(CeilInfo);
REGISTER(CoshInfo);
REGISTER(Expm1Info);
REGISTER(Log1pInfo);
REGISTER(SinInfo);
REGISTER(SinhInfo);
REGISTER(TanInfo);
REGISTER(RsqrtInfo);
REGISTER(InvInfo);
REGISTER(ReciprocalInfo);
REGISTER(RoundInfo);
REGISTER(FloorInfo);
REGISTER(SignInfo);
REGISTER(ErfInfo);
REGISTER(ErfcInfo);
REGISTER(ZerosLikeInfo);
REGISTER(OnesLikeInfo);
REGISTER(BesselI0eInfo);
REGISTER(BesselI1eInfo);
REGISTER(BatchMatMulInfo);
REGISTER(ExpandDimsInfo);
REGISTER(SqueezeInfo);
REGISTER(SigmoidCrossEntropyWithLogitsInfo);
REGISTER(SquareInfo);
REGISTER(UniformCandidateSamplerInfo);
REGISTER(UnsortedSegmentSumInfo);
REGISTER(UnsortedSegmentMinInfo);
REGISTER(UnsortedSegmentMaxInfo);
REGISTER(GatherPInfo);
REGISTER(EmbeddingLookupInfo);
REGISTER(TileInfo);
REGISTER(BroadcastToInfo);
REGISTER(StridedSliceInfo);
REGISTER(SliceInfo);
REGISTER(DropoutInfo);
REGISTER(StackInfo);
REGISTER(ConcatInfo);
REGISTER(SplitInfo);
REGISTER(UniqueInfo);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_DYNAMIC_CREATOR_H_
