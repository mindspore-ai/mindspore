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
#include "frontend/operator/graph_bprop/bprop_expander_meta_func_graph.h"
#include "frontend/operator/graph_bprop/utils.h"
#include "frontend/operator/graph_bprop/ops_utils.h"
#include "include/common/utils/utils.h"
#include "pipeline/pynative/grad/bprop_expander/bprop.h"

namespace mindspore {
namespace graph_bprop {
FuncGraphPtr BpropExpanderMetaFuncGraph::BpropExpanderFunc(const AbstractBasePtrList &args_spec_list) {
  int64_t list_size = SizeToLong(args_spec_list.size());
  auto fg = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> grads;
  grads.push_back(NewValueNode(primal_));
  for (int64_t i = 0; i < list_size - kTwo; ++i) {
    auto abs_i = args_spec_list[i];
    auto x = fg->add_parameter();
    x->set_abstract(args_spec_list[i]);
    x->abstract()->set_value(args_spec_list[i]->BuildValue());
    (void)grads.emplace_back(x);
  }
  auto out = fg->add_parameter();
  out->set_abstract(args_spec_list[list_size - kTwo]);
  (void)grads.emplace_back(out);
  auto dout = fg->add_parameter();
  dout->set_abstract(args_spec_list[list_size - kOne]);
  (void)grads.emplace_back(dout);
  auto newcnode = fg->NewCNode(grads);
  expander::bprop::BpropExpanderInGraphMode be;
  FuncGraphPtr bprop_fg = nullptr;
  if (be.Run(newcnode)) {
    bprop_fg = be.GetGraph();
    (void)mindspore::opt::ConvertPrimToPrimPy(bprop_fg);
  } else {
    MS_LOG(EXCEPTION) << "Expander failed. Prim is: " << primal_->name();
  }
  return bprop_fg;
}

FuncGraphPtr BpropExpanderMetaFuncGraph::GenerateFuncGraph(const abstract::AbstractBasePtrList &input_abs) {
  return BpropExpanderFunc(input_abs);
}

FuncGraphPtr GetExpandBprop(const PrimitivePtr &primal, const size_t &forward_inputs_size) {
  auto fg = std::make_shared<FuncGraph>();
  auto meta_graph = std::make_shared<BpropExpanderMetaFuncGraph>(primal);
  std::vector<AnfNodePtr> inputs{NewValueNode(meta_graph)};
  for (size_t i = 0; i < forward_inputs_size; ++i) {
    (void)inputs.emplace_back(fg->add_parameter());
  }
  (void)inputs.emplace_back(fg->add_parameter());
  (void)inputs.emplace_back(fg->add_parameter());
  fg->set_output(fg->NewCNode(inputs));
  return fg;
}

void RegMathBpropExpanderOps1() {
  REGISTER_EXPANDER_BPROP_IMPL(Sin);
  REGISTER_EXPANDER_BPROP_IMPL(MatrixInverse);
  REGISTER_EXPANDER_BPROP_IMPL(FloorDiv);
  REGISTER_EXPANDER_BPROP_IMPL(TruncateDiv);
  REGISTER_EXPANDER_BPROP_IMPL(Sqrt);
  REGISTER_EXPANDER_BPROP_IMPL(SqrtGrad);
  REGISTER_EXPANDER_BPROP_IMPL(Rsqrt);
  REGISTER_EXPANDER_BPROP_IMPL(RsqrtGrad);
  REGISTER_EXPANDER_BPROP_IMPL(Reciprocal);
  REGISTER_EXPANDER_BPROP_IMPL(Log);
  REGISTER_EXPANDER_BPROP_IMPL(Log1p);
  REGISTER_EXPANDER_BPROP_IMPL(Erf);
  REGISTER_EXPANDER_BPROP_IMPL(Erfc);
  REGISTER_EXPANDER_BPROP_IMPL(Exp);
  REGISTER_EXPANDER_BPROP_IMPL(Einsum);
  REGISTER_EXPANDER_BPROP_IMPL(Expm1);
  REGISTER_EXPANDER_BPROP_IMPL(Minimum);
  REGISTER_EXPANDER_BPROP_IMPL(Maximum);
  REGISTER_EXPANDER_BPROP_IMPL(CumSum);
  REGISTER_EXPANDER_BPROP_IMPL(CumProd);
  REGISTER_EXPANDER_BPROP_IMPL(ReduceAll);
  REGISTER_EXPANDER_BPROP_IMPL(ReduceAny);
  REGISTER_EXPANDER_BPROP_IMPL(IsFinite);
  REGISTER_EXPANDER_BPROP_IMPL(IsNan);
  REGISTER_EXPANDER_BPROP_IMPL(IsInf);
  REGISTER_EXPANDER_BPROP_IMPL(Equal);
  REGISTER_EXPANDER_BPROP_IMPL(NotEqual);
  REGISTER_EXPANDER_BPROP_IMPL(ApproximateEqual);
  REGISTER_EXPANDER_BPROP_IMPL(Greater);
  REGISTER_EXPANDER_BPROP_IMPL(GreaterEqual);
  REGISTER_EXPANDER_BPROP_IMPL(Less);
  REGISTER_EXPANDER_BPROP_IMPL(LessEqual);
  REGISTER_EXPANDER_BPROP_IMPL(LogicalNot);
  REGISTER_EXPANDER_BPROP_IMPL(LogicalAnd);
  REGISTER_EXPANDER_BPROP_IMPL(Asin);
  REGISTER_EXPANDER_BPROP_IMPL(Asinh);
  REGISTER_EXPANDER_BPROP_IMPL(AsinGrad);
  REGISTER_EXPANDER_BPROP_IMPL(AsinhGrad);
  REGISTER_EXPANDER_BPROP_IMPL(Sinh);
  REGISTER_EXPANDER_BPROP_IMPL(Cos);
  REGISTER_EXPANDER_BPROP_IMPL(ACos);
  REGISTER_EXPANDER_BPROP_IMPL(ACosGrad);
  REGISTER_EXPANDER_BPROP_IMPL(Acosh);
  REGISTER_EXPANDER_BPROP_IMPL(AcoshGrad);
  REGISTER_EXPANDER_BPROP_IMPL(Cosh);
  REGISTER_EXPANDER_BPROP_IMPL(Abs);
  REGISTER_EXPANDER_BPROP_IMPL(Conj);
  REGISTER_EXPANDER_BPROP_IMPL(Sign);
  REGISTER_EXPANDER_BPROP_IMPL(Round);
  REGISTER_EXPANDER_BPROP_IMPL(BesselI0e);
}

void RegMathBpropExpanderOps2() {
  REGISTER_EXPANDER_BPROP_IMPL(Atan);
  REGISTER_EXPANDER_BPROP_IMPL(AtanGrad);
  REGISTER_EXPANDER_BPROP_IMPL(Tan);
  REGISTER_EXPANDER_BPROP_IMPL(BesselI1e);
  REGISTER_EXPANDER_BPROP_IMPL(Atanh);
  REGISTER_EXPANDER_BPROP_IMPL(Inv);
  REGISTER_EXPANDER_BPROP_IMPL(LinSpace);
  REGISTER_EXPANDER_BPROP_IMPL(IndexAdd);
  REGISTER_EXPANDER_BPROP_IMPL(BesselK0);
  REGISTER_EXPANDER_BPROP_IMPL(BesselK1);
  REGISTER_EXPANDER_BPROP_IMPL(BesselK0e);
  REGISTER_EXPANDER_BPROP_IMPL(BesselK1e);
  REGISTER_EXPANDER_BPROP_IMPL(BesselY0);
  REGISTER_EXPANDER_BPROP_IMPL(BesselY1);
  REGISTER_EXPANDER_BPROP_IMPL(NPUGetFloatStatus);
  REGISTER_EXPANDER_BPROP_IMPL(NPUAllocFloatStatus);
  REGISTER_EXPANDER_BPROP_IMPL(NPUClearFloatStatus);
  REGISTER_EXPANDER_BPROP_IMPL(ScalarCast);
  REGISTER_EXPANDER_BPROP_IMPL(Logit);
  REGISTER_EXPANDER_BPROP_IMPL(LuUnpack);
  REGISTER_EXPANDER_BPROP_IMPL(Floor);
  REGISTER_EXPANDER_BPROP_IMPL(Ceil);
  REGISTER_EXPANDER_BPROP_IMPL(Square);
  REGISTER_EXPANDER_BPROP_IMPL(SquareSumAll);
  REGISTER_EXPANDER_BPROP_IMPL(Trunc);
  REGISTER_EXPANDER_BPROP_IMPL(Ger);
  REGISTER_EXPANDER_BPROP_IMPL(Cross);
  REGISTER_EXPANDER_BPROP_IMPL(Median);
  REGISTER_EXPANDER_BPROP_IMPL(Erfinv);
  REGISTER_EXPANDER_BPROP_IMPL(Bernoulli);
  REGISTER_EXPANDER_BPROP_IMPL(ComplexAbs);
  REGISTER_EXPANDER_BPROP_IMPL(Real);
  REGISTER_EXPANDER_BPROP_IMPL(Imag);
  REGISTER_EXPANDER_BPROP_IMPL(Complex);
  REGISTER_EXPANDER_BPROP_IMPL(MinimumGrad);
  REGISTER_EXPANDER_BPROP_IMPL(AddN);
  REGISTER_EXPANDER_BPROP_IMPL(Sinc);
  REGISTER_EXPANDER_BPROP_IMPL(MatrixPower);
  REGISTER_EXPANDER_BPROP_IMPL(TridiagonalMatMul);
  REGISTER_EXPANDER_BPROP_IMPL(LpNorm);
  REGISTER_EXPANDER_BPROP_IMPL(AccumulateNV2);
  REGISTER_EXPANDER_BPROP_IMPL(BesselI1);
  REGISTER_EXPANDER_BPROP_IMPL(BesselJ1);
  REGISTER_EXPANDER_BPROP_IMPL(Atan2);
  REGISTER_EXPANDER_BPROP_IMPL(RealDiv);
  REGISTER_EXPANDER_BPROP_IMPL(DivNoNan);
  REGISTER_EXPANDER_BPROP_IMPL(Xdivy);
  REGISTER_EXPANDER_BPROP_IMPL(FloorMod);
  REGISTER_EXPANDER_BPROP_IMPL(TruncateMod);
  REGISTER_EXPANDER_BPROP_IMPL(Mod);
  REGISTER_EXPANDER_BPROP_IMPL(Xlogy);
}

void RegMathBpropExpanderOps3() {
  REGISTER_EXPANDER_BPROP_IMPL(SquaredDifference);
  REGISTER_EXPANDER_BPROP_IMPL(Hypot);
  REGISTER_EXPANDER_BPROP_IMPL(Lerp);
  REGISTER_EXPANDER_BPROP_IMPL(AddV2);
  REGISTER_EXPANDER_BPROP_IMPL(Addcdiv);
  REGISTER_EXPANDER_BPROP_IMPL(Addcmul);
}

void RegNNBpropExpanderOps1() {
  REGISTER_EXPANDER_BPROP_IMPL(AdaptiveAvgPool2D);
  REGISTER_EXPANDER_BPROP_IMPL(AdaptiveMaxPool2D);
  REGISTER_EXPANDER_BPROP_IMPL(BatchNormGrad);
  REGISTER_EXPANDER_BPROP_IMPL(BinaryCrossEntropy);
  REGISTER_EXPANDER_BPROP_IMPL(CeLU);
  REGISTER_EXPANDER_BPROP_IMPL(CTCLoss);
  REGISTER_EXPANDER_BPROP_IMPL(CTCLossV2);
  REGISTER_EXPANDER_BPROP_IMPL(DeformableOffsets);
  REGISTER_EXPANDER_BPROP_IMPL(Dilation2D);
  REGISTER_EXPANDER_BPROP_IMPL(Dropout);
  REGISTER_EXPANDER_BPROP_IMPL(DropoutDoMask);
  REGISTER_EXPANDER_BPROP_IMPL(DropoutGenMask);
  REGISTER_EXPANDER_BPROP_IMPL(DropoutGrad);
  REGISTER_EXPANDER_BPROP_IMPL(Elu);
  REGISTER_EXPANDER_BPROP_IMPL(FastGeLU);
  REGISTER_EXPANDER_BPROP_IMPL(FastGelu);
  REGISTER_EXPANDER_BPROP_IMPL(FractionMaxPool);
  REGISTER_EXPANDER_BPROP_IMPL(FractionMaxPoolWithFixedKsize);
  REGISTER_EXPANDER_BPROP_IMPL(FractionMaxPool3DWithFixedKsize);
  REGISTER_EXPANDER_BPROP_IMPL(GridSampler2D);
  REGISTER_EXPANDER_BPROP_IMPL(GridSampler3D);
  REGISTER_EXPANDER_BPROP_IMPL(HShrink);
  REGISTER_EXPANDER_BPROP_IMPL(HSigmoid);
  REGISTER_EXPANDER_BPROP_IMPL(HSwish);
  REGISTER_EXPANDER_BPROP_IMPL(InstanceNorm);
  REGISTER_EXPANDER_BPROP_IMPL(InstanceNormV2);
  REGISTER_EXPANDER_BPROP_IMPL(LayerNormGrad);
  REGISTER_EXPANDER_BPROP_IMPL(LogSoftmax);
  REGISTER_EXPANDER_BPROP_IMPL(LRN);
  REGISTER_EXPANDER_BPROP_IMPL(L2Loss);
  REGISTER_EXPANDER_BPROP_IMPL(L2Normalize);
  REGISTER_EXPANDER_BPROP_IMPL(MaxPoolGradGrad);
  REGISTER_EXPANDER_BPROP_IMPL(MaxPoolV1);
  REGISTER_EXPANDER_BPROP_IMPL(MaxPoolWithArgmax);
  REGISTER_EXPANDER_BPROP_IMPL(MaxPool3D);
  REGISTER_EXPANDER_BPROP_IMPL(MaxPool3DGrad);
  REGISTER_EXPANDER_BPROP_IMPL(MaxPool3DGradGrad);
  REGISTER_EXPANDER_BPROP_IMPL(MaxPool3DWithArgmax);
  REGISTER_EXPANDER_BPROP_IMPL(MirrorPad);
  REGISTER_EXPANDER_BPROP_IMPL(Mish);
  REGISTER_EXPANDER_BPROP_IMPL(MultilabelMarginLoss);
  REGISTER_EXPANDER_BPROP_IMPL(MultiMarginLoss);
  REGISTER_EXPANDER_BPROP_IMPL(NLLLoss);
  REGISTER_EXPANDER_BPROP_IMPL(NthElement);
  REGISTER_EXPANDER_BPROP_IMPL(OneHot);
  REGISTER_EXPANDER_BPROP_IMPL(Pdist);
  REGISTER_EXPANDER_BPROP_IMPL(PReLU);
  REGISTER_EXPANDER_BPROP_IMPL(ReluGrad);
  REGISTER_EXPANDER_BPROP_IMPL(ReLUV2);
  REGISTER_EXPANDER_BPROP_IMPL(ReLUV3);
}

void RegNNBpropExpanderOps2() {
  REGISTER_EXPANDER_BPROP_IMPL(ReLU6);
  REGISTER_EXPANDER_BPROP_IMPL(ResizeBilinear);
  REGISTER_EXPANDER_BPROP_IMPL(ResizeLinear1D);
  REGISTER_EXPANDER_BPROP_IMPL(RNNTLoss);
  REGISTER_EXPANDER_BPROP_IMPL(SeLU);
  REGISTER_EXPANDER_BPROP_IMPL(Sigmoid);
  REGISTER_EXPANDER_BPROP_IMPL(SigmoidCrossEntropyWithLogits);
  REGISTER_EXPANDER_BPROP_IMPL(SigmoidGrad);
  REGISTER_EXPANDER_BPROP_IMPL(SmoothL1Loss);
  REGISTER_EXPANDER_BPROP_IMPL(SoftMarginLoss);
  REGISTER_EXPANDER_BPROP_IMPL(Softplus);
  REGISTER_EXPANDER_BPROP_IMPL(SoftShrink);
  REGISTER_EXPANDER_BPROP_IMPL(Softsign);
  REGISTER_EXPANDER_BPROP_IMPL(TanhGrad);
  REGISTER_EXPANDER_BPROP_IMPL(UpsampleTrilinear3D);
  REGISTER_EXPANDER_BPROP_IMPL(AvgPool);
  REGISTER_EXPANDER_BPROP_IMPL(MulNoNan);
  REGISTER_EXPANDER_BPROP_IMPL(Tanh);
  REGISTER_EXPANDER_BPROP_IMPL(LSTM);
  REGISTER_EXPANDER_BPROP_IMPL(Dropout2D);
  REGISTER_EXPANDER_BPROP_IMPL(Dropout3D);
  REGISTER_EXPANDER_BPROP_IMPL(UpsampleNearest3D);
  REGISTER_EXPANDER_BPROP_IMPL(PadV3);
  REGISTER_EXPANDER_BPROP_IMPL(MaxUnpool2D);
  REGISTER_EXPANDER_BPROP_IMPL(MaxUnpool3D);
  REGISTER_EXPANDER_BPROP_IMPL(AdaptiveAvgPool2DV1);
  REGISTER_EXPANDER_BPROP_IMPL(SparseSoftmaxCrossEntropyWithLogitsV2);
}

void RegArrayBpropExpanderOps1() {
  REGISTER_EXPANDER_BPROP_IMPL(Argmax);
  REGISTER_EXPANDER_BPROP_IMPL(Argmin);
  REGISTER_EXPANDER_BPROP_IMPL(BatchToSpace);
  REGISTER_EXPANDER_BPROP_IMPL(CheckNumerics);
  REGISTER_EXPANDER_BPROP_IMPL(Col2Im);
  REGISTER_EXPANDER_BPROP_IMPL(ConjugateTranspose);
  REGISTER_EXPANDER_BPROP_IMPL(DepthToSpace);
  REGISTER_EXPANDER_BPROP_IMPL(Diag);
  REGISTER_EXPANDER_BPROP_IMPL(DiagPart);
  REGISTER_EXPANDER_BPROP_IMPL(DType);
  REGISTER_EXPANDER_BPROP_IMPL(Fill);
  REGISTER_EXPANDER_BPROP_IMPL(Fills);
  REGISTER_EXPANDER_BPROP_IMPL(GatherD);
  REGISTER_EXPANDER_BPROP_IMPL(Identity);
  REGISTER_EXPANDER_BPROP_IMPL(IdentityN);
  REGISTER_EXPANDER_BPROP_IMPL(MaskedSelect);
  REGISTER_EXPANDER_BPROP_IMPL(MatrixDiagV3);
  REGISTER_EXPANDER_BPROP_IMPL(NonZero);
  REGISTER_EXPANDER_BPROP_IMPL(OnesLike);
  REGISTER_EXPANDER_BPROP_IMPL(Range);
  REGISTER_EXPANDER_BPROP_IMPL(Rank);
  REGISTER_EXPANDER_BPROP_IMPL(ReverseSequence);
  REGISTER_EXPANDER_BPROP_IMPL(ReverseV2);
  REGISTER_EXPANDER_BPROP_IMPL(ScatterMax);
  REGISTER_EXPANDER_BPROP_IMPL(ScatterMin);
  REGISTER_EXPANDER_BPROP_IMPL(SegmentMax);
  REGISTER_EXPANDER_BPROP_IMPL(SegmentMin);
  REGISTER_EXPANDER_BPROP_IMPL(Select);
  REGISTER_EXPANDER_BPROP_IMPL(Slice);
  REGISTER_EXPANDER_BPROP_IMPL(SpaceToBatch);
  REGISTER_EXPANDER_BPROP_IMPL(SpaceToDepth);
  REGISTER_EXPANDER_BPROP_IMPL(Split);
  REGISTER_EXPANDER_BPROP_IMPL(SplitV);
  REGISTER_EXPANDER_BPROP_IMPL(TensorScatterAdd);
  REGISTER_EXPANDER_BPROP_IMPL(TensorScatterDiv);
  REGISTER_EXPANDER_BPROP_IMPL(TensorScatterElement);
  REGISTER_EXPANDER_BPROP_IMPL(TensorScatterMul);
  REGISTER_EXPANDER_BPROP_IMPL(TensorScatterSub);
  REGISTER_EXPANDER_BPROP_IMPL(TensorScatterUpdate);
  REGISTER_EXPANDER_BPROP_IMPL(Tril);
  REGISTER_EXPANDER_BPROP_IMPL(Triu);
  REGISTER_EXPANDER_BPROP_IMPL(ZerosLike);
  REGISTER_EXPANDER_BPROP_IMPL(Concat);
}

void RegArrayBpropExpanderOps2() {}
void RegClipBpropExpanderOps() {}
void RegCommBpropExpanderOps() {}
void RegInnerBpropExpanderOps() {}
void RegOtherBpropExpanderOps() {}
void RegQuantBpropExpanderOps() {}
void RegSparseBpropExpanderOps() {}

void RegBpropExpanderOps() {
  RegMathBpropExpanderOps1();
  RegMathBpropExpanderOps2();
  RegMathBpropExpanderOps3();
  RegNNBpropExpanderOps1();
  RegNNBpropExpanderOps2();
  RegArrayBpropExpanderOps1();
  RegArrayBpropExpanderOps2();
  RegClipBpropExpanderOps();
  RegCommBpropExpanderOps();
  RegInnerBpropExpanderOps();
  RegOtherBpropExpanderOps();
  RegQuantBpropExpanderOps();
  RegSparseBpropExpanderOps();
}
}  // namespace graph_bprop
}  // namespace mindspore
