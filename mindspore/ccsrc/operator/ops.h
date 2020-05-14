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

#ifndef MINDSPORE_CCSRC_OPERATOR_OPS_H_
#define MINDSPORE_CCSRC_OPERATOR_OPS_H_

#include <iostream>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"

namespace mindspore {
// namespace to support primitive operators
namespace prim {
ValuePtr GetPythonOps(const std::string &op_name,
                      const std::string &module_name = "mindspore._extends.parse.standard_method");

// Arithmetic
extern const PrimitivePtr kPrimScalarAdd;
extern const PrimitivePtr kPrimScalarSub;
extern const PrimitivePtr kPrimScalarMul;
extern const PrimitivePtr kPrimScalarDiv;
extern const PrimitivePtr kPrimScalarFloordiv;
extern const PrimitivePtr kPrimScalarMod;
extern const PrimitivePtr kPrimScalarPow;
extern const PrimitivePtr kPrimScalarTrunc;
extern const PrimitivePtr kPrimScalarFloor;
extern const PrimitivePtr kPrimScalarUadd;
extern const PrimitivePtr kPrimScalarUsub;
extern const PrimitivePtr kPrimScalarExp;
extern const PrimitivePtr kPrimScalarLog;
extern const PrimitivePtr kPrimScalarSin;
extern const PrimitivePtr kPrimScalarCos;
extern const PrimitivePtr kPrimScalarTan;

// Comparisons
extern const PrimitivePtr kPrimScalarEq;
extern const PrimitivePtr kPrimScalarLt;
extern const PrimitivePtr kPrimScalarGt;
extern const PrimitivePtr kPrimScalarNe;
extern const PrimitivePtr kPrimScalarLe;
extern const PrimitivePtr kPrimScalarGe;
extern const PrimitivePtr kPrimBoolNot;
extern const PrimitivePtr kPrimBoolAnd;
extern const PrimitivePtr kPrimBoolOr;
extern const PrimitivePtr kPrimBoolEq;

// Type introspection
extern const PrimitivePtr kPrimTypeOf;
extern const PrimitivePtr kPrimHasType;

// Statements
extern const PrimitivePtr kPrimSwitch;
extern const PrimitivePtr kPrimSwitchLayer;
extern const PrimitivePtr kPrimReturn;
extern const PrimitivePtr kPrimAssign;
extern const PrimitivePtr kPrimAssignAdd;
extern const PrimitivePtr kPrimAssignSub;
extern const PrimitivePtr kPrimSelect;
extern const PrimitivePtr kPrimCall;

extern const PrimitivePtr kPrimDistribute;
extern const PrimitivePtr kPrimDot;
extern const PrimitivePtr kPrimIm2Col;
extern const PrimitivePtr kPrimCol2Im;
extern const PrimitivePtr kPrimIm2ColV1;
extern const PrimitivePtr kPrimCol2ImV1;

extern const PrimitivePtr kPrimResolve;
extern const PrimitivePtr kPrimEmbed;
extern const PrimitivePtr kPrimRefToEmbed;
extern const PrimitivePtr kPrimCreateInstance;

extern const PrimitivePtr kPrimLabelGoto;
extern const PrimitivePtr kPrimLabelSwitch;
extern const PrimitivePtr kPrimLabelSet;

// Structure
extern const PrimitivePtr kPrimStringEqual;
extern const PrimitivePtr kPrimStringConcat;
extern const PrimitivePtr kPrimMakeTuple;
extern const PrimitivePtr kPrimMakeList;
extern const PrimitivePtr kPrimMakeDict;
extern const PrimitivePtr kPrimMakeKeywordArg;
extern const PrimitivePtr kPrimExtractKeywordArg;
extern const PrimitivePtr kPrimMakeSlice;
extern const PrimitivePtr kPrimMakeRecord;
extern const PrimitivePtr kPrimTupleGetItem;
extern const PrimitivePtr kPrimListGetItem;
extern const PrimitivePtr kPrimArrayGetItem;
extern const PrimitivePtr kPrimTupleSetItem;
extern const PrimitivePtr kPrimListSetItem;
extern const PrimitivePtr kPrimArraySetItem;
extern const PrimitivePtr kPrimDictGetItem;
extern const PrimitivePtr kPrimDictSetItem;
extern const PrimitivePtr kPrimListAppend;
extern const PrimitivePtr kPrimGetAttr;
extern const PrimitivePtr kPrimTupleLen;
extern const PrimitivePtr kPrimDictLen;
extern const PrimitivePtr kPrimListLen;
extern const PrimitivePtr kPrimArrayLen;
extern const PrimitivePtr kPrimListMap;
extern const PrimitivePtr kPrimListReduce;
extern const PrimitivePtr kPrimTupleReversed;
extern const PrimitivePtr kPrimTileShape;
extern const PrimitivePtr kPrimReducedShape;
extern const PrimitivePtr kPrimTupleDiv;
extern const PrimitivePtr kPrimTupleToArray;
extern const PrimitivePtr kPrimShapeMul;
extern const PrimitivePtr kPrimGenerateShapeIndex;
extern const PrimitivePtr kPrimGenerateInverseIndex;
extern const PrimitivePtr kPrimTupleEqual;
extern const PrimitivePtr kPrimListEqual;
extern const PrimitivePtr kPrimMakeRange;
extern const PrimitivePtr kPrimStopGradient;

// Arrays
extern const PrimitivePtr kPrimScalarToArray;
extern const PrimitivePtr kPrimArrayToScalar;
extern const PrimitivePtr kPrimBroadcastShape;
extern const PrimitivePtr kPrimArrayMap;
extern const PrimitivePtr kPrimArrayReduce;
extern const PrimitivePtr kPrimShape;
extern const PrimitivePtr kPrimCast;
extern const PrimitivePtr kPrimConcat;
extern const PrimitivePtr kPrimSqueeze;
extern const PrimitivePtr kPrimTranspose;
extern const PrimitivePtr kPrimGatherV2;
extern const PrimitivePtr kPrimSize;
extern const PrimitivePtr kPrimArgMax;
extern const PrimitivePtr kPrimPack;
extern const PrimitivePtr kPrimUnpack;
extern const PrimitivePtr kPrimUnsortedSegmentMin;
extern const PrimitivePtr kPrimUnsortedSegmentSum;
extern const PrimitivePtr kPrimConcatOffset;
extern const PrimitivePtr kPrimReshape;
extern const PrimitivePtr kPrimTile;
extern const PrimitivePtr kPrimAddN;
extern const PrimitivePtr KPrimTransData;

// Maths
extern const PrimitivePtr kPrimTensorAdd;
extern const PrimitivePtr kPrimMatMul;
extern const PrimitivePtr kPrimBatchMatMul;
extern const PrimitivePtr kPrimMaximumGrad;
extern const PrimitivePtr kPrimMinimumGrad;
extern const PrimitivePtr kPrimReduceMean;
extern const PrimitivePtr kPrimReduceSum;
extern const PrimitivePtr kPrimReduceAll;
extern const PrimitivePtr kPrimReduceMax;
extern const PrimitivePtr kPrimReduceMin;
extern const PrimitivePtr kPrimNeg;
extern const PrimitivePtr kPrimSub;
extern const PrimitivePtr kPrimMul;
extern const PrimitivePtr kPrimMinimum;
extern const PrimitivePtr kPrimMaximum;
extern const PrimitivePtr kPrimSquare;
extern const PrimitivePtr kPrimEqual;
extern const PrimitivePtr kPrimLess;
extern const PrimitivePtr kPrimLessEqual;
extern const PrimitivePtr kPrimCumSum;
extern const PrimitivePtr kPrimCumProd;

// NN
extern const PrimitivePtr kPrimFlatten;
extern const PrimitivePtr kPrimLogSoftmax;
extern const PrimitivePtr kPrimLogSoftmaxGrad;
extern const PrimitivePtr kPrimTanh;
extern const PrimitivePtr kPrimTanhGrad;
extern const PrimitivePtr kPrimPooling;
extern const PrimitivePtr kPrimPoolingGrad;
extern const PrimitivePtr kPrimFusedBatchNorm;
extern const PrimitivePtr kPrimBatchNorm;
extern const PrimitivePtr kPrimBatchNormGrad;
extern const PrimitivePtr kPrimConv2D;
extern const PrimitivePtr kPrimMaxPool;
extern const PrimitivePtr kPrimMaxPoolGrad;
extern const PrimitivePtr kPrimAvgPoolGrad;
extern const PrimitivePtr kPrimFusedBatchNormGrad;
extern const PrimitivePtr kPrimReluGrad;
extern const PrimitivePtr kPrimConv2DBackpropInput;
extern const PrimitivePtr kPrimConv2DBackpropFilter;
extern const PrimitivePtr kPrimDepthwiseConv2dNative;
extern const PrimitivePtr kPrimDepthwiseConv2dNativeBackpropFilter;
extern const PrimitivePtr kPrimDepthwiseConv2dNativeBackpropInput;

extern const PrimitivePtr kPrimBiasAddGrad;
extern const PrimitivePtr kPrimSoftmaxCrossEntropyWithLogits;
extern const PrimitivePtr kPrimSparseSoftmaxCrossEntropyWithLogits;
extern const PrimitivePtr kPrimMomentum;
extern const PrimitivePtr kPrimApplyMomentum;
extern const PrimitivePtr kPrimLayerNorm;
extern const PrimitivePtr kPrimLayerNormGrad;
extern const PrimitivePtr kPrimLayerNormXBackprop;
extern const PrimitivePtr kPrimLayerNormBetaGammaBackprop;
extern const PrimitivePtr kPrimDropoutGenMask;
extern const PrimitivePtr kPrimOneHot;
extern const PrimitivePtr kPrimGelu;
extern const PrimitivePtr kPrimGeluGrad;
extern const PrimitivePtr kPrimRelu;
extern const PrimitivePtr kPrimReluV2;
extern const PrimitivePtr kPrimActivation;
extern const PrimitivePtr kPrimZerosLikeTensor;
extern const PrimitivePtr kPrimFakeBprop;
extern const PrimitivePtr kPrimBpropCut;

// Other Miscellaneous
extern const PrimitivePtr kPrimIdentity;
extern const PrimitivePtr kPrimPartial;
extern const PrimitivePtr kPrimJ;
extern const PrimitivePtr kPrimEnvSetItem;
extern const PrimitivePtr kPrimEnvGetItem;
extern const PrimitivePtr kPrimEnvAdd;
extern const PrimitivePtr kPrimMakeRefKey;
extern const PrimitivePtr kPrimMakeRef;
extern const PrimitivePtr kPrimGetRefKey;
extern const PrimitivePtr kPrimGetRefValue;
extern const PrimitivePtr kPrimGetRefOrigin;
extern const PrimitivePtr kPrimInsertGradientOf;
extern const PrimitivePtr kPrimHookBackward;
extern const PrimitivePtr kPrimPrintShapeType;
extern const PrimitivePtr kPrimPrint;
extern const PrimitivePtr kPrimSameTypeShape;
extern const PrimitivePtr kPrimCheckBprop;
extern const PrimitivePtr kPrimDepend;
extern const PrimitivePtr kPrimStateSetItem;
extern const PrimitivePtr kPrimScalarSummary;
extern const PrimitivePtr kPrimImageSummary;
extern const PrimitivePtr kPrimTensorSummary;
extern const PrimitivePtr kPrimHistogramSummary;
extern const PrimitivePtr kPrimBroadcastGradientArgs;
extern const PrimitivePtr kPrimControlDepend;
extern const PrimitivePtr kPrimIs_;
extern const PrimitivePtr kPrimIsNot;
extern const PrimitivePtr kPrimInDict;
extern const PrimitivePtr kPrimNotInDict;

// Comm ops
extern const PrimitivePtr kPrimAllReduce;
extern const PrimitivePtr kPrimMirror;
extern const PrimitivePtr kPrimVirtualDiv;
extern const PrimitivePtr kPrimVirtualDataset;

class DoSignaturePrimitive : public Primitive {
 public:
  explicit DoSignaturePrimitive(const std::string &name, const ValuePtr &function)
      : Primitive("S-Prim-" + name), function_(function) {}

  ~DoSignaturePrimitive() override = default;

  MS_DECLARE_PARENT(DoSignaturePrimitive, Primitive)

  const ValuePtr function() const { return function_; }

 private:
  ValuePtr function_;
};
using DoSignaturePrimitivePtr = std::shared_ptr<DoSignaturePrimitive>;

class UnpackGraphPrimitive : public Primitive {
 public:
  explicit UnpackGraphPrimitive(const std::string &name, const bool &with_sens, const bool &need_unpack_args)
      : Primitive("UnpackGraph"), with_sens_in_args_(with_sens), need_unpack_args_(need_unpack_args) {}
  ~UnpackGraphPrimitive() override = default;
  MS_DECLARE_PARENT(UnpackGraphPrimitive, Primitive)
  bool with_sens_in_args() const { return with_sens_in_args_; }
  bool need_unpack_args() const { return need_unpack_args_; }

 private:
  bool with_sens_in_args_;
  bool need_unpack_args_;
};
using UnpackGraphPrimitivePtr = std::shared_ptr<UnpackGraphPrimitive>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_OPERATOR_OPS_H_
