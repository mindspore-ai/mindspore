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

#ifndef MINDSPORE_CORE_BASE_LITE_OPS_H_
#define MINDSPORE_CORE_BASE_LITE_OPS_H_

#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/hash_map.h"
#include "ops/lite_op_name.h"

namespace mindspore {
namespace prim {
// Only used in lite
GVAR_DEF(PrimitivePtr, kPrimLeakyRelu, std::make_shared<Primitive>("LeakyRelu"));
GVAR_DEF(PrimitivePtr, kPrimConstant, std::make_shared<Primitive>("Constant"));
GVAR_DEF(PrimitivePtr, kPrimLocalResponseNormalization, std::make_shared<Primitive>("LocalResponseNormalization"));
GVAR_DEF(PrimitivePtr, kPrimFftReal, std::make_shared<Primitive>("FftReal"));
GVAR_DEF(PrimitivePtr, kPrimMfcc, std::make_shared<Primitive>("Mfcc"));
GVAR_DEF(PrimitivePtr, kPrimRfft, std::make_shared<Primitive>("Rfft"));
GVAR_DEF(PrimitivePtr, kPrimFftImag, std::make_shared<Primitive>("FftImag"));
GVAR_DEF(PrimitivePtr, kPrimSkipGram, std::make_shared<Primitive>("SkipGram"));
GVAR_DEF(PrimitivePtr, kPrimConv2DFusion, std::make_shared<Primitive>("Conv2DFusion"));
GVAR_DEF(PrimitivePtr, kPrimConv2dTransposeFusion, std::make_shared<Primitive>("Conv2dTransposeFusion"));
GVAR_DEF(PrimitivePtr, kPrimDepthWiseConv2DFusion, std::make_shared<Primitive>("DepthWiseConv2DFusion"));
GVAR_DEF(PrimitivePtr, kPrimAddFusion, std::make_shared<Primitive>("AddFusion"));
GVAR_DEF(PrimitivePtr, kPrimScaleFusion, std::make_shared<Primitive>("ScaleFusion"));
GVAR_DEF(PrimitivePtr, kPrimSubFusion, std::make_shared<Primitive>("SubFusion"));
GVAR_DEF(PrimitivePtr, kPrimMulFusion, std::make_shared<Primitive>("MulFusion"));
GVAR_DEF(PrimitivePtr, kPrimSigmoid, std::make_shared<Primitive>("Sigmoid"));
GVAR_DEF(PrimitivePtr, kPrimSigmoidGrad, std::make_shared<Primitive>("SigmoidGrad"));
GVAR_DEF(PrimitivePtr, kPrimHSigmoid, std::make_shared<Primitive>("HSigmoid"));
GVAR_DEF(PrimitivePtr, kPrimHSigmoidGrad, std::make_shared<Primitive>("HSigmoidGrad"));
GVAR_DEF(PrimitivePtr, kPrimLogSigmoid, std::make_shared<Primitive>("LogSigmoid"));
GVAR_DEF(PrimitivePtr, kPrimClip, std::make_shared<Primitive>("Clip"));
GVAR_DEF(PrimitivePtr, kPrimHardTanh, std::make_shared<Primitive>("HardTanh"));
GVAR_DEF(PrimitivePtr, kPrimDepthWiseConv2DTransposeFusion,
         std::make_shared<Primitive>("DepthWiseConv2DTransposeFusion"));
GVAR_DEF(PrimitivePtr, kPrimArgMinFusion, std::make_shared<Primitive>("ArgMinFusion"));
GVAR_DEF(PrimitivePtr, kPrimArgMaxFusion, std::make_shared<Primitive>("ArgMaxFusion"));
GVAR_DEF(PrimitivePtr, kPrimSpaceToDepth, std::make_shared<Primitive>("SpaceToDepth"));
GVAR_DEF(PrimitivePtr, kPrimPadFusion, std::make_shared<Primitive>("PadFusion"));
GVAR_DEF(PrimitivePtr, kPrimPowFusion, std::make_shared<Primitive>("PowFusion"));
GVAR_DEF(PrimitivePtr, kPrimResize, std::make_shared<Primitive>("Resize"));
GVAR_DEF(PrimitivePtr, kPrimIf, std::make_shared<Primitive>("If"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolFusion, std::make_shared<Primitive>("AvgPoolFusion"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolFusion, std::make_shared<Primitive>("MaxPoolFusion"));
GVAR_DEF(PrimitivePtr, kPrimActivation, std::make_shared<Primitive>("Activation"));
GVAR_DEF(PrimitivePtr, kPrimActivationGrad, std::make_shared<Primitive>("ActivationGrad"));
GVAR_DEF(PrimitivePtr, kPrimPReLUFusion, std::make_shared<Primitive>("PReLUFusion"));
GVAR_DEF(PrimitivePtr, kPrimTopKFusion, std::make_shared<Primitive>("TopKFusion"));
GVAR_DEF(PrimitivePtr, kPrimTileFusion, std::make_shared<Primitive>("TileFusion"));
GVAR_DEF(PrimitivePtr, kPrimReduceFusion, std::make_shared<Primitive>("ReduceFusion"));
GVAR_DEF(PrimitivePtr, kPrimReduceSumD, std::make_shared<Primitive>("ReduceSumD"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormFusion, std::make_shared<Primitive>("LayerNormFusion"));
GVAR_DEF(PrimitivePtr, kPrimDivFusion, std::make_shared<Primitive>("DivFusion"));
GVAR_DEF(PrimitivePtr, kPrimExpFusion, std::make_shared<Primitive>("ExpFusion"));
GVAR_DEF(PrimitivePtr, kPrimErf, std::make_shared<Primitive>("Erf"));
GVAR_DEF(PrimitivePtr, kPrimErfc, std::make_shared<Primitive>("Erfc"));
GVAR_DEF(PrimitivePtr, kPrimSplice, std::make_shared<Primitive>("Splice"));
GVAR_DEF(PrimitivePtr, kPrimAffine, std::make_shared<Primitive>("Affine"));
GVAR_DEF(PrimitivePtr, kPrimEltwise, std::make_shared<Primitive>("Eltwise"));
GVAR_DEF(PrimitivePtr, kPrimMatMulFusion, std::make_shared<Primitive>("MatMulFusion"));
GVAR_DEF(PrimitivePtr, kPrimDynamicQuant, std::make_shared<Primitive>("DynamicQuant"));
GVAR_DEF(PrimitivePtr, kPrimPartialFusion, std::make_shared<Primitive>("PartialFusion"));
GVAR_DEF(PrimitivePtr, kPrimFSEDecode, std::make_shared<Primitive>("FSEDecode"));
GVAR_DEF(PrimitivePtr, kPrimKVCacheMgr, std::make_shared<Primitive>("KVCacheMgr"));
GVAR_DEF(PrimitivePtr, kPrimMakeTupleV2, std::make_shared<Primitive>("make_tuple"));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_LITE_OPS_H_
