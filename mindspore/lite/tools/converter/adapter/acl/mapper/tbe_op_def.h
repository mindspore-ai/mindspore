/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_TBE_OP_DEF_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_TBE_OP_DEF_H_

#include "ops/primitive_c.h"

namespace mindspore {
namespace lite {
namespace acl {
#define ADD_CONVERTER_TBE_OP(name)       \
  constexpr auto kName##name = #name;    \
  class name : public ops::PrimitiveC {  \
   public:                               \
    name() : PrimitiveC(kName##name) {}  \
    ~name() = default;                   \
    MS_DECLARE_PARENT(name, PrimitiveC); \
  };

ADD_CONVERTER_TBE_OP(Pooling)
ADD_CONVERTER_TBE_OP(AvgPoolV2)
ADD_CONVERTER_TBE_OP(MaxPoolV3)
ADD_CONVERTER_TBE_OP(PadV3)
ADD_CONVERTER_TBE_OP(PadV2)
ADD_CONVERTER_TBE_OP(Pad)
ADD_CONVERTER_TBE_OP(MirrorPad)
ADD_CONVERTER_TBE_OP(StridedSliceV2)
ADD_CONVERTER_TBE_OP(GlobalAveragePool)
ADD_CONVERTER_TBE_OP(BNInference)
ADD_CONVERTER_TBE_OP(Deconvolution)
ADD_CONVERTER_TBE_OP(Upsample)
ADD_CONVERTER_TBE_OP(Conv2DTransposeD)
ADD_CONVERTER_TBE_OP(DepthwiseConv2dNative)
ADD_CONVERTER_TBE_OP(ArgMaxV2)
ADD_CONVERTER_TBE_OP(ResizeNearestNeighborV2)
ADD_CONVERTER_TBE_OP(ResizeBilinearV2)
ADD_CONVERTER_TBE_OP(Conv2DBackpropInputV2)
ADD_CONVERTER_TBE_OP(ConcatV2)
ADD_CONVERTER_TBE_OP(FillV1)
ADD_CONVERTER_TBE_OP(Quant)
ADD_CONVERTER_TBE_OP(Dequant)
ADD_CONVERTER_TBE_OP(SpaceToBatchTF)
ADD_CONVERTER_TBE_OP(BatchToSpaceTF)
ADD_CONVERTER_TBE_OP(ClipByValue)
ADD_CONVERTER_TBE_OP(SqueezeV3)
ADD_CONVERTER_TBE_OP(DynamicReduceProd)
ADD_CONVERTER_TBE_OP(TopKV2)
ADD_CONVERTER_TBE_OP(CommonLSTM)
ADD_CONVERTER_TBE_OP(Swish)
}  // namespace acl
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_TBE_OP_DEF_H_
