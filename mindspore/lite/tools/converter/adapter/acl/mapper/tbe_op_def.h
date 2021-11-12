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

#ifndef ACL_MAPPER_TBE_OP_DEF_H
#define ACL_MAPPER_TBE_OP_DEF_H

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
ADD_CONVERTER_TBE_OP(PadV1)
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
ADD_CONVERTER_TBE_OP(Conv2DBackpropInputV2)
ADD_CONVERTER_TBE_OP(ConcatV2D)
ADD_CONVERTER_TBE_OP(FillV1)
}  // namespace acl
}  // namespace lite
}  // namespace mindspore
#endif  // ACL_MAPPER_TBE_OP_DEF_H
