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

#ifndef LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_OPS_DEF_H_
#define LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_OPS_DEF_H_
#include "schema/inner/model_generated.h"
#include "ops/primitive_c.h"
using mindspore::ops::PrimitiveC;

namespace mindspore {
namespace lite {
#define ADD_CONVERTER_ONLY_OP(name)      \
  constexpr auto kName##name = #name;    \
  class name : public PrimitiveC {       \
   public:                               \
    name() : PrimitiveC(kName##name) {}  \
    ~name() = default;                   \
    MS_DECLARE_PARENT(name, PrimitiveC); \
  };

ADD_CONVERTER_ONLY_OP(Enter);
ADD_CONVERTER_ONLY_OP(Exit);
ADD_CONVERTER_ONLY_OP(If);
ADD_CONVERTER_ONLY_OP(LoopCond);
ADD_CONVERTER_ONLY_OP(NextIteration);
ADD_CONVERTER_ONLY_OP(TensorArrayGatherV3);
ADD_CONVERTER_ONLY_OP(TensorArrayReadV3);
ADD_CONVERTER_ONLY_OP(TensorArrayScatterV3);
ADD_CONVERTER_ONLY_OP(TensorArraySizeV3);
ADD_CONVERTER_ONLY_OP(TensorArrayV3);
ADD_CONVERTER_ONLY_OP(TensorArrayWriteV3);
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_NEXTITERATION_H_
