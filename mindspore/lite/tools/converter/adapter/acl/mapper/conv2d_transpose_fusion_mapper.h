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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_CONV2D_TRANSPOSE_FUSION_MAPPER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_CONV2D_TRANSPOSE_FUSION_MAPPER_H_

#include "tools/converter/adapter/acl/mapper/primitive_mapper.h"
#include "ops/fusion/conv2d_transpose_fusion.h"

using mindspore::ops::kNameConv2dTransposeFusion;

namespace mindspore {
namespace lite {
class Conv2dTransposeMapper : public PrimitiveMapper {
 public:
  Conv2dTransposeMapper() : PrimitiveMapper(kNameConv2dTransposeFusion) {}
  ~Conv2dTransposeMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;

 private:
  STATUS AdjustGeAttr(const CNodePtr &cnode, const PrimitivePtr &dst_prim);
  STATUS AdjustOutputPadding(const PrimitivePtr &dst_prim);
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_CONV2D_TRANSPOSE_FUSION_MAPPER_H_
