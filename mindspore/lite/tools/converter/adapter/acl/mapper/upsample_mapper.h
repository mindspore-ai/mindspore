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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_UPSAMPLE_MAPPER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_UPSAMPLE_MAPPER_H_

#include "tools/converter/adapter/acl/mapper/primitive_mapper.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "ops/resize.h"

namespace mindspore {
namespace lite {
using mindspore::ops::kNameResize;

class UpsampleMapper : public PrimitiveMapper {
 public:
  UpsampleMapper() : PrimitiveMapper(acl::kNameUpsample) {}

  ~UpsampleMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;

 private:
  STATUS AttrAdjust(const PrimitivePtr &src_prim, const ValueNodePtr &val_node);
  STATUS RemoveConstInput(const CNodePtr &cnode);
};
}  // namespace lite
}  // namespace mindspore
#endif  // ACL_MAPPER_PRIMITIVE_TRANSPOSE_MAPPER_H
