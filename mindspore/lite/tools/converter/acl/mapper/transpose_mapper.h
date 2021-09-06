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

#ifndef ACL_MAPPER_PRIMITIVE_TRANSPOSE_MAPPER_H
#define ACL_MAPPER_PRIMITIVE_TRANSPOSE_MAPPER_H

#include "tools/converter/acl/mapper/primitive_mapper.h"
#include "ops/transpose.h"

using mindspore::ops::kNameTranspose;

namespace mindspore {
namespace lite {
class TransposeMapper : public PrimitiveMapper {
 public:
  TransposeMapper() : PrimitiveMapper(kNameTranspose) {}

  ~TransposeMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // ACL_MAPPER_PRIMITIVE_TRANSPOSE_MAPPER_H
