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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_ARITHMETIC_MAPPER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_ARITHMETIC_MAPPER_H_

#include "tools/converter/adapter/acl/mapper/primitive_mapper.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/pow_fusion.h"
#include "ops/fusion/sub_fusion.h"
#include "ops/fusion/exp_fusion.h"

namespace mindspore {
namespace lite {
using mindspore::ops::kNameAddFusion;
using mindspore::ops::kNameDivFusion;
using mindspore::ops::kNameExpFusion;
using mindspore::ops::kNameMulFusion;
using mindspore::ops::kNamePowFusion;
using mindspore::ops::kNameSubFusion;

class AddFusionMapper : public PrimitiveMapper {
 public:
  AddFusionMapper() : PrimitiveMapper(kNameAddFusion) {}

  ~AddFusionMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};

class DivFusionMapper : public PrimitiveMapper {
 public:
  DivFusionMapper() : PrimitiveMapper(kNameDivFusion) {}

  ~DivFusionMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};

class MulFusionMapper : public PrimitiveMapper {
 public:
  MulFusionMapper() : PrimitiveMapper(kNameMulFusion) {}

  ~MulFusionMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};

class PowFusionMapper : public PrimitiveMapper {
 public:
  PowFusionMapper() : PrimitiveMapper(kNamePowFusion) {}

  ~PowFusionMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};

class SubFusionMapper : public PrimitiveMapper {
 public:
  SubFusionMapper() : PrimitiveMapper(kNameSubFusion) {}

  ~SubFusionMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};

class ExpFusionMapper : public PrimitiveMapper {
 public:
  ExpFusionMapper() : PrimitiveMapper(kNameExpFusion) {}

  ~ExpFusionMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_MAPPER_ARITHMETIC_MAPPER_H_
