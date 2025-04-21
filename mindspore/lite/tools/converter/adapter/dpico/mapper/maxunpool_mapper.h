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

#ifndef DPICO_MAPPER_MAXUNPOOL_MAPPER_H_
#define DPICO_MAPPER_MAXUNPOOL_MAPPER_H_

#include <memory>
#include <vector>
#include "mapper/op_mapper.h"
#include "mapper/op_mapper_registry.h"

namespace mindspore {
namespace dpico {
class MaxUnpoolMapper : public OpMapper {
 public:
  MaxUnpoolMapper() : OpMapper("MaxUnpool") {}
  ~MaxUnpoolMapper() override = default;
  STATUS Map(const api::CNodePtr &node, std::vector<BaseOperatorPtr> *base_operators, const api::PrimitivePtr &prim,
             const api::CNodePtrList &output_cnodes) override;
};
}  // namespace dpico
}  // namespace mindspore

#endif  // DPICO_MAPPER_MAXUNPOOL_MAPPER_H_
