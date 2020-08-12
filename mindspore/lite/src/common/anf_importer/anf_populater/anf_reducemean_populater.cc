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
#include "src/common/anf_importer/anf_populater/anf_reducemean_populater.h"
#include <vector>
#include <memory>
#include "src/common/anf_importer/anf_populater/anf_node_populater_registry.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"

namespace mindspore::lite {
namespace {
  constexpr int kReduceInputNum = 3;
  constexpr int kReduceInputIndex = 2;
}
int AnfReduceMeanPopulater::Populate(const PrimitivePtr &prim, PrimitiveTValue *primitiveTValuePtr,
                                     const std::vector<AnfNodePtr> &inputs) {
  auto primitive = std::make_unique<schema::PrimitiveT>();
  auto attr = std::make_unique<schema::ReduceT>();
  attr->mode = schema::ReduceMode_ReduceMean;

  attr->keepDims = GetValue<bool>(prim->GetAttr("keep_dims"));
  if (inputs.size() == kReduceInputNum) {
    auto inputNode = inputs[kReduceInputIndex];
    MS_ASSERT(inputNode != nullptr);
    if (inputNode->isa<ValueNode>()) {
      auto valueNode = inputNode->cast<ValueNodePtr>();
      MS_ASSERT(valueNode != nullptr);
      auto value = valueNode->value();
      MS_ASSERT(value != nullptr);
      if (value->isa<ValueTuple>()) {
        auto valTuplPtr = dyn_cast<ValueTuple>(value);
        MS_ASSERT(valTuplPtr != nullptr);
        for (size_t i = 0; i < valTuplPtr->size(); i++) {
          auto elem = dyn_cast<Int32Imm>((*valTuplPtr)[i]);
          MS_ASSERT(elem != nullptr);
          attr->axes.emplace_back(elem->value());
        }
      }
    }
  }

  primitive->value.type = schema::PrimitiveType_Reduce;
  primitive->value.value = attr.release();
  MS_ASSERT(primitiveTValuePtr != nullptr);
  primitiveTValuePtr->SetPrimitiveT(primitive.release());
  return 0;
}
AnfNodePopulaterRegistrar anfReduceMeanPopulater("ReduceMean", new AnfReduceMeanPopulater());
}  // namespace mindspore::lite
