/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/common/anf_exporter/anf_populater/anf_transpose_populater.h"
#include <vector>
#include <string>
#include <memory>
#include "src/common/anf_exporter/anf_populater/anf_node_populater_registry.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"

namespace mindspore::lite {
int mindspore::lite::AnfTransposePopulater::Parse(mindspore::CNodePtr cnodePtr, schema::CNodeT *node,
                                          std::vector<schema::TensorT *> *outputs) {
  auto attr = std::make_unique<schema::TransposeT>();

  MS_ASSERT(cnodePtr->size() == kAnfPopulaterThree);
  auto inputNode = cnodePtr->input(kAnfPopulaterTwo);
  if (inputNode->isa<ValueNode>()) {
    auto valNode = inputNode->cast<ValueNodePtr>();
    MS_ASSERT(valNode != nullptr);
    auto val = valNode->value();
    MS_ASSERT(val != nullptr);
    if (val->isa<ValueTuple>()) {
      auto tuple = val->cast<ValueTuplePtr>();
      MS_ASSERT(tuple != nullptr);
      for (size_t i = 0; i < tuple->size(); i++) {
        auto elem = tuple->value()[i]->cast<Int32ImmPtr>();
        MS_ASSERT(elem != nullptr);
        attr->perm.emplace_back(static_cast<int>(elem->value()));
      }
    }
  }

  node->nodeType = schema::NodeType_CNode;
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Transpose;
  node->primitive->value.value = attr.release();
  return 0;
}
AnfNodePopulaterRegistrar anfTransposeParser("Transpose", new AnfTransposePopulater());
}  // namespace mindspore::lite
