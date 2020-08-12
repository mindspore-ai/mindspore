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
#include "src/common/anf_importer/anf_populater/anf_activation_populater.h"
#include <vector>
#include <memory>
#include "src/common/anf_importer/anf_populater/anf_node_populater_registry.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"

namespace mindspore::lite {
int AnfActivationPopulater::Populate(const PrimitivePtr &prim, PrimitiveTValue *primitiveTValuePtr,
                                     const std::vector<AnfNodePtr> &inputs) {
  auto primitive = std::make_unique<schema::PrimitiveT>();
  auto attr = std::make_unique<schema::ActivationT>();
  if (prim->name() == "ReLU") {
    attr->type = schema::ActivationType_RELU;
  } else if (prim->name() == "Sigmoid") {
    attr->type = schema::ActivationType_SIGMOID;
  } else if (prim->name() == "ReLU6") {
    attr->type = schema::ActivationType_RELU6;
  }

  primitive->value.type = schema::PrimitiveType_Activation;
  primitive->value.value = attr.release();
  MS_ASSERT(primitiveTValuePtr != nullptr);
  primitiveTValuePtr->SetPrimitiveT(primitive.release());
  return 0;
}
AnfNodePopulaterRegistrar anfReLUPopulater("ReLU", new AnfActivationPopulater());
AnfNodePopulaterRegistrar anfReLU6Populater("ReLU6", new AnfActivationPopulater());
AnfNodePopulaterRegistrar anfSigmoidPopulater("Sigmoid", new AnfActivationPopulater());
}  // namespace mindspore::lite
