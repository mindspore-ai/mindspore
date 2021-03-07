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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {
namespace {
typedef struct WhileParemeter {
  OpParameter op_parameter_;
  int body_subgraph_index;
  int cond_subgraph_index;
} WhileParemeter;

OpParameter *PopulateWhileParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto while_prim = primitive->value_as_While();
  WhileParemeter *while_paremeter = reinterpret_cast<WhileParemeter *>(malloc(sizeof(WhileParemeter)));
  if (while_paremeter == nullptr) {
    MS_LOG(ERROR) << "malloc WhileParemeter failed.";
    return nullptr;
  }
  memset(while_paremeter, 0, sizeof(WhileParemeter));

  while_paremeter->op_parameter_.type_ = schema::PrimitiveType_While;
  while_paremeter->body_subgraph_index = while_prim->bodySubgraphIndex();
  while_paremeter->cond_subgraph_index = while_prim->condSubgraphIndex();
  return reinterpret_cast<OpParameter *>(while_paremeter);
}
}  // namespace

Registry g_whileV0ParemeterRegistry(schema::v0::PrimitiveType_While, PopulateWhileParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
