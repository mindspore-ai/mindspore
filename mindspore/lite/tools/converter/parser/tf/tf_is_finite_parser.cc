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
#include "tools/converter/parser/tf/tf_is_finite_parser.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "tools/common/node_util.h"

namespace mindspore {
namespace lite {
STATUS TFIsFiniteParser::Parse(const tensorflow::NodeDef &tf_op,
                               const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                               PrimitiveC **primitiveC, std::vector<std::string> *inputs, int *output_size) {
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return RET_NULL_PTR;
  }

  int status = CreateOperator<schema::IsFiniteT>(primitive, schema::PrimitiveType_IsFinite);
  if (status != RET_OK) {
    return status;
  }
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }

  return RET_OK;
}
TFNodeRegistrar g_tf_is_finite_parser("IsFinite", new TFIsFiniteParser());
}  // namespace lite
}  // namespace mindspore
