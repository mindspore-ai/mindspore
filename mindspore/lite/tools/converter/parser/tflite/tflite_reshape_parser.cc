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

#include "tools/converter/parser/tflite/tflite_reshape_parser.h"
#include <vector>
#include <memory>
#include "ops/reshape.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteReshapeParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                                            const std::unique_ptr<tflite::ModelT> &tfliteModel) {
  auto prim = std::make_unique<ops::Reshape>();

  MS_ASSERT(tfliteOp != nullptr);
  MS_ASSERT(tfliteModel != nullptr);
  std::vector<int32_t> shape;
  const auto &tflite_subgraph = tfliteModel->subgraphs.front();
  if (tflite_subgraph == nullptr) {
    MS_LOG(ERROR) << "tflite_subgraph is nullptr";
    return nullptr;
  }
  const auto &tflite_attr = tfliteOp->builtin_options.AsReshapeOptions();
  if (tflite_attr != nullptr) {
    shape.resize(tflite_attr->new_shape.size());
    for (size_t i = 0; i < tflite_attr->new_shape.size(); ++i) {
      shape[i] = tflite_attr->new_shape[i];
    }
    prim->AddAttr("shape", MakeValue(shape));
  }

  return prim.release();
}

TfliteNodeRegister g_tfliteReshapeParser(tflite::BuiltinOperator_RESHAPE, new TfliteReshapeParser());
}  // namespace lite
}  // namespace mindspore
