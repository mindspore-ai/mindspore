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
#include "tools/converter/parser/tf/tf_reduce_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/fusion/reduce_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFReduceParser::Parse(const tensorflow::NodeDef &tf_op,
                                       const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                       std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::ReduceFusion>();

  if (tf_op.op() == "Sum") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Sum);
  } else if (tf_op.op() == "Max") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Max);
  } else if (tf_op.op() == "Min") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Min);
  } else if (tf_op.op() == "Mean") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Mean);
  } else if (tf_op.op() == "Prod") {
    prim->set_mode(mindspore::ReduceMode::Reduce_Prod);
  } else if (tf_op.op() == "All") {
    prim->set_mode(mindspore::ReduceMode::Reduce_All);
  } else {
    MS_LOG(ERROR) << "unsupported reduce mode: " << tf_op.op();
    return nullptr;
  }

  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "keep_dims", &attr_value)) {
    MS_LOG(ERROR) << "The keep_dims attr should be specified";
    return nullptr;
  }

  if (attr_value.value_case() != tensorflow::AttrValue::kB) {
    MS_LOG(ERROR) << "the keep_dims attr of reduce should be bool type";
    return nullptr;
  }
  prim->set_keep_dims(attr_value.b());

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}

TFNodeRegistrar g_tfSumParser("Sum", new TFReduceParser());
TFNodeRegistrar g_tfMaxParser("Max", new TFReduceParser());
TFNodeRegistrar g_tfMinParser("Min", new TFReduceParser());
TFNodeRegistrar g_tfMeanParser("Mean", new TFReduceParser());
TFNodeRegistrar g_tfProdParser("Prod", new TFReduceParser());
TFNodeRegistrar g_tfAllParser("All", new TFReduceParser());
}  // namespace lite
}  // namespace mindspore
