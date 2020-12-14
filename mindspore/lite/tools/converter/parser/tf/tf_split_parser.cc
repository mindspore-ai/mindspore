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
#include "tools/converter/parser/tf/tf_split_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
STATUS TFSplitParser::Parse(const tensorflow::NodeDef &tf_op,
                            const std::map<string, const tensorflow::NodeDef *> &tf_node_map, PrimitiveC **primitiveC,
                            std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF SplitParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "New PrimitiveT failed";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::SplitT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return RET_NULL_PTR;
  }

  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "num_split", &attr_value)) {
    MS_LOG(ERROR) << "The attribute num_split should be specified";
    return RET_PARAM_INVALID;
  }
  attr->numberSplit = (int32_t)(attr_value.i());

  int split_dim_index;
  int input_index;
  if (tf_op.op() == "Split") {
    split_dim_index = 0;
    input_index = 1;
  } else {
    split_dim_index = 2;
    input_index = 0;
  }

  auto split_dim_node = GetConstInputNode(tf_node_map, tf_op.input(split_dim_index));
  if (split_dim_node == nullptr) {
    MS_LOG(ERROR) << "Find Split input split_dim node failed";
    return RET_ERROR;
  }
  if (!TensorFlowUtils::FindAttrValue(*split_dim_node, "value", &attr_value)) {
    MS_LOG(ERROR) << "The attribute splitDim should be specified";
    return RET_PARAM_INVALID;
  }
  auto split_dim_tensor = attr_value.tensor();
  attr->splitDim = split_dim_tensor.int_val(0);
  *output_size = attr->numberSplit;

  if (tf_op.op() == "SplitV") {
    auto size_splits_node = GetConstInputNode(tf_node_map, tf_op.input(1));
    if (size_splits_node == nullptr) {
      MS_LOG(ERROR) << "Find Split input size_splits failed";
      return RET_ERROR;
    }
    if (!TensorFlowUtils::FindAttrValue(*size_splits_node, "value", &attr_value)) {
      MS_LOG(ERROR) << "The attribute size splits should be specified";
      return RET_PARAM_INVALID;
    }
    auto size_splits_tensor = attr_value.tensor();
    auto size = size_splits_tensor.tensor_content().size() / sizeof(int32_t);
    attr->sizeSplits.resize(size);
    auto ret = memcpy_s(attr->sizeSplits.data(), size * sizeof(int32_t), size_splits_tensor.tensor_content().data(),
                        size * sizeof(int32_t));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed";
      return RET_ERROR;
    }
  }

  primitive->value.type = schema::PrimitiveType_Split;
  primitive->value.value = attr.release();
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  auto status = AddOpInput(tf_op, input_index, inputs);
  return status;
}
TFNodeRegistrar g_tfSplitParser("Split", new TFSplitParser());
TFNodeRegistrar g_tfSplitVParser("SplitV", new TFSplitParser());
}  // namespace lite
}  // namespace mindspore
