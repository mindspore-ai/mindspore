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

namespace mindspore {
namespace lite {
STATUS TFReduceParser::Parse(const tensorflow::NodeDef &tf_op,
                             const std::map<string, const tensorflow::NodeDef *> &tf_node_map, PrimitiveC **primitiveC,
                             std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF ReduceParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "New PrimitiveT failed";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::ReduceT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return RET_NULL_PTR;
  }

  if (tf_op.op() == "Sum") {
    attr->mode = schema::ReduceMode_ReduceSum;
  } else if (tf_op.op() == "Max") {
    attr->mode = schema::ReduceMode_ReduceMax;
  } else if (tf_op.op() == "Min") {
    attr->mode = schema::ReduceMode_ReduceMin;
  } else if (tf_op.op() == "Mean") {
    attr->mode = schema::ReduceMode_ReduceMean;
  } else if (tf_op.op() == "Prod") {
    attr->mode = schema::ReduceMode_ReduceProd;
  } else if (tf_op.op() == "All") {
    attr->mode = schema::ReduceMode_ReduceAll;
  } else {
    MS_LOG(ERROR) << "unsupported reduce mode: " << tf_op.op();
    return RET_ERROR;
  }
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "keep_dims", &attr_value)) {
    MS_LOG(ERROR) << "The keep_dims attr should be specified";
    return RET_ERROR;
  }
  if (attr_value.value_case() != tensorflow::AttrValue::kB) {
    MS_LOG(ERROR) << "the keep_dims attr of reduce should be bool type";
    return RET_ERROR;
  }
  attr->keepDims = attr_value.b();

  auto axis_node = GetConstInputNode(tf_node_map, tf_op.input(1));
  if (axis_node == nullptr) {
    MS_LOG(ERROR) << "Find Reduce input axis failed";
    return RET_ERROR;
  }
  if (!TensorFlowUtils::FindAttrValue(*axis_node, "value", &attr_value)) {
    MS_LOG(ERROR) << "The value attr should be specified";
    return RET_ERROR;
  }
  auto tensor_proto = attr_value.tensor();
  if (tensor_proto.int_val_size() > 0) {
    for (int i = 0; i < tensor_proto.int_val_size(); ++i) {
      attr->axes.push_back(tensor_proto.int_val(i));
    }
  } else {
    auto data_num = tensor_proto.tensor_content().size() / sizeof(int32_t);
    auto data = reinterpret_cast<const int32_t *>(tensor_proto.tensor_content().data());
    for (size_t i = 0; i < data_num; ++i) {
      attr->axes.push_back(data[i]);
    }
  }

  primitive->value.type = schema::PrimitiveType_Reduce;
  primitive->value.value = attr.release();
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  *output_size = 1;
  auto status = AddOpInput(tf_op, 0, inputs);
  return status;
}
TFNodeRegistrar g_tfSumParser("Sum", new TFReduceParser());
TFNodeRegistrar g_tfMaxParser("Max", new TFReduceParser());
TFNodeRegistrar g_tfMinParser("Min", new TFReduceParser());
TFNodeRegistrar g_tfMeanParser("Mean", new TFReduceParser());
TFNodeRegistrar g_tfProdParser("Prod", new TFReduceParser());
TFNodeRegistrar g_tfAllParser("All", new TFReduceParser());
}  // namespace lite
}  // namespace mindspore
