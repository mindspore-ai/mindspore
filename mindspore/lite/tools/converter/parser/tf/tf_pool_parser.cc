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
#include "tools/converter/parser/tf/tf_pool_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "tools/converter/parser/tf/tf_util.h"

namespace mindspore {
namespace lite {
STATUS TFPoolParser::Parse(const tensorflow::NodeDef &tf_op,
                           const std::map<string, const tensorflow::NodeDef *> &tf_node_map, PrimitiveC **primitiveC,
                           std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF PoolParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::PoolingT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  if (tf_op.op() == "MaxPool") {
    attr->poolingMode = schema::PoolMode_MAX_POOLING;
  } else if (tf_op.op() == "AvgPool") {
    attr->poolingMode = schema::PoolMode_MEAN_POOLING;
  }

  tensorflow::AttrValue attr_value;
  if (TensorFlowUtils::FindAttrValue(tf_op, "padding", &attr_value)) {
    if (attr_value.s() == "VALID") {
      attr->padMode = schema::PadMode_VALID;
    } else if (attr_value.s() == "SAME") {
      attr->padMode = schema::PadMode_SAME_UPPER;
    }
  }

  attr->format = TensorFlowUtils::ParseNodeFormat(tf_op);

  if (TensorFlowUtils::FindAttrValue(tf_op, "strides", &attr_value)) {
    const auto &stride_list = attr_value.list();
    if (attr->format == schema::Format_NCHW) {
      attr->strideH = (int32_t)stride_list.i(2);
      attr->strideW = (int32_t)stride_list.i(3);
    } else {
      attr->strideH = (int32_t)stride_list.i(1);
      attr->strideW = (int32_t)stride_list.i(2);
    }
  }

  if (TensorFlowUtils::FindAttrValue(tf_op, "ksize", &attr_value)) {
    const auto &kernel_list = attr_value.list();
    if (attr->format == schema::Format_NCHW) {
      attr->windowH = (int32_t)kernel_list.i(2);
      attr->windowW = (int32_t)kernel_list.i(3);
    } else {
      attr->windowH = (int32_t)kernel_list.i(1);
      attr->windowW = (int32_t)kernel_list.i(2);
    }
  }

  primitive->value.type = schema::PrimitiveType_Pooling;
  primitive->value.value = attr.release();
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
TFNodeRegistrar g_tfMaxPoolParser("MaxPool", new TFPoolParser());
TFNodeRegistrar g_tfAvgPoolParser("AvgPool", new TFPoolParser());
}  // namespace lite
}  // namespace mindspore
