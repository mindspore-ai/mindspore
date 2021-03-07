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
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fusion/max_pool_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFMaxPoolParser::Parse(const tensorflow::NodeDef &tf_op,
                                        const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                        std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::MaxPoolFusion>();

  tensorflow::AttrValue attr_value;
  if (TensorFlowUtils::FindAttrValue(tf_op, "padding", &attr_value)) {
    if (attr_value.s() == "VALID") {
      prim->set_pad_mode(mindspore::PadMode::VALID);
    } else if (attr_value.s() == "SAME") {
      prim->set_pad_mode(mindspore::PadMode::SAME);
    }
  }

  auto format = TensorFlowUtils::ParseNodeFormat(tf_op);
  prim->set_format(format);

  if (TensorFlowUtils::FindAttrValue(tf_op, "strides", &attr_value)) {
    const auto &stride_list = attr_value.list();
    if (format == mindspore::Format::NCHW) {
      prim->set_strides({stride_list.i(2), stride_list.i(3)});
    } else {
      prim->set_strides({stride_list.i(1), stride_list.i(2)});
    }
  }

  if (TensorFlowUtils::FindAttrValue(tf_op, "ksize", &attr_value)) {
    const auto &kernel_list = attr_value.list();
    if (format == mindspore::Format::NCHW) {
      prim->set_kernel_size({kernel_list.i(2), kernel_list.i(3)});
    } else {
      prim->set_kernel_size({kernel_list.i(1), kernel_list.i(2)});
    }
  }

  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }

  return prim.release();
}

ops::PrimitiveC *TFAvgPoolParser::Parse(const tensorflow::NodeDef &tf_op,
                                        const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                        std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::AvgPoolFusion>();

  tensorflow::AttrValue attr_value;
  if (TensorFlowUtils::FindAttrValue(tf_op, "padding", &attr_value)) {
    if (attr_value.s() == "VALID") {
      prim->set_pad_mode(mindspore::PadMode::VALID);
    } else if (attr_value.s() == "SAME") {
      prim->set_pad_mode(mindspore::PadMode::SAME);
    }
  }

  auto format = TensorFlowUtils::ParseNodeFormat(tf_op);
  prim->set_format(format);

  if (TensorFlowUtils::FindAttrValue(tf_op, "strides", &attr_value)) {
    const auto &stride_list = attr_value.list();
    if (format == mindspore::Format::NCHW) {
      prim->set_strides({stride_list.i(2), stride_list.i(3)});
    } else {
      prim->set_strides({stride_list.i(1), stride_list.i(2)});
    }
  }

  if (TensorFlowUtils::FindAttrValue(tf_op, "ksize", &attr_value)) {
    const auto &kernel_list = attr_value.list();
    if (format == mindspore::Format::NCHW) {
      prim->set_kernel_size({kernel_list.i(2), kernel_list.i(3)});
    } else {
      prim->set_kernel_size({kernel_list.i(1), kernel_list.i(2)});
    }
  }

  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }

  return prim.release();
}

TFNodeRegistrar g_tfMaxPoolParser("MaxPool", new TFMaxPoolParser());
TFNodeRegistrar g_tfAvgPoolParser("AvgPool", new TFAvgPoolParser());
}  // namespace lite
}  // namespace mindspore
