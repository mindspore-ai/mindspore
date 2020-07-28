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
#include "src/common/anf_exporter/anf_populater/anf_pool_populater.h"
#include <vector>
#include <string>
#include <memory>
#include "src/common/anf_exporter/anf_populater/anf_node_populater_registry.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"

namespace mindspore::lite {
int mindspore::lite::AnfPoolPopulater::Parse(mindspore::CNodePtr cnodePtr, schema::CNodeT *node,
                                          std::vector<schema::TensorT *> *outputs) {
  auto p = GetCNodePrimitive(cnodePtr);
  auto attr = std::make_unique<schema::PoolingT>();
  if (p->instance_name() == "MaxPool") {
    attr->poolingMode = schema::PoolMode_MAX_POOLING;
  } else if (p->instance_name() == "MeanPool") {
    attr->poolingMode = schema::PoolMode_MEAN_POOLING;
  }

  auto format = GetValue<std::string>(p->GetAttr("data_format"));
  if (format == "NCHW") {
    attr->format = schema::Format_NCHW;
  } else if (format == "NHWC") {
    attr->format = schema::Format_NHWC;
  } else {
    attr->format = schema::Format_NUM_OF_FORMAT;
  }

  auto pad_mode = GetValue<std::string>(p->GetAttr("padding"));
  if (pad_mode == "VALID") {
    attr->padMode = schema::PadMode_VALID;
  } else if (pad_mode == "SAME") {
    attr->padMode = schema::PadMode_SAME;
  } else {
    attr->padMode = schema::PadMode_NOTSET;
  }

  auto kernel_size = GetValue<std::vector<int>>(p->GetAttr("ksize"));
  attr->windowH = kernel_size[2];
  attr->windowW = kernel_size[3];

  auto stride = GetValue<std::vector<int>>(p->GetAttr("strides"));
  attr->strideH = stride[2];
  attr->strideW = stride[3];

  node->nodeType = schema::NodeType_CNode;
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Pooling;
  node->primitive->value.value = attr.release();
  return 0;
}
AnfNodePopulaterRegistrar anfMaxPoolParser("MaxPool", new AnfPoolPopulater());
}  // namespace mindspore::lite
