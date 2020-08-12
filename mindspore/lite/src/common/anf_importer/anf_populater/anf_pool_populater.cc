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
#include "src/common/anf_importer/anf_populater/anf_pool_populater.h"
#include <vector>
#include <string>
#include <memory>
#include "src/common/anf_importer/anf_populater/anf_node_populater_registry.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"

namespace mindspore::lite {
int AnfPoolPopulater::Populate(const PrimitivePtr &prim, PrimitiveTValue *primitiveTValuePtr,
                               const std::vector<AnfNodePtr> &inputs) {
  auto primitive = std::make_unique<schema::PrimitiveT>();
  auto attr = std::make_unique<schema::PoolingT>();
  if (prim->instance_name() == "MaxPool") {
    attr->poolingMode = schema::PoolMode_MAX_POOLING;
  } else if (prim->instance_name() == "MeanPool") {
    attr->poolingMode = schema::PoolMode_MEAN_POOLING;
  }

  auto format = GetValue<std::string>(prim->GetAttr("data_format"));
  if (format == "NCHW") {
    attr->format = schema::Format_NCHW;
  } else if (format == "NHWC") {
    attr->format = schema::Format_NHWC;
  } else {
    attr->format = schema::Format_NUM_OF_FORMAT;
  }

  auto pad_mode = GetValue<std::string>(prim->GetAttr("padding"));
  if (pad_mode == "VALID") {
    attr->padMode = schema::PadMode_VALID;
  } else if (pad_mode == "SAME") {
    attr->padMode = schema::PadMode_SAME;
  } else {
    attr->padMode = schema::PadMode_NOTSET;
  }

  auto kernel_size = GetValue<std::vector<int>>(prim->GetAttr("ksize"));
  attr->windowH = kernel_size[2];
  attr->windowW = kernel_size[3];

  auto stride = GetValue<std::vector<int>>(prim->GetAttr("strides"));
  attr->strideH = stride[2];
  attr->strideW = stride[3];

  primitive->value.type = schema::PrimitiveType_Pooling;
  primitive->value.value = attr.release();
  MS_ASSERT(primitiveTValuePtr != nullptr);
  primitiveTValuePtr->SetPrimitiveT(primitive.release());
  return 0;
}
AnfNodePopulaterRegistrar anfMaxPoolPopulater("MaxPool", new AnfPoolPopulater());
}  // namespace mindspore::lite
