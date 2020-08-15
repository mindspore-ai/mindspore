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

#include <memory>
#include "tools/converter/parser/onnx/onnx_pool_parser.h"

namespace mindspore {
namespace lite {
STATUS OnnxPoolParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx PoolParser";
  std::unique_ptr<schema::PoolingT> attr(new schema::PoolingT());

  const auto &pool_type = onnx_node.op_type();
  if (pool_type == "MaxPool") {
    attr->poolingMode = schema::PoolMode_MAX_POOLING;
    attr->global = false;
  } else if (pool_type == "AveragePool") {
    attr->poolingMode = schema::PoolMode_MEAN_POOLING;
    attr->global = false;
  } else if (pool_type == "GlobalMaxPool") {
    attr->poolingMode = schema::PoolMode_MAX_POOLING;
    attr->global = true;
  } else if (pool_type == "GlobalAveragePool") {
    attr->poolingMode = schema::PoolMode_MEAN_POOLING;
    attr->global = true;
  } else {
    // MS_LOGE("Pooling param`s PoolingMode is not MAX either AVE. MindSpore support MAX and AVE only.");
    return RET_ERROR;
  }

  attr->roundMode = schema::RoundMode_FLOOR;
  attr->strideW = 1;
  attr->strideH = 1;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "kernel_shape") {
      if (onnx_node_attr.ints_size() == 2) {
        attr->windowW = static_cast<int32_t>(onnx_node_attr.ints(0));
        attr->windowH = static_cast<int32_t>(onnx_node_attr.ints(1));
      }
    }
    if (attribute_name == "strides") {
      if (onnx_node_attr.ints_size() == 2) {
        attr->strideW = static_cast<int32_t>(onnx_node_attr.ints(0));
        attr->strideH = static_cast<int32_t>(onnx_node_attr.ints(1));
      }
    }
    if (attribute_name == "auto_pad") {
      MS_ASSERT(false);
    }
    if (attribute_name == "pads") {
      if (onnx_node_attr.ints_size() == 4) {
        attr->padMode = schema::PadMode_CAFFE;
        attr->padUp = static_cast<int32_t>(onnx_node_attr.ints(0));
        attr->padDown = static_cast<int32_t>(onnx_node_attr.ints(1));
        attr->padLeft = static_cast<int32_t>(onnx_node_attr.ints(0));
        attr->padRight = static_cast<int32_t>(onnx_node_attr.ints(1));
      }
    }
    if (attribute_name == "ceil_mode") {
      MS_ASSERT(false);  // todo (h00500767)
      attr->roundMode = schema::RoundMode_CEIL;
    }
    if (attribute_name == "dilations") {
      MS_ASSERT(false);  // todo pooling op not support dilations now
    }
  }
  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Pooling;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxMaxPoolParser("MaxPool", new OnnxPoolParser());
OnnxNodeRegistrar g_onnxAveragePoolParser("AveragePool", new OnnxPoolParser());
OnnxNodeRegistrar g_onnxGlobalAveragePoolParser("GlobalAveragePool", new OnnxPoolParser());
OnnxNodeRegistrar g_onnxGlobalMaxPoolParser("GlobalMaxPool", new OnnxPoolParser());
}  // namespace lite
}  // namespace mindspore

