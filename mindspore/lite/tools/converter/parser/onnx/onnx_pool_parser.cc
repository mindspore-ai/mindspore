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

#include "tools/converter/parser/onnx/onnx_pool_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
lite::PrimitiveC *OnnxPoolParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                     const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx PoolParser";
  auto attr = std::make_unique<schema::PoolingT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  attr->format = schema::Format::Format_NCHW;
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
  } else if (pool_type == "Int8AveragePool") {
    attr->poolingMode = schema::PoolMode_MEAN_POOLING;
    attr->global = false;
  } else {
    MS_LOG(ERROR) << "Pooling param`s PoolingMode is not MAX either AVE. MindSpore support MAX and AVE only.";
    return nullptr;
  }

  attr->roundMode = schema::RoundMode_FLOOR;
  attr->strideW = 1;
  attr->strideH = 1;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "kernel_shape") {
      if (onnx_node_attr.ints_size() == 2) {
        attr->windowH = static_cast<int32_t>(onnx_node_attr.ints(0));
        attr->windowW = static_cast<int32_t>(onnx_node_attr.ints(1));
      }
    }
    if (attribute_name == "strides") {
      if (onnx_node_attr.ints_size() == 2) {
        attr->strideH = static_cast<int32_t>(onnx_node_attr.ints(0));
        attr->strideW = static_cast<int32_t>(onnx_node_attr.ints(1));
      }
    }
    if (attribute_name == "auto_pad") {
      if (onnx_node_attr.s() == "SAME_UPPER") {
        attr->padMode = schema::PadMode_SAME_UPPER;
      } else if (onnx_node_attr.s() == "SAME_LOWER") {
        attr->padMode = schema::PadMode_SAME_LOWER;
      }
    }
    if (attribute_name == "pads") {
      if (onnx_node_attr.ints_size() == 4) {
        attr->padMode = schema::PadMode_CAFFE;
        attr->padUp = static_cast<int32_t>(onnx_node_attr.ints(0));
        attr->padDown = static_cast<int32_t>(onnx_node_attr.ints(2));
        attr->padLeft = static_cast<int32_t>(onnx_node_attr.ints(1));
        attr->padRight = static_cast<int32_t>(onnx_node_attr.ints(3));
      }
    }
    if (attribute_name == "ceil_mode") {
      if (onnx_node_attr.i() == 0) {
        attr->roundMode = schema::RoundMode_FLOOR;
      } else {
        attr->roundMode = schema::RoundMode_CEIL;
      }
    }
    if (attribute_name == "dilations") {
      MS_LOG(ERROR) << "pooling op not support dilations now";
      return nullptr;
    }
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Pooling;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

OnnxNodeRegistrar g_onnxMaxPoolParser("MaxPool", new OnnxPoolParser());
OnnxNodeRegistrar g_onnxAveragePoolParser("AveragePool", new OnnxPoolParser());
OnnxNodeRegistrar g_onnxGlobalAveragePoolParser("GlobalAveragePool", new OnnxPoolParser());
OnnxNodeRegistrar g_onnxGlobalMaxPoolParser("GlobalMaxPool", new OnnxPoolParser());
OnnxNodeRegistrar g_onnxInt8AveragePoolParser("Int8AveragePool", new OnnxPoolParser());
}  // namespace lite
}  // namespace mindspore
