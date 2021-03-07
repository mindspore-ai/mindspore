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
#include <vector>
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fusion/max_pool_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxAvgPoolParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::AvgPoolFusion>();

  prim->set_format(mindspore::Format::NCHW);
  prim->set_pad_mode(mindspore::PadMode::PAD);
  mindspore::RoundMode roundMode = mindspore::RoundMode::FLOOR;
  std::vector<int64_t> kernels;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "kernel_shape") {
      if (onnx_node_attr.ints_size() == 2) {
        kernels.push_back(onnx_node_attr.ints(0));
        kernels.push_back(onnx_node_attr.ints(1));
        prim->set_kernel_size(kernels);
      }
    }
    if (attribute_name == "strides") {
      if (onnx_node_attr.ints_size() == 2) {
        strides.push_back(onnx_node_attr.ints(0));
        strides.push_back(onnx_node_attr.ints(1));
      }
    }
    if (attribute_name == "auto_pad") {
      if (onnx_node_attr.s() == "SAME_UPPER") {
        prim->set_pad_mode(mindspore::PadMode::SAME);
      } else if (onnx_node_attr.s() == "SAME_LOWER") {
        MS_LOG(ERROR) << "PadMode_SAME_LOWER is not supported now";
        return nullptr;
      }
    }
    if (attribute_name == "pads") {
      if (onnx_node_attr.ints_size() == 4) {
        pads.push_back(onnx_node_attr.ints(0));
        pads.push_back(onnx_node_attr.ints(2));
        pads.push_back(onnx_node_attr.ints(1));
        pads.push_back(onnx_node_attr.ints(3));
      }
    }
    if (attribute_name == "ceil_mode") {
      if (onnx_node_attr.i() == 0) {
        roundMode = mindspore::RoundMode::FLOOR;
      } else {
        roundMode = mindspore::RoundMode::CEIL;
      }
    }
    if (attribute_name == "dilations") {
      MS_LOG(ERROR) << "pooling op not support dilations now";
      return nullptr;
    }
  }
  prim->set_round_mode(roundMode);

  if (strides.empty()) {
    strides.push_back(1);
    strides.push_back(1);
  }
  prim->set_strides(strides);
  if (pads.empty()) {
    pads = {0, 0, 0, 0};
  }
  prim->set_pad(pads);
  if (onnx_node.op_type() == "GlobalAveragePool") {
    prim->set_global(true);
  } else {
    prim->set_global(false);
  }

  return prim.release();
}

ops::PrimitiveC *OnnxMaxPoolParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::MaxPoolFusion>();

  prim->set_format(mindspore::Format::NCHW);
  mindspore::RoundMode roundMode = mindspore::RoundMode::FLOOR;
  std::vector<int64_t> kernels;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "kernel_shape") {
      if (onnx_node_attr.ints_size() == 2) {
        kernels.push_back(onnx_node_attr.ints(0));
        kernels.push_back(onnx_node_attr.ints(1));
        prim->set_kernel_size(kernels);
      }
    }
    if (attribute_name == "strides") {
      if (onnx_node_attr.ints_size() == 2) {
        strides.push_back(onnx_node_attr.ints(0));
        strides.push_back(onnx_node_attr.ints(1));
      }
    }
    if (attribute_name == "auto_pad") {
      if (onnx_node_attr.s() == "SAME_UPPER") {
        prim->set_pad_mode(mindspore::PadMode::SAME);
      } else if (onnx_node_attr.s() == "SAME_LOWER") {
        MS_LOG(ERROR) << "PadMode_SAME_LOWER is not supported now";
        return nullptr;
      }
    }
    if (attribute_name == "pads") {
      if (onnx_node_attr.ints_size() == 4) {
        prim->set_pad_mode(mindspore::PadMode::PAD);
        pads.push_back(onnx_node_attr.ints(0));
        pads.push_back(onnx_node_attr.ints(2));
        pads.push_back(onnx_node_attr.ints(1));
        pads.push_back(onnx_node_attr.ints(3));
      }
    }
    if (attribute_name == "ceil_mode") {
      if (onnx_node_attr.i() == 0) {
        roundMode = mindspore::RoundMode::FLOOR;
      } else {
        roundMode = mindspore::RoundMode::CEIL;
      }
    }
    if (attribute_name == "dilations") {
      MS_LOG(ERROR) << "pooling op not support dilations now";
      return nullptr;
    }
  }
  prim->set_round_mode(roundMode);

  if (pads.empty()) {
    pads = {0, 0, 0, 0};
  }
  prim->set_pad(pads);

  if (strides.empty()) {
    strides.push_back(1);
    strides.push_back(1);
  }
  prim->set_strides(strides);

  prim->set_global(onnx_node.op_type() == "GlobalMaxPool");

  return prim.release();
}

OnnxNodeRegistrar g_onnxAveragePoolParser("AveragePool", new OnnxAvgPoolParser());
OnnxNodeRegistrar g_onnxGlobalAveragePoolParser("GlobalAveragePool", new OnnxAvgPoolParser());
OnnxNodeRegistrar g_onnxInt8AveragePoolParser("Int8AveragePool", new OnnxAvgPoolParser());

OnnxNodeRegistrar g_onnxMaxPoolParser("MaxPool", new OnnxMaxPoolParser());
OnnxNodeRegistrar g_onnxGlobalMaxPoolParser("GlobalMaxPool", new OnnxMaxPoolParser());
}  // namespace lite
}  // namespace mindspore
