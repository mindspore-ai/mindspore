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

#include "tools/converter/parser/onnx/onnx_custom_op_parser.h"
#include <memory>
#include <vector>
#include "tools/converter/parser/onnx/onnx_model_parser.h"
#include "tools/converter/ops/ops_def.h"
#include "mindspore/core/ops/op_name.h"
#include "nnacl/op_base.h"
#include "ops/affine_grid.h"
#include "ops/histogram.h"
#include "ops/auto_generate/gen_lite_ops.h"
#include "ops/ops_func_impl/xlogy.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxAffineGridParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_shared<ops::AffineGrid>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "align_corners") {
      prim->set_align_corners(static_cast<bool>(onnx_node_attr.i()));
    } else if (attribute_name == "size") {
      auto prim_c = prim->GetPrim();
      MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
      auto size_attr_size = onnx_node_attr.ints().size();
      std::vector<int32_t> size;
      size.reserve(size_attr_size);
      for (int idx = 0; idx < size_attr_size; idx++) {
        size.emplace_back(onnx_node_attr.ints(idx));
      }
      prim_c->AddAttr(ops::kSize, MakeValue<std::vector<int32_t>>(size));
    }
  }
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  return prim->GetPrim();
}

PrimitiveCPtr OnnxHistogramParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_shared<ops::Histogram>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "bins") {
      prim->set_bins(static_cast<int64_t>(onnx_node_attr.i()));
    } else if (attribute_name == "max") {
      prim->set_max(static_cast<float>(onnx_node_attr.i()));
    } else if (attribute_name == "min") {
      prim->set_min(static_cast<float>(onnx_node_attr.i()));
    }
  }
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  return prim->GetPrim();
}

PrimitiveCPtr OnnxLogicalNotParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_shared<ops::LogicalNot>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxRot90Parser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_shared<Rot90>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "dims") {
      auto dim_size = onnx_node_attr.ints().size();
      std::vector<int64_t> dims;
      dims.reserve(dim_size);
      for (int idx = 0; idx < dim_size; idx++) {
        dims.emplace_back(onnx_node_attr.ints(idx));
      }
      prim->AddAttr(ops::kDims, MakeValue(dims));
    } else if (attribute_name == "axis") {
      auto dim_size = onnx_node_attr.ints().size();
      std::vector<int64_t> axis;
      axis.reserve(dim_size);
      for (int idx = 0; idx < dim_size; idx++) {
        axis.emplace_back(onnx_node_attr.ints(idx));
      }
      prim->AddAttr(ops::kAxis, MakeValue(axis));
    }
  }
  return prim;
}

PrimitiveCPtr OnnxXlogyParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_shared<ops::Xlogy>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr OnnxRandomUniformLikeParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<RandomUniformLike>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "dtype") {
      auto onnx_dtype = static_cast<onnx::TensorProto_DataType>(onnx_node_attr.i());
      auto data_type = OnnxNodeParser::GetDataTypeFromOnnx(onnx_dtype);
      prim->AddAttr("dtype", MakeValue(static_cast<int64_t>(data_type)));
    } else if (attribute_name == "high") {
      prim->AddAttr("high", MakeValue(static_cast<float>(onnx_node_attr.f())));
    } else if (attribute_name == "low") {
      prim->AddAttr("low", MakeValue(static_cast<float>(onnx_node_attr.f())));
    } else if (attribute_name == "seed") {
      prim->AddAttr(ops::kSeed, MakeValue(static_cast<float>(onnx_node_attr.f())));
    }
  }
  return prim;
}

OnnxNodeRegistrar g_onnxAffineGridParser("affine_grid", new OnnxAffineGridParser());
OnnxNodeRegistrar g_onnxHistogramParser("histc", new OnnxHistogramParser());
OnnxNodeRegistrar g_onnxLogicalNotParser("logical_not", new OnnxLogicalNotParser());
OnnxNodeRegistrar g_onnxRot90Parser("rot90", new OnnxRot90Parser());
OnnxNodeRegistrar g_onnxXlogyParser("xlogy", new OnnxXlogyParser());
OnnxNodeRegistrar g_onnxRandomUniformLikeParser("RandomUniformLike", new OnnxRandomUniformLikeParser());
}  // namespace lite
}  // namespace mindspore
