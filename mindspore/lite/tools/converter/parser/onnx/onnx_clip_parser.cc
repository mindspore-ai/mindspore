/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/onnx/onnx_clip_parser.h"
#include <memory>
#include "ops/clip.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kClipInputIndex1 = 1;
constexpr int kClipInputIndex2 = 2;
}  // namespace

STATUS ParseAttr(const google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorProto> node_iter,
                 float *value) {
  if (value == nullptr) {
    MS_LOG(ERROR) << "parameter is null.";
    return RET_ERROR;
  }

  size_t element_size;
  switch (node_iter->data_type()) {
    case onnx::TensorProto_DataType_FLOAT:
      if (node_iter->float_data_size() > 0) {
        *value = node_iter->float_data(0);
      } else {
        element_size = node_iter->raw_data().size() / sizeof(float);
        if (element_size != 1) {
          MS_LOG(ERROR) << "element size is incorrect.";
          return RET_ERROR;
        }
        CHECK_NULL_RETURN(node_iter->raw_data().data());
        *value = *reinterpret_cast<const float *>(node_iter->raw_data().data());
      }
      break;
    case onnx::TensorProto_DataType_INT32:
      if (node_iter->int32_data_size() > 0) {
        *value = static_cast<float>(node_iter->int32_data(0));
      } else {
        element_size = node_iter->raw_data().size() / sizeof(int32_t);
        if (element_size != 1) {
          MS_LOG(ERROR) << "element size is incorrect.";
          return RET_ERROR;
        }
        CHECK_NULL_RETURN(node_iter->raw_data().data());
        *value = static_cast<float>(*reinterpret_cast<const int32_t *>(node_iter->raw_data().data()));
      }
      break;
    case onnx::TensorProto_DataType_INT64:
      if (node_iter->int64_data_size() > 0) {
        *value = static_cast<float>(node_iter->int64_data(0));
      } else {
        element_size = node_iter->raw_data().size() / sizeof(int64_t);
        if (element_size != 1) {
          MS_LOG(ERROR) << "element size is incorrect.";
          return RET_ERROR;
        }
        CHECK_NULL_RETURN(node_iter->raw_data().data());
        *value = static_cast<float>(*reinterpret_cast<const int64_t *>(node_iter->raw_data().data()));
      }
      break;
    default:
      MS_LOG(ERROR) << "do not support data_type: " << node_iter->data_type();
      return RET_ERROR;
  }
  return RET_OK;
}

PrimitiveCPtr OnnxClipParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Clip>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_min(-FLT_MAX);
  prim->set_max(FLT_MAX);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "max") {
      prim->set_max(onnx_node_attr.f());
    } else if (attribute_name == "min") {
      prim->set_min(onnx_node_attr.f());
    }
  }
  for (int i = 1; i < onnx_node.input_size(); i++) {
    const auto &input_name = onnx_node.input(i);
    auto node_iter = std::find_if(onnx_graph.initializer().begin(), onnx_graph.initializer().end(),
                                  [input_name](const onnx::TensorProto &proto) { return proto.name() == input_name; });
    if (node_iter == onnx_graph.initializer().end()) {
      MS_LOG(INFO) << "not find node: " << input_name;
      return prim->GetPrim();
    }
    float value = 0.0;
    if (ParseAttr(node_iter, &value) != RET_OK) {
      MS_LOG(ERROR) << "parse attr failed.";
      return nullptr;
    }
    if (i == kClipInputIndex1) {
      prim->set_min(value);
    } else if (i == kClipInputIndex2) {
      prim->set_max(value);
    }
  }
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxClipParser("Clip", new OnnxClipParser());
}  // namespace lite
}  // namespace mindspore
