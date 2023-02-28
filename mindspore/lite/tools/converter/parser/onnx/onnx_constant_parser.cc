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

#include "tools/converter/parser/onnx/onnx_constant_parser.h"
#include <vector>
#include <memory>
#include "tools/converter/parser/onnx/onnx_model_parser.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/common/tensor_util.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
STATUS OnnxConstantParser::AddDataInfoAttr(const onnx::TensorProto &onnx_const_tensor, PrimitiveCPtr prim) {
  MS_ASSERT(prim != nullptr);
  auto tensor_info = OnnxNodeParser::CopyOnnxTensorData(onnx_const_tensor);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "get value failed.";
    return RET_ERROR;
  }
  prim->set_attr("const_data", tensor_info);
  return RET_OK;
}

PrimitiveCPtr OnnxConstantParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_shared<lite::Constant>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &attr : onnx_node.attribute()) {
    if (attr.name() == "sparse_value") {
      MS_LOG(WARNING) << "sparse_value";
      continue;
    }
    if (attr.name() == "value") {
      const auto &const_tensor = attr.t();
      if (AddDataInfoAttr(const_tensor, prim) != RET_OK) {
        MS_LOG(ERROR) << "add basic attr failed.";
        return nullptr;
      }
    } else {
      MS_LOG(ERROR) << "processing Constant op attr " << attr.name() << " not implemented";
      return nullptr;
    }
  }
  return prim;
}

OnnxNodeRegistrar g_onnxConstantParser("Constant", new OnnxConstantParser());
}  // namespace lite
}  // namespace mindspore
