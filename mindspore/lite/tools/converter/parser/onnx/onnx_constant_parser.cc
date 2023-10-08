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

#include "tools/converter/parser/onnx/onnx_constant_parser.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include "tools/converter/parser/onnx/onnx_model_parser.h"
#include "tools/converter/parser/onnx/onnx_node_parser.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/common/tensor_util.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
STATUS OnnxConstantParser::AddDataInfoAttr(const onnx::TensorProto &onnx_const_tensor, PrimitiveCPtr prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, RET_ERROR);
  tensor::TensorPtr tensor_info;
  if (onnx_const_tensor.data_location() != onnx::TensorProto::EXTERNAL) {
    tensor_info = OnnxNodeParser::CopyOnnxTensorData(onnx_const_tensor);
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << "get value failed.";
      return RET_ERROR;
    }
    prim->set_attr("const_data", tensor_info);
  } else {  // Load constant data from external file
    std::string model_file = OnnxNodeParser::GetOnnxModelFile();
    if (model_file.empty()) {
      MS_LOG(ERROR) << "Loading constant data from external file failed! model file is empty.";
      return RET_ERROR;
    }
    auto data_type =
      OnnxNodeParser::GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(onnx_const_tensor.data_type()));
    if (data_type == kTypeUnknown) {
      MS_LOG(ERROR) << "not support onnx data type "
                    << static_cast<onnx::TensorProto_DataType>(onnx_const_tensor.data_type());
      return RET_ERROR;
    }
    std::vector<int64_t> shape_vector(onnx_const_tensor.dims().begin(), onnx_const_tensor.dims().end());
    tensor_info = std::make_shared<tensor::Tensor>(data_type, shape_vector);
    MS_CHECK_TRUE_MSG(tensor_info != nullptr, RET_ERROR, "create tensor_info return nullptr");
    std::vector<int> shape;
    std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(shape),
                   [](const int64_t &value) { return static_cast<int>(value); });

    std::map<std::string, std::pair<size_t, uint8_t *>> external_datas;
    auto free_external_data = [&external_datas]() {
      for (auto &item : external_datas) {
        if (item.second.second) {
          delete[] item.second.second;
        }
      }
      external_datas.clear();
    };

    auto status =
      OnnxNodeParser::LoadOnnxExternalTensorData(onnx_const_tensor, tensor_info, model_file, &external_datas);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "load external data failed";
      free_external_data();
      return status;
    }
    prim->set_attr("const_data", tensor_info);
    free_external_data();
  }

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
