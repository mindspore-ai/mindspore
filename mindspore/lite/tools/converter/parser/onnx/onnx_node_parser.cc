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

#include "tools/converter/parser/onnx/onnx_node_parser.h"
#include <algorithm>
#include <vector>
#include <memory>
#include "tools/converter/parser/onnx/onnx_model_parser.h"

namespace mindspore {
namespace lite {
int OnnxNodeParser::opset_version_ = 0;

mindspore::PadMode OnnxNodeParser::GetOnnxPadMode(const onnx::AttributeProto &onnx_node_attr) {
  if (onnx_node_attr.s() == "NOTSET") {
    return mindspore::PadMode::PAD;
  } else if (onnx_node_attr.s() == "SAME_UPPER" || onnx_node_attr.s() == "SAME_LOWER") {
    return mindspore::PadMode::SAME;
  } else if (onnx_node_attr.s() == "VALID") {
    return mindspore::PadMode::VALID;
  } else {
    MS_LOG(ERROR) << "unsupported padMode";
    return mindspore::PadMode::PAD;
  }
}

STATUS OnnxNodeParser::GetPadMode(const onnx::AttributeProto &onnx_node_attr, std::string *mode) {
  if (onnx_node_attr.s() == "NOTSET") {
    *mode = "NOTSET";
    return RET_OK;
  } else if (onnx_node_attr.s() == "SAME_UPPER") {
    *mode = "SAME_UPPER";
    return RET_OK;
  } else if (onnx_node_attr.s() == "SAME_LOWER") {
    *mode = "SAME_LOWER";
    return RET_OK;
  } else if (onnx_node_attr.s() == "VALID") {
    *mode = "VALID";
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "unsupported padMode";
    *mode = "NOTSET";
    return RET_ERROR;
  }
}

STATUS OnnxNodeParser::GetTensorDataFromOnnx(const onnx::TensorProto &onnx_tensor, std::vector<float> *value,
                                             int *type) {
  MS_ASSERT(value != nullptr);
  MS_ASSERT(type != nullptr);
  size_t data_count = 1;
  std::for_each(onnx_tensor.dims().begin(), onnx_tensor.dims().end(), [&data_count](int dim) { data_count *= dim; });
  switch (onnx_tensor.data_type()) {
    case onnx::TensorProto_DataType_FLOAT:
      *type = OnnxModelParser::GetDataTypeFromOnnx(onnx::TensorProto_DataType_FLOAT);
      if (onnx_tensor.float_data_size() > 0) {
        for (int i = 0; i < onnx_tensor.float_data_size(); i++) {
          value->push_back(onnx_tensor.float_data(i));
        }
      } else {
        for (size_t i = 0; i < data_count; i++) {
          value->push_back(reinterpret_cast<const float *>(onnx_tensor.raw_data().data())[i]);
        }
      }
      break;
    case onnx::TensorProto_DataType_INT32:
      *type = OnnxModelParser::GetDataTypeFromOnnx(onnx::TensorProto_DataType_INT32);
      if (onnx_tensor.int32_data_size() > 0) {
        for (int i = 0; i < onnx_tensor.int32_data_size(); i++) {
          value->push_back(onnx_tensor.int32_data(i));
        }
      } else {
        for (size_t i = 0; i < data_count; i++) {
          value->push_back(static_cast<float>(reinterpret_cast<const int32_t *>(onnx_tensor.raw_data().data())[i]));
        }
      }
      break;
    case onnx::TensorProto_DataType_INT64:
      *type = OnnxModelParser::GetDataTypeFromOnnx(onnx::TensorProto_DataType_INT32);
      if (onnx_tensor.int64_data_size() > 0) {
        for (int i = 0; i < onnx_tensor.int64_data_size(); i++) {
          value->push_back(onnx_tensor.int64_data(i));
        }
      } else {
        for (size_t i = 0; i < data_count; i++) {
          value->push_back(static_cast<float>(reinterpret_cast<const int64_t *>(onnx_tensor.raw_data().data())[i]));
        }
      }
      break;
    default:
      MS_LOG(ERROR) << "The data type is not supported.";
      return RET_ERROR;
  }
  return RET_OK;
}

void OnnxNodeParser::Split(const std::string &src_str, std::vector<std::string> *dst_str, const std::string &chr) {
  std::string ::size_type p1 = 0, p2 = src_str.find(chr);
  while (std::string::npos != p2) {
    dst_str->push_back(src_str.substr(p1, p2 - p1));
    p1 = p2 + chr.size();
    p2 = src_str.find(chr, p1);
  }
  if (p1 != src_str.length()) {
    dst_str->push_back(src_str.substr(p1));
  }
}
}  // namespace lite
}  // namespace mindspore
