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
#include "tools/converter/parser/tf/tf_fill_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
STATUS TFFillParser::Parse(const tensorflow::NodeDef &tf_op,
                           const std::map<string, const tensorflow::NodeDef *> &tf_node_map, PrimitiveC **primitiveC,
                           std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF FillParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::FillT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  inputs->emplace_back(tf_op.input(1));
  // parse dims
  tensorflow::AttrValue attr_value;
  auto dims_node = GetConstInputNode(tf_node_map, tf_op.input(0));
  if (dims_node != nullptr) {
    if (!TensorFlowUtils::FindAttrValue(*dims_node, "value", &attr_value)) {
      MS_LOG(ERROR) << "fill dims input not have value attr";
      return RET_ERROR;
    }
    if (attr_value.value_case() != tensorflow::AttrValue::kTensor) {
      MS_LOG(ERROR) << "The attrValue of value should have tensor type, actual: " << attr_value.value_case()
                    << ", node: " << tf_op.name().c_str();
      return RET_ERROR;
    }
    const tensorflow::TensorProto &dims_tensor = attr_value.tensor();
    if (dims_tensor.dtype() != tensorflow::DT_INT32) {
      MS_LOG(ERROR) << "The dimsTensor dataType should be DT_INT32, actual : " << dims_tensor.dtype();
      return RET_ERROR;
    }
    const tensorflow::TensorShapeProto &dims_tensor_shape = dims_tensor.tensor_shape();
    size_t shape_size = 1;
    for (int i = 0; i < dims_tensor_shape.dim_size(); i++) {
      shape_size *= dims_tensor_shape.dim(i).size();
    }
    size_t size = dims_tensor.int_val().size();
    if (size > 0) {
      for (size_t i = 0; i < shape_size; i++) {
        attr->dims.emplace_back(dims_tensor.int_val().Get(i));
      }
    } else {
      size = dims_tensor.tensor_content().length();
      if (size > 0) {
        if (size != shape_size * sizeof(int32_t)) {
          MS_LOG(ERROR) << "tensor size mismatch";
          return RET_ERROR;
        }
        attr->dims.resize(shape_size);
        if (EOK != ::memcpy_s(attr->dims.data(), size, dims_tensor.tensor_content().data(), size)) {
          MS_LOG(ERROR) << "Memcpy_s from dimsTensor to attr failed";
          return RET_ERROR;
        }
      } else {
        MS_LOG(DEBUG) << "empty dims";
      }
    }
  } else {
    inputs->emplace_back(tf_op.input(0));
  }

  primitive->value.type = schema::PrimitiveType_Fill;
  primitive->value.value = attr.release();
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }
  *output_size = 1;
  return RET_OK;
}
TFNodeRegistrar g_tfFillParser("Fill", new TFFillParser());
}  // namespace lite
}  // namespace mindspore
