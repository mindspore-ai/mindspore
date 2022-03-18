/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include <map>
#include <memory>
#include <string>
#include "schema/model_generated.h"
#include "src/tensor.h"
#include "src/common/utils.h"
#include "nnacl/infer/common_infer.h"
#include "nnacl/infer/infer_register.h"
#include "common/graph_kernel/lite_adapter/common/graph_kernel_op_parameter.h"

namespace mindspore::graphkernel {
namespace {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

std::vector<std::string> SplitString(const std::string &raw_str, char delimiter) {
  std::vector<std::string> res;
  std::string::size_type last_pos = 0;
  auto cur_pos = raw_str.find(delimiter);
  while (cur_pos != std::string::npos) {
    (void)res.emplace_back(raw_str.substr(last_pos, cur_pos - last_pos));
    cur_pos++;
    last_pos = cur_pos;
    cur_pos = raw_str.find(delimiter, cur_pos);
  }
  if (last_pos < raw_str.size()) {
    (void)res.emplace_back(raw_str.substr(last_pos, raw_str.size() - last_pos + 1));
  }
  return res;
}

int GetCustomShape(const std::string &attr, std::vector<std::vector<int>> *shapes) {
  auto split_shape_str = SplitString(attr, ',');
  for (size_t i = 0; i < split_shape_str.size(); i++) {
    size_t dim = std::stoul(split_shape_str[i]);
    if (i + dim >= split_shape_str.size()) {
      MS_LOG(ERROR) << "Shape string is invalid. The shape dim is " << dim << ", but only "
                    << split_shape_str.size() - i << " values follow.";
      return RET_ERROR;
    }
    std::vector<int> shape;
    for (size_t j = i + 1; j <= i + dim; j++) {
      shape.push_back(std::stoi(split_shape_str[j]));
    }
    i += dim;
    shapes->push_back(shape);
  }
  return RET_OK;
}

int SetOutputsShape(TensorC **outputs, size_t outputs_size, const std::string &outputs_shape_str) {
  std::vector<std::vector<int>> shapes;
  if (GetCustomShape(outputs_shape_str, &shapes) != RET_OK) {
    return RET_ERROR;
  }
  if (shapes.size() != outputs_size) {
    MS_LOG(ERROR) << "The saved outputs is not equal to the outputs_size: " << shapes.size() << " vs " << outputs_size;
    return RET_ERROR;
  }
  for (size_t i = 0; i < outputs_size; i++) {
    if (shapes[i].size() > MAX_SHAPE_SIZE) {
      MS_LOG(ERROR) << "The output shape size " << shapes.size() << " is greater than max size " << MAX_SHAPE_SIZE;
      return RET_ERROR;
    }
    for (size_t j = 0; j < shapes[i].size(); j++) {
      outputs[i]->shape_[j] = shapes[i][j];
    }
    outputs[i]->shape_size_ = shapes[i].size();
  }
  return RET_OK;
}

int SetOutputsFormat(TensorC **outputs, size_t outputs_size, const std::string &output_format_str) {
  auto formats = SplitString(output_format_str, ',');
  if (formats.size() != outputs_size) {
    MS_LOG(ERROR) << "The saved outputs is not equal to the outputs_size: " << formats.size() << " vs " << outputs_size;
    return RET_ERROR;
  }
  for (size_t i = 0; i < formats.size(); i++) {
    outputs[i]->format_ = std::stoi(formats[i]);
  }
  return RET_OK;
}

int SetOutputsType(TensorC **outputs, size_t outputs_size, const std::string &output_type_str) {
  auto types = SplitString(output_type_str, ',');
  if (types.size() != outputs_size) {
    MS_LOG(ERROR) << "The saved outputs is not equal to the outputs_size: " << types.size() << " vs " << outputs_size;
    return RET_ERROR;
  }
  for (size_t i = 0; i < types.size(); i++) {
    outputs[i]->data_type_ = std::stoi(types[i]);
  }
  return RET_OK;
}
}  // namespace

int InferShape(const TensorC **inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
               OpParameter *parameter) {
  auto param = reinterpret_cast<GraphKernelParameter *>(parameter);
  auto prim = static_cast<schema::Primitive *>(param->prim_)->value_as_Custom();
  for (size_t i = 0; i < prim->attr()->size(); i++) {
    auto attr = prim->attr()->Get(i);
    if (attr->name()->str() == "outputs_shape") {
      std::string data(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
      if (SetOutputsShape(outputs, outputs_size, data) != RET_OK) {
        return RET_ERROR;
      }
    } else if (attr->name()->str() == "outputs_format") {
      std::string data(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
      if (SetOutputsFormat(outputs, outputs_size, data) != RET_OK) {
        return RET_ERROR;
      }
    } else if (attr->name()->str() == "outputs_type") {
      std::string data(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
      if (SetOutputsType(outputs, outputs_size, data) != RET_OK) {
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::graphkernel

#ifdef __cplusplus
extern "C" {
#endif
int GraphKernelInferShape(const TensorC **inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                          OpParameter *parameter) {
  return mindspore::graphkernel::InferShape(inputs, inputs_size, outputs, outputs_size, parameter);
}
#ifdef __cplusplus
}
#endif

REG_INFER(GraphKernel, PrimType_Inner_GraphKernel, GraphKernelInferShape)
