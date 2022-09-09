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
#include <unordered_map>
#include <memory>
#include <string>
#include "tools/graph_kernel/common/utils.h"
#include "schema/model_generated.h"
#include "src/tensor.h"
#include "src/common/utils.h"
#include "nnacl/infer/common_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/custom_parameter.h"

namespace mindspore::graphkernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace {
int SetOutputsShape(TensorC **outputs, size_t outputs_size, const std::string &outputs_shape_str, int batch) {
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
      MS_LOG(ERROR) << "The output shape size " << shapes[i].size() << " is greater than max size " << MAX_SHAPE_SIZE;
      return RET_ERROR;
    }
    for (size_t j = 0; j < shapes[i].size(); j++) {
      outputs[i]->shape_[j] = j == 0 ? shapes[i][j] * batch : shapes[i][j];
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
int InferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
               OpParameter *parameter) {
  // in PopulateCustomParameter, the primitive is store in attr_data[0]
  auto param = reinterpret_cast<CustomParameter *>(parameter)->attr_data[0];
  auto prim = reinterpret_cast<schema::Primitive *>(param)->value_as_Custom();
  std::unordered_map<std::string, std::string> attr_map;
  for (size_t i = 0; i < prim->attr()->size(); i++) {
    auto attr = prim->attr()->Get(i);
    std::string data;
    if (attr->name()->str() == "inputs_shape") {
      data = std::string(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
    } else if (attr->name()->str() == "outputs_shape") {
      data = std::string(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
    } else if (attr->name()->str() == "outputs_format") {
      data = std::string(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
    } else if (attr->name()->str() == "outputs_type") {
      data = std::string(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
    } else if (attr->name()->str() == "dynamic_input_index") {
      data = std::string(reinterpret_cast<const char *>(attr->data()->Data()), attr->data()->size());
    } else {
      continue;
    }
    (void)attr_map.emplace(attr->name()->str(), data);
  }
  int batch = 1;

  if (attr_map.count("inputs_shape") != 0 && attr_map.count("dynamic_input_index") != 0) {
    std::vector<std::vector<int>> shapes;
    if (GetCustomShape(attr_map["inputs_shape"], &shapes) != RET_OK) {
      return RET_ERROR;
    }
    std::vector<size_t> index;
    GetCustomIndex(attr_map["dynamic_input_index"], &index);
    if (CalculateDynamicBatchSize(inputs, inputs_size, shapes, index, &batch) != RET_OK) {
      return RET_ERROR;
    }
  }
  if (attr_map.count("outputs_shape") == 0 ||
      SetOutputsShape(outputs, outputs_size, attr_map["outputs_shape"], batch) != RET_OK) {
    return RET_ERROR;
  }
  if (attr_map.count("outputs_format") == 0 ||
      SetOutputsFormat(outputs, outputs_size, attr_map["outputs_format"]) != RET_OK) {
    return RET_ERROR;
  }
  if (attr_map.count("outputs_type") == 0 ||
      SetOutputsType(outputs, outputs_size, attr_map["outputs_type"]) != RET_OK) {
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace mindspore::graphkernel

#ifdef __cplusplus
extern "C" {
#endif
int GraphKernelInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                          OpParameter *parameter) {
  return mindspore::graphkernel::InferShape(inputs, inputs_size, outputs, outputs_size, parameter);
}
#ifdef __cplusplus
}
#endif

REG_INFER(GraphKernel, PrimType_Inner_GraphKernel, GraphKernelInferShape)
