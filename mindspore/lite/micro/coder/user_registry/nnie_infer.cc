/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "coder/user_registry/nnie_infer.h"
#include <string>
#include <iostream>
#include "include/errorcode.h"
#include "include/api/format.h"
#include "include/registry/register_kernel_interface.h"
#include "utils/log_adapter.h"

using mindspore::kernel::KernelInterface;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Custom;
#define MAX_SIZE 1024

namespace mindspore {
namespace nnie {
std::shared_ptr<KernelInterface> CustomInferCreater() {
  auto infer = new (std::nothrow) CustomInterface();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "new custom infer is nullptr";
    return nullptr;
  }
  return std::shared_ptr<KernelInterface>(infer);
}

int GetCustomShape(const mindspore::schema::Custom *op, const std::string &attr,
                   std::vector<std::vector<int64_t>> *shapes) {
  char buf[MAX_SIZE];
  bool has_outputs_shape = false;

  for (size_t i = 0; i < op->attr()->size(); i++) {
    if (op->attr()->Get(i)->name()->str() == attr) {
      auto output_info = op->attr()->Get(i)->data();
      int attr_size = static_cast<int>(output_info->size());
      if (attr_size >= MAX_SIZE) {
        MS_LOG(ERROR) << "attr size too big";
        return RET_ERROR;
      }
      for (int j = 0; j < attr_size; j++) {
        buf[j] = static_cast<char>(output_info->Get(j));
      }
      buf[attr_size] = 0;
      has_outputs_shape = true;
      break;
    }
  }

  if (!has_outputs_shape) {
    MS_LOG(ERROR) << "Custom op don't have " << attr.c_str() << " attr.";
    return RET_ERROR;
  }

  char delims[] = ",";
  char *res = nullptr;
  char *save_ptr = nullptr;
  res = strtok_r(buf, delims, &save_ptr);
  while (res != nullptr) {
    int64_t ndims = strtol(res, &res, 10);
    int j = 0;
    std::vector<int64_t> shape;
    shape.resize(ndims);
    for (; j < ndims; j++) {
      res = strtok_r(NULL, delims, &save_ptr);
      shape[j] = static_cast<int64_t>(strtol(res, &res, 10));
    }
    shapes->push_back(shape);

    res = strtok_r(NULL, delims, &save_ptr);
  }
  return RET_OK;
}

Status CustomInterface::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                              const mindspore::schema::Primitive *primitive) {
  if (inputs->empty()) {
    MS_LOG(ERROR) << "Inputs size 0";
    return kLiteError;
  }
  if (outputs->empty()) {
    MS_LOG(ERROR) << "Outputs size 0";
    return kLiteError;
  }
  if (primitive->value_type() != mindspore::schema::PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Primitive type is not PrimitiveType_Custom";
    return kLiteError;
  }

  auto op = primitive->value_as_Custom();
  if (op->attr()->size() < 1) {
    MS_LOG(ERROR) << "There are at least 1 attribute of Custom";
    return kLiteError;
  }
  std::vector<std::vector<int64_t>> inputs_shape;
  if (GetCustomShape(op, "inputs_shape", &inputs_shape) != RET_OK) {
    MS_LOG(ERROR) << "parser inputs_shape attribute err.";
    return kLiteError;
  }
  std::vector<std::vector<int64_t>> outputs_shape;
  if (GetCustomShape(op, "outputs_shape", &outputs_shape) != RET_OK) {
    MS_LOG(ERROR) << "parser outputs_shape attribute err.";
    return kLiteError;
  }
  if (inputs_shape.size() != (inputs->size() - 1)) {
    MS_LOG(ERROR) << "inputs num diff inputs_shape num.";
    return kLiteError;
  }
  if (inputs_shape[0].size() != (*inputs)[0].Shape().size()) {
    MS_LOG(ERROR) << "shape size err.";
    return kLiteError;
  }
  bool resize_flag = false;
  int resize_num = 1;
  for (size_t i = 0; i < inputs_shape[0].size(); i++) {
    if (inputs_shape[0][i] != (*inputs)[0].Shape()[i]) {
      if (i == 0) {
        resize_flag = true;
        resize_num = (*inputs)[0].Shape()[i];
      } else {
        MS_LOG(ERROR) << "Custom of NNIE only support batch_num resize.";
        return kLiteError;
      }
    }
  }
  if (resize_flag) {
    for (auto &output_shape : outputs_shape) {
      output_shape[0] = resize_num;
    }
  }
  for (size_t i = 0; i < outputs->size(); i++) {
    (*outputs)[i].SetShape(outputs_shape[i]);
    (*outputs)[i].SetDataType(DataType::kNumberTypeFloat32);
    (*outputs)[i].SetFormat(Format::NCHW);
  }
  return kSuccess;
}
}  // namespace nnie
}  // namespace mindspore
namespace mindspore {
namespace kernel {
REGISTER_CUSTOM_KERNEL_INTERFACE(NNIE, NNIE, nnie::CustomInferCreater);
}  // namespace kernel
}  // namespace mindspore
