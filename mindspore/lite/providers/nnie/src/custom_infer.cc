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

#include "src/custom_infer.h"
#include <string>
#include <iostream>
#include "include/errorcode.h"
#include "src/nnie_print.h"
#include "include/api/format.h"
#include "include/registry/register_kernel_interface.h"

using mindspore::kernel::KernelInterface;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Custom;

namespace mindspore {
namespace nnie {
std::shared_ptr<KernelInterface> CustomInferCreater() {
  auto infer = new (std::nothrow) CustomInterface();
  if (infer == nullptr) {
    LOGE("new custom infer is nullptr");
    return nullptr;
  }
  return std::shared_ptr<KernelInterface>(infer);
}

int GetCustomShape(const mindspore::schema::Custom *op, const std::string &attr,
                   std::vector<std::vector<int64_t>> *shapes) {
  char buf[kMaxSize];
  bool has_outputs_shape = false;

  for (size_t i = 0; i < op->attr()->size(); i++) {
    if (op->attr()->Get(i)->name()->str() == attr) {
      auto output_info = op->attr()->Get(i)->data();
      int attr_size = static_cast<int>(output_info->size());
      if (attr_size >= kMaxSize) {
        LOGE("attr size too big");
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
    LOGE("Custom op don't have %s attr.", attr.c_str());
    return RET_ERROR;
  }

  char delims[] = ",";
  char *res = nullptr;
  char *save_ptr = nullptr;
  res = strtok_r(buf, delims, &save_ptr);
  while (res != nullptr) {
    int64_t ndims = strtol(res, &res, kDecimal);
    int j = 0;
    std::vector<int64_t> shape;
    shape.resize(ndims);
    for (; j < ndims; j++) {
      res = strtok_r(NULL, delims, &save_ptr);
      shape[j] = static_cast<int64_t>(strtol(res, &res, kDecimal));
    }
    shapes->push_back(shape);

    res = strtok_r(NULL, delims, &save_ptr);
  }
  return RET_OK;
}

Status CustomInterface::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                              const mindspore::schema::Primitive *primitive) {
  if (inputs->empty()) {
    LOGE("Inputs size 0");
    return kLiteError;
  }
  if (outputs->empty()) {
    LOGE("Outputs size 0");
    return kLiteError;
  }
  if (primitive->value_type() != mindspore::schema::PrimitiveType_Custom) {
    LOGE("Primitive type is not PrimitiveType_Custom");
    return kLiteError;
  }

  auto op = primitive->value_as_Custom();
  if (op->attr()->size() < 1) {
    LOGE("There are at least 1 attribute of Custom");
    return kLiteError;
  }
  std::vector<std::vector<int64_t>> inputs_shape;
  if (GetCustomShape(op, "inputs_shape", &inputs_shape) != RET_OK) {
    LOGE("parser inputs_shape attribute err.");
    return kLiteError;
  }
  std::vector<std::vector<int64_t>> outputs_shape;
  if (GetCustomShape(op, "outputs_shape", &outputs_shape) != RET_OK) {
    LOGE("parser outputs_shape attribute err.");
    return kLiteError;
  }
  if (inputs_shape.size() != (inputs->size() - 1)) {
    LOGE("inputs num diff inputs_shape num.");
    return kLiteError;
  }
  if (inputs_shape[0].size() != (*inputs)[0].Shape().size()) {
    LOGE("shape size err.");
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
        LOGE("Custom of NNIE only support batch_num resize.");
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
