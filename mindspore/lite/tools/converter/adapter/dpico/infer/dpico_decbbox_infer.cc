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

#include "infer/dpico_decbbox_infer.h"
#include <iostream>
#include <vector>
#include <memory>
#include "common/op_enum.h"
#include "utils/log_adapter.h"
#include "common/infer_util.h"
#include "common/op_attr.h"
#include "include/errorcode.h"
#include "include/registry/register_kernel_interface.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
namespace kernel {
namespace {
constexpr int kDimsOfBbox = 6;  // [xmin, ymin, xmax, ymax, score class_id]
}  // namespace
std::shared_ptr<KernelInterface> DpicoDecBBoxInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<DpicoDecBBoxInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "make shared failed, infer is nullptr.";
    return nullptr;
  }
  return infer;
}
Status DpicoDecBBoxInterface::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                                    const schema::Primitive *primitive, const kernel::Kernel *kernel) {
  auto status = dpico::CheckCustomInputOutput(inputs, outputs, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Check custom input output failed.";
    return kLiteError;
  }
  auto param = primitive->value_as_Custom();
  if (dpico::CheckCustomParam(param, "DecBBox") != RET_OK) {
    MS_LOG(ERROR) << "custom param is invalid.";
    return kLiteError;
  }
  if (param->attr() == nullptr) {
    MS_LOG(ERROR) << "param attr is nullptr.";
    return kLiteError;
  }
  bool has_num_classes = false;
  uint32_t num_classes = 0;
  for (uint32_t i = 0; i < static_cast<uint32_t>(param->attr()->size()); i++) {
    if (param->attr()->Get(i) == nullptr || param->attr()->Get(i)->name() == nullptr) {
      MS_LOG(ERROR) << "param->attr()->Get(i) is nullptr or param->attr()->Get(i)->name() is nullptr";
      return kLiteError;
    }
    if (param->attr()->Get(i)->name()->str() == dpico::kNumClasses) {
      auto output_info = param->attr()->Get(i)->data();
      if (output_info == nullptr) {
        MS_LOG(ERROR) << "output_shape is nullptr";
        return kLiteError;
      }
      if (memcpy_s(&num_classes, sizeof(uint32_t), output_info->data(), output_info->size()) != EOK) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return kLiteError;
      }
      has_num_classes = true;
      break;
    }
  }
  if (!has_num_classes) {
    MS_LOG(ERROR) << dpico::kNumClasses << " is needed for DecBBox";
    return kLiteError;
  }

  const auto &input = (*inputs)[0];
  auto &output = (*outputs)[0];
  std::vector<int64_t> output_shape(input.Shape());
  if (output_shape.size() != dpico::kDims4) {
    MS_LOG(ERROR) << "output_shape should be 4 dims, which is " << output_shape.size();
    return kLiteError;
  }
  output_shape.at(dpico::kAxis1) = num_classes;
  output_shape.at(dpico::kAxis2) = kDimsOfBbox;
  output.SetShape(output_shape);
  return kSuccess;
}
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, DecBBox, DpicoDecBBoxInferCreater)
}  // namespace kernel
}  // namespace mindspore
