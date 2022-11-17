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

#include "infer/dpico_upsample_infer.h"
#include <vector>
#include <memory>
#include <map>
#include <string>
#include "common/op_enum.h"
#include "common/op_attr.h"
#include "utils/log_adapter.h"
#include "common/infer_util.h"
#include "include/errorcode.h"
#include "ops/op_name.h"
#include "include/registry/register_kernel_interface.h"

using mindspore::kernel::KernelInterface;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
namespace kernel {
std::shared_ptr<KernelInterface> DpicoUpsampleInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<DpicoUpsampleInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "make shared failed, infer is nullptr.";
    return nullptr;
  }
  return infer;
}
Status DpicoUpsampleInterface::Infer(std::vector<mindspore::MSTensor> *inputs,
                                     std::vector<mindspore::MSTensor> *outputs, const schema::Primitive *primitive,
                                     const kernel::Kernel *kernel) {
  auto status = dpico::CheckCustomInputOutput(inputs, outputs, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Check custom input output failed.";
    return kLiteError;
  }
  auto param = primitive->value_as_Custom();
  if (dpico::CheckCustomParam(param, "Upsample") != RET_OK) {
    MS_LOG(ERROR) << "custom param is invalid.";
    return kLiteError;
  }

  // get param value
  std::map<std::string, const flatbuffers::Vector<uint8_t> *> custom_attrs;
  float scale = 0;
  uint32_t upsample_h = 0;
  uint32_t upsample_w = 0;
  bool has_scale = false;
  bool has_upsample_h = false;
  bool has_upsample_w = false;
  if (param->attr() == nullptr) {
    MS_LOG(ERROR) << "param->attr() is nullptr";
    return kLiteError;
  }
  for (uint32_t i = 0; i < static_cast<uint32_t>(param->attr()->size()); i++) {
    if (param->attr()->Get(i) == nullptr || param->attr()->Get(i)->name() == nullptr) {
      MS_LOG(ERROR) << "param->attr()->Get(i) is nullptr or param->attr()->Get(i)->name() is nullptr";
      return kLiteError;
    }
    (void)custom_attrs.emplace(std::pair(param->attr()->Get(i)->name()->str(), param->attr()->Get(i)->data()));
  }
  if (custom_attrs.count(ops::kScale) == 1) {
    if (memcpy_s(&scale, sizeof(float), custom_attrs[ops::kScale]->data(), custom_attrs[ops::kScale]->size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return kLiteError;
    }
    has_scale = true;
  }
  if (custom_attrs.count(dpico::kUpsampleH) == 1) {
    if (memcpy_s(&upsample_h, sizeof(uint32_t), custom_attrs[dpico::kUpsampleH]->data(),
                 custom_attrs[dpico::kUpsampleH]->size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return kLiteError;
    }
    has_upsample_h = true;
  }
  if (custom_attrs.count(dpico::kUpsampleW) == 1) {
    if (memcpy_s(&upsample_w, sizeof(uint32_t), custom_attrs[dpico::kUpsampleW]->data(),
                 custom_attrs[dpico::kUpsampleW]->size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return kLiteError;
    }
    has_upsample_w = true;
  }

  const auto &input = (*inputs)[0];
  if (input.Shape().size() != dpico::kDims4) {
    MS_LOG(ERROR) << "inputs_0's shape should be 4 dims, but now it's " << input.Shape().size() << " dims";
    return kLiteError;
  }

  if (has_scale) {
    if (!has_upsample_h) {
      upsample_h = input.Shape().at(dpico::kAxis2) * scale;
    }
    if (!has_upsample_w) {
      upsample_w = input.Shape().at(dpico::kAxis3) * scale;
    }
  } else if (!has_upsample_h || !has_upsample_w) {
    MS_LOG(ERROR) << "scale attr or (upsample_h && upsample_w) attr should be provided.";
    return kLiteError;
  }

  auto &output = (*outputs)[0];
  std::vector<int64_t> output_shape(dpico::kDims4);
  output_shape[0] = input.Shape().at(0);
  output_shape[dpico::kAxis1] = input.Shape().at(1);
  output_shape[dpico::kAxis2] = upsample_h;
  output_shape[dpico::kAxis3] = upsample_w;
  output.SetShape(output_shape);

  return kSuccess;
}
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Upsample, DpicoUpsampleInferCreater)
}  // namespace kernel
}  // namespace mindspore
