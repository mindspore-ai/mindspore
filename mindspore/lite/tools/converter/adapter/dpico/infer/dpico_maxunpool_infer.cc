/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "infer/dpico_maxunpool_infer.h"
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
std::shared_ptr<KernelInterface> DpicoMaxunpoolInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<DpicoMaxunpoolInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "make shared failed, infer is nullptr.";
    return nullptr;
  }
  return infer;
}
Status DpicoMaxunpoolInterface::Infer(std::vector<mindspore::MSTensor> *inputs,
                                      std::vector<mindspore::MSTensor> *outputs, const schema::Primitive *primitive,
                                      const kernel::Kernel *) {
  auto status = dpico::CheckCustomInputOutput(inputs, outputs, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Check custom input output failed.";
    return kLiteError;
  }
  auto param = primitive->value_as_Custom();
  if (dpico::CheckCustomParam(param, "MaxUnpool") != RET_OK) {
    MS_LOG(ERROR) << "custom param is invalid.";
    return kLiteError;
  }

  // Get param value
  std::map<std::string, const flatbuffers::Vector<uint8_t> *> custom_attrs;
  int32_t strides = 0;
  int32_t kernel_shape = 0;
  uint32_t maxunpool_h = 0;
  uint32_t maxunpool_w = 0;
  bool has_strides = false;
  bool has_kernel_shape = false;
  if (param->attr() == nullptr) {
    MS_LOG(ERROR) << "param->attr() is nullptr";
    return kLiteError;
  }
  for (uint32_t i = 0; i < static_cast<uint32_t>(param->attr()->size()); i++) {
    if (param->attr()->Get(i) == nullptr || param->attr()->Get(i)->name() == nullptr ||
        param->attr()->Get(i)->data() == nullptr) {
      MS_LOG(ERROR) << "param->attr()->Get " << i << " is nullptr or param->attr()->Get(i)->name() or data is nullptr";
      return kLiteError;
    }
    (void)custom_attrs.emplace(std::pair(param->attr()->Get(i)->name()->str(), param->attr()->Get(i)->data()));
  }
  if (custom_attrs.count(ops::kStrides) == 1) {
    if (memcpy_s(&strides, sizeof(int32_t), custom_attrs[ops::kStrides]->data(), custom_attrs[ops::kStrides]->size()) !=
        EOK) {
      MS_LOG(ERROR) << "strides memcpy_s failed.";
      return kLiteError;
    }
    has_strides = true;
  }
  if (custom_attrs.count(dpico::kKernelShape) == 1) {
    if (memcpy_s(&kernel_shape, sizeof(int32_t), custom_attrs[dpico::kKernelShape]->data(),
                 custom_attrs[dpico::kKernelShape]->size()) != EOK) {
      MS_LOG(ERROR) << "kernel_shape memcpy_s failed.";
      return kLiteError;
    }
    has_kernel_shape = true;
  }

  const auto &input = (*inputs)[0];
  if (input.Shape().size() != dpico::kDims4) {
    MS_LOG(ERROR) << "inputs_0's shape should be 4 dims, but now it's " << input.Shape().size() << " dims";
    return kLiteError;
  }

  if (has_strides && has_kernel_shape) {
    maxunpool_h = (input.Shape().at(dpico::kAxis2) - 1) * strides + kernel_shape;
    maxunpool_w = (input.Shape().at(dpico::kAxis3) - 1) * strides + kernel_shape;
  } else if (!has_strides && has_kernel_shape) {
    maxunpool_h = input.Shape().at(dpico::kAxis2) - 1 + kernel_shape;
    maxunpool_w = input.Shape().at(dpico::kAxis3) - 1 + kernel_shape;
  } else {
    MS_LOG(ERROR) << "kernel_shape attr should be provided.";
    return kLiteError;
  }

  auto &output = (*outputs)[0];
  std::vector<int64_t> output_shape(dpico::kDims4);
  output_shape[0] = input.Shape().at(0);
  output_shape[dpico::kAxis1] = input.Shape().at(1);
  output_shape[dpico::kAxis2] = maxunpool_h;
  output_shape[dpico::kAxis3] = maxunpool_w;
  output.SetShape(output_shape);

  return kSuccess;
}
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, MaxUnpool, DpicoMaxunpoolInferCreater)
}  // namespace kernel
}  // namespace mindspore
