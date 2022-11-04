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

#include "infer/dpico_passthrough_infer.h"
#include <vector>
#include <memory>
#include <map>
#include <string>
#include "common/op_enum.h"
#include "common/op_attr.h"
#include "utils/log_adapter.h"
#include "common/infer_util.h"
#include "include/errorcode.h"
#include "include/registry/register_kernel_interface.h"

using mindspore::kernel::KernelInterface;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
namespace kernel {
std::shared_ptr<KernelInterface> DpicoPassThroughInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<DpicoPassThroughInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "make shared failed, infer is nullptr.";
    return nullptr;
  }
  return infer;
}
Status DpicoPassThroughInterface::Infer(std::vector<mindspore::MSTensor> *inputs,
                                        std::vector<mindspore::MSTensor> *outputs, const schema::Primitive *primitive,
                                        const kernel::Kernel *kernel) {
  auto status = dpico::CheckCustomInputOutput(inputs, outputs, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Check custom input output failed.";
    return kLiteError;
  }
  auto param = primitive->value_as_Custom();
  if (dpico::CheckCustomParam(param, "PassThrough") != RET_OK) {
    MS_LOG(ERROR) << "custom param is invalid.";
    return kLiteError;
  }

  // get param value
  std::map<std::string, const flatbuffers::Vector<uint8_t> *> custom_attrs;
  int64_t num_output = 0;
  int64_t block_height = 0;
  int64_t block_width = 0;
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
  if (custom_attrs.count(dpico::kNumOutput) == 1) {
    if (memcpy_s(&num_output, sizeof(uint32_t), custom_attrs[dpico::kNumOutput]->data(),
                 custom_attrs[dpico::kNumOutput]->size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return kLiteError;
    }
  } else {
    MS_LOG(ERROR) << "num_output attr doesn't exist.";
    return kLiteError;
  }
  if (custom_attrs.count(dpico::kBlockHeight) == 1) {
    if (memcpy_s(&block_height, sizeof(uint32_t), custom_attrs[dpico::kBlockHeight]->data(),
                 custom_attrs[dpico::kBlockHeight]->size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return kLiteError;
    }
    if (block_height == 0) {
      MS_LOG(ERROR) << "block_height shouldn't be 0.";
      return kLiteError;
    }
  } else {
    MS_LOG(ERROR) << "block_height attr doesn't exist.";
    return kLiteError;
  }
  if (custom_attrs.count(dpico::kBlockWidth) == 1) {
    if (memcpy_s(&block_width, sizeof(uint32_t), custom_attrs[dpico::kBlockWidth]->data(),
                 custom_attrs[dpico::kBlockWidth]->size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return kLiteError;
    }
    if (block_width == 0) {
      MS_LOG(ERROR) << "block_width shouldn't be 0.";
      return kLiteError;
    }
  } else {
    MS_LOG(ERROR) << "block_width attr doesn't exist.";
    return kLiteError;
  }

  const auto &input = (*inputs)[0];
  if (input.Shape().size() != dpico::kDims4) {
    MS_LOG(ERROR) << "inputs_0's shape should be 4 dims, but now it's " << input.Shape().size() << " dims";
    return kLiteError;
  }
  auto &output = (*outputs)[0];
  std::vector<int64_t> output_shape(dpico::kDims4);
  output_shape[0] = input.Shape().at(0);
  output_shape[dpico::kAxis1] = num_output;
  output_shape[dpico::kAxis2] = input.Shape().at(dpico::kAxis2) / block_height;
  output_shape[dpico::kAxis3] = input.Shape().at(dpico::kAxis3) / block_width;
  output.SetShape(output_shape);

  return kSuccess;
}
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, PassThrough, DpicoPassThroughInferCreater)
}  // namespace kernel
}  // namespace mindspore
