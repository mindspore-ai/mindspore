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

#include "infer/dpico_lstm_onnx_infer.h"
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
std::shared_ptr<KernelInterface> DpicoLSTMOnnxInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<DpicoLSTMOnnxInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "make shared failed, infer is nullptr.";
    return nullptr;
  }
  return infer;
}
Status DpicoLSTMOnnxInterface::Infer(std::vector<mindspore::MSTensor> *inputs,
                                     std::vector<mindspore::MSTensor> *outputs, const schema::Primitive *primitive,
                                     const kernel::Kernel *) {
  auto status = dpico::CheckCustomInputOutput(inputs, outputs, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Check custom input output failed.";
    return kLiteError;
  }
  auto param = primitive->value_as_Custom();
  if (dpico::CheckCustomParam(param, "LSTM") != RET_OK) {
    MS_LOG(ERROR) << "custom param is invalid.";
    return kLiteError;
  }

  // get param value
  std::map<std::string, const flatbuffers::Vector<uint8_t> *> custom_attrs;
  int32_t hidden_size = 0;
  int32_t direction = 0;
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
  if (custom_attrs.count("hidden_size") == 1) {
    if (memcpy_s(&hidden_size, sizeof(int32_t), custom_attrs["hidden_size"]->data(),
                 custom_attrs["hidden_size"]->size()) != EOK) {
      MS_LOG(ERROR) << "hidden_size memcpy_s failed.";
      return kLiteError;
    }
  }
  if (custom_attrs.count("direction") == 1) {
    if (memcpy_s(&direction, sizeof(int32_t), custom_attrs["direction"]->data(), custom_attrs["direction"]->size()) !=
        EOK) {
      MS_LOG(ERROR) << "direction memcpy_s failed.";
      return kLiteError;
    }
  }

  const auto &input = (*inputs)[0];
  auto &output = (*outputs)[0];
  std::vector<int64_t> output_shape(dpico::kDims4);
  output_shape[0] = input.Shape().at(0);
  output_shape[dpico::kAxis1] = direction;
  output_shape[dpico::kAxis2] = input.Shape().at(1);
  output_shape[dpico::kAxis3] = hidden_size;
  output.SetShape(output_shape);

  return kSuccess;
}
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, LSTM, DpicoLSTMOnnxInferCreater)
}  // namespace kernel
}  // namespace mindspore
