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

#include "infer/dpico_extract_infer.h"
#include <vector>
#include <memory>
#include <map>
#include <string>
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
std::shared_ptr<KernelInterface> DpicoExtractInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<DpicoExtractInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "make shared failed, infer is nullptr.";
    return nullptr;
  }
  return infer;
}
Status DpicoExtractInterface::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                                    const schema::Primitive *primitive, const kernel::Kernel *kernel) {
  auto status = dpico::CheckCustomInputOutput(inputs, outputs, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Check custom input output failed.";
    return kLiteError;
  }
  auto param = primitive->value_as_Custom();
  if (dpico::CheckCustomParam(param, "Extract") != RET_OK) {
    MS_LOG(ERROR) << "custom param is invalid.";
    return kLiteError;
  }

  // get param value
  std::map<std::string, const flatbuffers::Vector<uint8_t> *> custom_attrs;
  int raw_axis = 1;
  uint32_t slice_begin = 0;
  uint32_t slice_end = 1;
  if (param->attr() == nullptr) {
    MS_LOG(ERROR) << "param->attr() is nullptr";
    return kLiteError;
  }
  for (uint16_t i = 0; i < static_cast<uint32_t>(param->attr()->size()); i++) {
    if (param->attr()->Get(i) == nullptr || param->attr()->Get(i)->name() == nullptr) {
      MS_LOG(ERROR) << "param->attr()->Get(i) is nullptr or param->attr()->Get(i)->name() is nullptr";
      return kLiteError;
    }
    (void)custom_attrs.emplace(std::pair(param->attr()->Get(i)->name()->str(), param->attr()->Get(i)->data()));
  }
  if (custom_attrs.count(ops::kAxis) == 1) {
    if (memcpy_s(&raw_axis, sizeof(int32_t), custom_attrs[ops::kAxis]->data(), custom_attrs[ops::kAxis]->size()) !=
        EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return kLiteError;
    }
  }
  if (custom_attrs.count(dpico::kSlicePointBegin) == 1) {
    if (memcpy_s(&slice_begin, sizeof(uint32_t), custom_attrs[dpico::kSlicePointBegin]->data(),
                 custom_attrs[dpico::kSlicePointBegin]->size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return kLiteError;
    }
  }
  if (custom_attrs.count(dpico::kSlicePointEnd) == 1) {
    if (memcpy_s(&slice_end, sizeof(uint32_t), custom_attrs[dpico::kSlicePointEnd]->data(),
                 custom_attrs[dpico::kSlicePointEnd]->size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return kLiteError;
    }
  }

  const auto &input = (*inputs)[0];
  auto &output = (*outputs)[0];

  auto input_shape = input.Shape();
  if (input_shape.empty()) {
    MS_LOG(ERROR) << "input shape is empty.";
    return kLiteError;
  }
  size_t axis = (raw_axis + static_cast<int32_t>(input_shape.size())) % input_shape.size();
  if (input_shape.size() <= axis) {
    MS_LOG(ERROR) << "input_shape size: " << input_shape.size() << " is less than axis: " << axis;
    return kLiteError;
  }
  if (input_shape.at(axis) < slice_end) {
    MS_LOG(ERROR) << "slice_point_end " << slice_end << " is greater than dim " << input_shape.at(axis);
    return kLiteError;
  }
  std::vector<int64_t> output_shape(input_shape);
  output_shape[axis] = slice_end - slice_begin;
  output.SetShape(output_shape);
  return kSuccess;
}
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Extract, DpicoExtractInferCreater)
}  // namespace kernel
}  // namespace mindspore
