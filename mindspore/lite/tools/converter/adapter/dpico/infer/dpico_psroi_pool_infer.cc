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

#include "infer/dpico_psroi_pool_infer.h"
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
std::shared_ptr<KernelInterface> DpicoPsRoiPoolInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<DpicoPsRoiPoolInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "make shared failed, infer is nullptr.";
    return nullptr;
  }
  return infer;
}
Status DpicoPsRoiPoolInterface::Infer(std::vector<mindspore::MSTensor> *inputs,
                                      std::vector<mindspore::MSTensor> *outputs, const schema::Primitive *primitive,
                                      const kernel::Kernel *kernel) {
  auto status = dpico::CheckCustomInputOutput(inputs, outputs, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Check custom input output failed.";
    return kLiteError;
  }
  auto param = primitive->value_as_Custom();
  if (dpico::CheckCustomParam(param, "PsRoiPool") != RET_OK) {
    MS_LOG(ERROR) << "custom param is invalid.";
    return kLiteError;
  }

  // get param value
  std::map<std::string, const flatbuffers::Vector<uint8_t> *> custom_attrs;
  int32_t output_dim = 0;
  int32_t group_size = 0;
  if (param->attr() == nullptr) {
    MS_LOG(ERROR) << "param->attr() is nullptr";
    return kLiteError;
  }
  for (size_t i = 0; i < param->attr()->size(); i++) {
    if (param->attr()->Get(i) == nullptr || param->attr()->Get(i)->name() == nullptr) {
      MS_LOG(ERROR) << "param->attr()->Get(i) is nullptr or param->attr()->Get(i)->name() is nullptr";
      return kLiteError;
    }
    (void)custom_attrs.emplace(std::pair(param->attr()->Get(i)->name()->str(), param->attr()->Get(i)->data()));
  }
  if (custom_attrs.count(dpico::kGroupSize) == 1) {
    if (memcpy_s(&group_size, sizeof(int32_t), custom_attrs[dpico::kGroupSize]->data(),
                 custom_attrs[dpico::kGroupSize]->size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return kLiteError;
    }
  } else {
    MS_LOG(ERROR) << "group_size attr doesn't exist.";
    return kLiteError;
  }
  if (custom_attrs.count(dpico::kOutputDim) == 1) {
    if (memcpy_s(&output_dim, sizeof(int32_t), custom_attrs[dpico::kOutputDim]->data(),
                 custom_attrs[dpico::kOutputDim]->size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return kLiteError;
    }
  } else {
    MS_LOG(ERROR) << "output_dim attr doesn't exist.";
    return kLiteError;
  }

  int pooled_h = group_size;
  int pooled_w = group_size;

  if (inputs->size() != dpico::kDims2) {
    MS_LOG(ERROR) << "psroi input size is invalid, which is " << inputs->size();
    return kLiteError;
  }
  const auto &psroi = (*inputs)[1];
  if (psroi.Shape().empty()) {
    MS_LOG(ERROR) << "inputs_1 shape is empty.";
    return kLiteError;
  }
  auto &output = (*outputs)[0];
  std::vector<int64_t> output_shape(dpico::kDims4);
  output_shape[0] = psroi.Shape().at(0);
  output_shape[dpico::kAxis1] = output_dim;
  output_shape[dpico::kAxis2] = pooled_h;
  output_shape[dpico::kAxis3] = pooled_w;
  output.SetShape(output_shape);

  return kSuccess;
}
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, PsRoiPool, DpicoPsRoiPoolInferCreater)
}  // namespace kernel
}  // namespace mindspore
