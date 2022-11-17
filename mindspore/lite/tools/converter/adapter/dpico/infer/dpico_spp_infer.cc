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

#include "infer/dpico_spp_infer.h"
#include <vector>
#include <memory>
#include <string>
#include <cmath>
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
namespace {
constexpr auto kSquareNum = 2;
constexpr auto kBaseNum = 2;
}  // namespace
std::shared_ptr<KernelInterface> DpicoSppInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<DpicoSppInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "make shared failed, infer is nullptr.";
    return nullptr;
  }
  return infer;
}
Status DpicoSppInterface::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                                const schema::Primitive *primitive, const kernel::Kernel *kernel) {
  auto status = dpico::CheckCustomInputOutput(inputs, outputs, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Check custom input output failed.";
    return kLiteError;
  }
  auto param = primitive->value_as_Custom();
  if (dpico::CheckCustomParam(param, "Spp") != RET_OK) {
    MS_LOG(ERROR) << "custom param is invalid.";
    return kLiteError;
  }

  if (param->attr() == nullptr) {
    MS_LOG(ERROR) << "param attr is nullptr.";
    return kLiteError;
  }
  bool has_pyramid_height = false;
  uint32_t pyramid_height = 0;
  for (uint32_t i = 0; i < static_cast<uint32_t>(param->attr()->size()); i++) {
    if (param->attr()->Get(i) == nullptr || param->attr()->Get(i)->name() == nullptr) {
      MS_LOG(ERROR) << "param->attr()->Get(i) is nullptr or param->attr()->Get(i)->name() is nullptr";
      return kLiteError;
    }
    if (param->attr()->Get(i)->name()->str() == dpico::kPyramidHeight) {
      auto output_info = param->attr()->Get(i)->data();
      if (output_info == nullptr) {
        MS_LOG(ERROR) << "output_shape is nullptr";
        return kLiteError;
      }
      if (memcpy_s(&pyramid_height, sizeof(uint32_t), output_info->data(), output_info->size()) != EOK) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return kLiteError;
      }
      has_pyramid_height = true;
      break;
    }
  }
  if (!has_pyramid_height) {
    MS_LOG(ERROR) << dpico::kPyramidHeight << " attr is needed for spp";
    return kLiteError;
  }

  const auto &input = (*inputs)[0];
  auto &output = (*outputs)[0];

  auto input_shape = input.Shape();
  if (input_shape.size() != dpico::kDims4) {
    MS_LOG(ERROR) << "input_shape should have 4 dims, but in fact it's " << input_shape.size();
    return kLiteError;
  }
  std::vector<int64_t> output_shape;
  output_shape.push_back(input_shape.at(0));
  int64_t output_planes_size = 0;
  for (int64_t i = 0; i < pyramid_height; i++) {  // spp output plane size is 1 * 1, 2 * 2, 4 * 4, ...
    output_planes_size += std::pow(kBaseNum, kSquareNum * i);
  }
  output_shape.push_back(input_shape.at(1) * output_planes_size);
  output.SetShape(output_shape);
  return kSuccess;
}
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Spp, DpicoSppInferCreater)
}  // namespace kernel
}  // namespace mindspore
