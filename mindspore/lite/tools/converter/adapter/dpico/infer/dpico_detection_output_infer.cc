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

#include "infer/dpico_detection_output_infer.h"
#include <iostream>
#include <vector>
#include <memory>
#include "utils/log_adapter.h"
#include "common/infer_util.h"
#include "include/errorcode.h"
#include "include/registry/register_kernel_interface.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
namespace kernel {
namespace {
constexpr int kDimensionOfBbox = 7;  // [image_id, label, confidence, xmin, ymin, xmax, ymax]
}  // namespace
std::shared_ptr<KernelInterface> DpicoDetectionOutputInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<DpicoDetectionOutputInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "make shared failed, infer is nullptr.";
    return nullptr;
  }
  return infer;
}
Status DpicoDetectionOutputInterface::Infer(std::vector<mindspore::MSTensor> *inputs,
                                            std::vector<mindspore::MSTensor> *outputs,
                                            const schema::Primitive *primitive, const kernel::Kernel *kernel) {
  auto status = dpico::CheckCustomInputOutput(inputs, outputs, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Check custom input output failed.";
    return kLiteError;
  }
  auto param = primitive->value_as_Custom();
  if (dpico::CheckCustomParam(param, "DetectionOutput") != RET_OK) {
    MS_LOG(ERROR) << "custom param is invalid.";
    return kLiteError;
  }

  const auto &input = (*inputs)[0];
  auto &output = (*outputs)[0];
  output.SetShape(std::vector<int64_t>({input.Shape()[0], kDimensionOfBbox}));
  return kSuccess;
}
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, DetectionOutput, DpicoDetectionOutputInferCreater)
}  // namespace kernel
}  // namespace mindspore
