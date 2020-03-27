/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PREDICT_H_
#define MINDSPORE_MINDSPORE_CCSRC_PREDICT_H_

#include <memory>
#include <vector>
#include <string>
#include "session/session_basic.h"
#include "predict/converter/kernel2ms.h"

namespace mindspore {
namespace predictmodel {
using KernelGraphPtrNew = std::shared_ptr<mindspore::session::KernelGraph>;
void StepConvertGraph(const KernelGraphPtrNew &kernel_graph_ptr);
void StepConvertWeight(const std::vector<tensor::TensorPtr> &inputs);
executor::TargetMode GetDeviceTarget(const std::string &device_target);
}  // namespace predictmodel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PREDICT_H_
