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
#ifndef MINDSPORE_LITE_INCLUDE_KERNEL_INTERFACE_H_
#define MINDSPORE_LITE_INCLUDE_KERNEL_INTERFACE_H_

#include <memory>
#include <vector>
#include "include/api/types.h"
#include "include/api/status.h"
#include "include/lite_utils.h"
#include "schema/model_generated.h"

namespace mindspore {
namespace kernel {
/// \brief KernelInterface defined customized op's interface, such as infershape, and so on.
class MS_API KernelInterface {
 public:
  /// \brief Destructor of KernelInterface.
  virtual ~KernelInterface() = default;

  /// \brief Method to infer customized op's output shape.
  ///
  /// \param[in] inputs Define the input tensors of op.
  /// \param[in] outputs Define the output tensors of op.
  /// \param[in] primitive Define the attributes of op.
  ///
  /// \return  Status as a status identification of inferring.
  virtual Status Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                       const schema::Primitive *primitive) {
    return kSuccess;
  }
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_KERNEL_INTERFACE_H_
