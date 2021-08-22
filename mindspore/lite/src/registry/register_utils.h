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

#ifndef MINDSPORE_LITE_SRC_REGISTRY_REGISTER_UTILS_H_
#define MINDSPORE_LITE_SRC_REGISTRY_REGISTER_UTILS_H_
#include <string>
#include "include/registry/register_kernel.h"
#include "schema/model_generated.h"
#include "ir/dtype/type_id.h"

namespace mindspore {
namespace kernel {
/// \brief KernelDesc defined kernel's basic attribute.
struct KernelDesc {
  TypeId data_type;     /**< kernel data type argument */
  int type;             /**< op type argument */
  std::string arch;     /**< deviceType argument */
  std::string provider; /**< user identification argument */

  bool operator<(const KernelDesc &dst) const {
    if (provider != dst.provider) {
      return provider < dst.provider;
    } else if (arch != dst.arch) {
      return arch < dst.arch;
    } else if (data_type != dst.data_type) {
      return data_type < dst.data_type;
    } else {
      return type < dst.type;
    }
  }
};

/// \brief RegisterKernel Defined registration of kernel.
class RegisterUtils {
 public:
  /// \brief Static methon to get a kernel's create function.
  ///
  /// \param[in] desc Define kernel's basic attribute.
  /// \param[in] primitive Define the attributes of op.
  ///
  /// \return Function pointer to create a kernel.
  static CreateKernel GetCreator(const schema::Primitive *primitive, kernel::KernelDesc *desc);
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_REGISTRY_REGISTER_UTILS_H_
