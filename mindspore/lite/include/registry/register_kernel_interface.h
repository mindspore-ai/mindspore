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

#ifndef MINDSPORE_LITE_INCLUDE_REGISTRY_KERNEL_INTERFACE_H_
#define MINDSPORE_LITE_INCLUDE_REGISTRY_KERNEL_INTERFACE_H_

#include <set>
#include <string>
#include <vector>
#include <memory>
#include "include/kernel_interface.h"
#include "schema/model_generated.h"

namespace mindspore {
namespace kernel {
class Kernel;
}
namespace registry {
/// \brief KernelInterfaceCreator defined a functor to create KernelInterface.
using KernelInterfaceCreator = std::function<std::shared_ptr<kernel::KernelInterface>()>;

/// \brief RegisterKernelInterface defined registration and acquisition of KernelInterface.
class MS_API RegisterKernelInterface {
 public:
  /// \brief Static method to register op whose primitive type is custom.
  ///
  /// \param[in] provider Define the identification of user.
  /// \param[in] op_type Define the concrete type of a custom op.
  /// \param[in] creator Define the KernelInterface create function.
  ///
  /// \return Status as a status identification of registering.
  static Status CustomReg(const std::string &provider, const std::string &op_type,
                          const KernelInterfaceCreator creator);

  /// \brief Static method to register op whose primitive type is ordinary.
  ///
  /// \param[in] provider Define the identification of user.
  /// \param[in] op_type Define the ordinary op type.
  /// \param[in] creator Define the KernelInterface create function.
  ///
  /// \return Status as a status identification of registering.
  static Status Reg(const std::string &provider, int op_type, const KernelInterfaceCreator creator);

  /// \brief Static method to get registration of a certain op.
  ///
  /// \param[in] provider Define the identification of user.
  /// \param[in] primitive Define the attributes of a certain op.
  /// \param[in] kernel Define the kernel of a certain op.
  ///
  /// \return Boolean value to represent registration of a certain op is existing or not.
  static std::shared_ptr<kernel::KernelInterface> GetKernelInterface(const std::string &provider,
                                                                     const schema::Primitive *primitive,
                                                                     const kernel::Kernel *kernel = nullptr);
};

/// \brief KernelInterfaceReg defined registration class of KernelInterface.
class MS_API KernelInterfaceReg {
 public:
  /// \brief Constructor of KernelInterfaceReg to register an ordinary op.
  ///
  /// \param[in] provider Define the identification of user.
  /// \param[in] op_type Define the ordinary op type.
  /// \param[in] creator Define the KernelInterface create function.
  KernelInterfaceReg(const std::string &provider, int op_type, const KernelInterfaceCreator creator) {
    RegisterKernelInterface::Reg(provider, op_type, creator);
  }

  /// \brief Constructor of KernelInterfaceReg to register custom op.
  ///
  /// \param[in] provider Define the identification of user.
  /// \param[in] op_type Define the concrete type of a custom op.
  /// \param[in] creator Define the KernelInterface create function.
  KernelInterfaceReg(const std::string &provider, const std::string &op_type, const KernelInterfaceCreator creator) {
    RegisterKernelInterface::CustomReg(provider, op_type, creator);
  }

  virtual ~KernelInterfaceReg() = default;
};

/// \brief Defined registering macro to register ordinary op, which called by user directly.
///
/// \param[in] provider Define the identification of user.
/// \param[in] op_type Define the ordinary op type.
/// \param[in] creator Define the KernelInterface create function.
#define REGISTER_KERNEL_INTERFACE(provider, op_type, creator)                                                    \
  namespace {                                                                                                    \
  static mindspore::registry::KernelInterfaceReg g_##provider##op_type##_inter_reg(#provider, op_type, creator); \
  }  // namespace

/// \brief Defined registering macro to register custom op, which called by user directly.
///
/// \param[in] provider Define the identification of user.
/// \param[in] op_type Define the concrete type of a custom op.
/// \param[in] creator Define the KernelInterface create function.
#define REGISTER_CUSTOM_KERNEL_INTERFACE(provider, op_type, creator)                                           \
  namespace {                                                                                                  \
  static mindspore::registry::KernelInterfaceReg g_##provider##op_type##_custom_inter_reg(#provider, #op_type, \
                                                                                          creator);            \
  }  // namespace
}  // namespace registry
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_REGISTRY_KERNEL_INTERFACE_H_
