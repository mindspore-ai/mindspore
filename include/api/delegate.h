/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_INCLUDE_API_DELEGATE_H
#define MINDSPORE_INCLUDE_API_DELEGATE_H

#include <map>
#include <vector>
#include <memory>
#include "schema/model_generated.h"
#include "include/api/kernel.h"
#include "include/api/status.h"

namespace mindspore {
typedef enum {
  SCHEMA_INVALID = -1, /**< invalid version */
  SCHEMA_CUR,          /**< current version for ms model defined in model.fbs*/
  SCHEMA_V0,           /**< previous version for ms model defined in model_v0.fbs*/
} SchemaVersion;

using KernelIter = std::vector<kernel::Kernel *>::iterator;
class MS_API DelegateModel {
 public:
  /// \brief Constructor of MindSpore Lite DelegateModel.
  DelegateModel(std::vector<kernel::Kernel *> *kernels, const std::vector<MSTensor> &inputs,
                const std::vector<MSTensor> &outputs,
                const std::map<kernel::Kernel *, const schema::Primitive *> &primitives, SchemaVersion version)
      : kernels_(kernels), inputs_(inputs), outputs_(outputs), primitives_(primitives), version_(version) {}

  /// \brief Destructor of MindSpore Lite DelegateModel.
  ~DelegateModel() = default;

  /// \brief Get Primitive of kernel::Kernel.
  ///
  /// \param[in] a kernel in DelegateModel kernels vector.
  ///
  /// \return The schema::Primitive of The kernel.
  const schema::Primitive *GetPrimitive(kernel::Kernel *kernel) const;

  /// \brief Get the begin iterator of the DelegateModel kernels vector.
  ///
  /// \return The begin iterator of the DelegateModel kernels vector.
  KernelIter BeginKernelIterator();

  /// \brief Get the end iterator of the DelegateModel kernels vector.
  ///
  /// \return The end iterator of the DelegateModel kernels vector.
  KernelIter EndKernelIterator();

  /// \brief Replace the continuous kernel supported by the delegate with a delegate graph kernel.
  ///
  /// \param[in] from Define the begin iterator of continuous kernel supported by the delegate.
  /// \param[in] end Define the end iterator of continuous kernel supported by the delegate.
  ///
  /// \return The next iterator after graph_kernel, point to the next kernel that is not visited.
  KernelIter Replace(KernelIter from, KernelIter end, kernel::Kernel *graph_kernel);

  /// \brief Get the input tensors of DelegateModel.
  ///
  /// \return The input tensor vector of DelegateModel.
  const std::vector<mindspore::MSTensor> &inputs() { return this->inputs_; }

  /// \brief Get the output tensors of DelegateModel.
  ///
  /// \return The ioutput tensor vector of DelegateModel.
  const std::vector<mindspore::MSTensor> &outputs() { return this->outputs_; }

  /// \brief Get the ms model version.
  ///
  /// \return The schema version for the primitives map.
  const SchemaVersion GetVersion() { return version_; }

 protected:
  std::vector<kernel::Kernel *> *kernels_;
  const std::vector<mindspore::MSTensor> &inputs_;
  const std::vector<mindspore::MSTensor> &outputs_;
  const std::map<kernel::Kernel *, const schema::Primitive *> &primitives_;
  SchemaVersion version_;
};

class MS_API Delegate {
 public:
  /// \brief Constructor of MindSpore Lite Delegate.
  Delegate() = default;

  /// \brief Destructor of MindSpore Lite Delegate.
  virtual ~Delegate() = default;

  /// \brief Init delegate.
  ///
  /// \note Init willed be called in Model::Build.
  ///
  /// \return Status. If Status is kLiteNotSupport, the program will return to the MindSpore Lite inner inference.
  virtual Status Init() = 0;

  /// \brief Build delegate graph for MindSpore Lite model.
  ///
  /// \note Build willed be called in Model::Build.
  ///
  /// \param[in] model Define the delegate model to be built.
  virtual Status Build(DelegateModel *model) = 0;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_DELEGATE_H
