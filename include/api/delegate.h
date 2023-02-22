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
#include <string>
#include <utility>
#include "schema/model_generated.h"
#include "include/api/kernel.h"
#include "include/api/delegate_api.h"

namespace mindspore {
typedef enum {
  SCHEMA_INVALID = -1, /**< invalid version */
  SCHEMA_CUR,          /**< current version for ms model defined in model.fbs*/
  SCHEMA_V0,           /**< previous version for ms model defined in model_v0.fbs*/
} SchemaVersion;

using KernelIter = std::vector<kernel::Kernel *>::iterator;

template <class T>
class MS_API DelegateModel {
 public:
  DelegateModel() = default;
  /// \brief Constructor of MindSpore Lite DelegateModel.
  DelegateModel(std::vector<kernel::Kernel *> *kernels, const std::vector<MSTensor> &inputs,
                const std::vector<MSTensor> &outputs, const std::map<kernel::Kernel *, const T *> &primitives,
                SchemaVersion version)
      : kernels_(kernels), inputs_(inputs), outputs_(outputs), primitives_(primitives), version_(version) {}

  /// \brief Destructor of MindSpore Lite DelegateModel.
  ~DelegateModel() = default;

  /// \brief Get Primitive of kernel::Kernel.
  ///
  /// \param[in] kernel kernel in DelegateModel kernels vector.
  ///
  /// \return The Primitive of The kernel.
  const T *GetPrimitive(kernel::Kernel *kernel) const {
    if (primitives_.find(kernel) != primitives_.end()) {
      return primitives_.at(kernel);
    } else {
      return nullptr;
    }
  }

  /// \brief Get the begin iterator of the DelegateModel kernels vector.
  ///
  /// \return The begin iterator of the DelegateModel kernels vector.
  KernelIter BeginKernelIterator() { return kernels_->begin(); }

  /// \brief Get the end iterator of the DelegateModel kernels vector.
  ///
  /// \return The end iterator of the DelegateModel kernels vector.
  KernelIter EndKernelIterator() { return kernels_->end(); }

  /// \brief Replace the continuous kernel supported by the delegate with a delegate graph kernel.
  ///
  /// \param[in] from Define the begin iterator of continuous kernel supported by the delegate.
  /// \param[in] end Define the end iterator of continuous kernel supported by the delegate.
  ///
  /// \return The next iterator after graph_kernel, point to the next kernel that is not visited.
  KernelIter Replace(KernelIter from, KernelIter end, kernel::Kernel *graph_kernel) {
    size_t insert_index = from - BeginKernelIterator();
    if (insert_index >= kernels_->size()) {
      return BeginKernelIterator();
    }
    kernels_->erase(from, end);
    kernels_->insert(BeginKernelIterator() + insert_index, graph_kernel);
    return BeginKernelIterator() + insert_index + 1;
  }

  /// \brief Get the nodes of DelegateModel.
  ///
  /// \return The pointer to nodes vector of DelegateModel.
  std::vector<kernel::Kernel *> *nodes() { return kernels_; }

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
  SchemaVersion GetVersion() const { return version_; }

 protected:
  std::vector<kernel::Kernel *> *kernels_;
  const std::vector<mindspore::MSTensor> &inputs_;
  const std::vector<mindspore::MSTensor> &outputs_;
  const std::map<kernel::Kernel *, const T *> &primitives_;
  SchemaVersion version_;
};

// lite delegate use kernel::Kernel as graph node.
using LiteDelegateGraph = DelegateModel<schema::Primitive>;
class Delegate : public IDelegate<LiteDelegateGraph, kernel::Kernel, kernel::Kernel> {
 public:
  Delegate() = default;
  Delegate(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs)
      : IDelegate<LiteDelegateGraph, kernel::Kernel, kernel::Kernel>(inputs, outputs) {}
  virtual ~Delegate() = default;
  /// \brief Init delegate.
  ///
  /// \note Init will be called in Model::Build.
  ///
  /// \return Status. If Status is kLiteNotSupport, the program will return to the MindSpore Lite inner inference.
  virtual Status Init() = 0;

  std::shared_ptr<kernel::Kernel> CreateKernel(const std::shared_ptr<kernel::Kernel> &node) override {
    // return node as kernel since they are same one.
    return node;
  }

  bool IsDelegateNode(const std::shared_ptr<kernel::Kernel> &node) override { return false; }

  /// \brief Replace the nodes in model with delegate nodes, delegate will create kernels by its delegate nodes.
  ///
  /// \param[in] graph The graph to be built.
  void ReplaceNodes(const std::shared_ptr<LiteDelegateGraph> &graph) override {}

  /// \brief Build delegate graph for MindSpore model.
  ///
  /// \note Build will be called in Model::Build.
  ///
  /// \param[in] model Define the delegate model to be built.
  ///
  /// \note deprecated, use ReplaceNodes and CreateKernel to build delegate model.
  virtual Status Build(LiteDelegateGraph *model) = 0;
};

class MS_API CoreMLDelegate : public Delegate {
 public:
  /// \brief Constructor of MindSpore Lite CoreML Delegate.
  CoreMLDelegate();

  /// \brief Init CoreML delegate.
  ///
  /// \note Init will be called in Model::Build.
  ///
  /// \return Status. If Status is kLiteNotSupport, the program will return to the MindSpore Lite inner inference.
  Status Init() override;

  /// \brief Build CoreML delegate graph for MindSpore Lite model.
  ///
  /// \note Build will be called in Model::Build.
  ///
  /// \param[in] model Define the delegate model to be built.
  Status Build(LiteDelegateGraph *model) override;

 protected:
  std::shared_ptr<Delegate> impl_;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_DELEGATE_H
