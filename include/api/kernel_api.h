/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_INCLUDE_API_KERNEL_API_H
#define MINDSPORE_INCLUDE_API_KERNEL_API_H
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "include/api/types.h"
#include "include/api/status.h"
namespace mindspore {
class Context;
namespace kernel {
/// \brief The Kernel class is used to define a MindSpore Kernel.
class MS_API MSKernel {
 public:
  MSKernel() = default;
  /// \brief Constructor.
  ///
  /// \param[in] inputs define the input tensors for kernel.
  /// \param[in] outputs define the output tensors for kernel.
  /// \param[in] primitive define the primitive of kernel.
  /// \param[in] ctx define the context for kernel.
  MSKernel(std::vector<mindspore::MSTensor> inputs, std::vector<mindspore::MSTensor> outputs,
           const mindspore::Context *ctx)
      : context_(ctx), inputs_(std::move(inputs)), outputs_(std::move(outputs)) {}
  /// \brief Destructor.
  virtual ~MSKernel() = default;
  /// \brief infer shape, datatype and format for output tensor of kernel.
  ///
  /// \return result code.
  virtual int InferShape() { return kLiteError; }
  /// \brief prepare for executing kernel.
  ///
  /// \return result code.
  virtual int Prepare() = 0;
  /// \brief execute the kernel.
  ///
  /// \return result code.
  virtual int Execute() = 0;
  /// \brief resize the kernel input shape, memory need to refresh.
  ///
  /// \return result code.
  virtual int ReSize() = 0;
  /// \brief set kernel's input tensors.
  ///
  /// \param[in] in_tensors define the input tensors.
  virtual void set_inputs(const std::vector<mindspore::MSTensor> &in_tensors) { this->inputs_ = in_tensors; }
  /// \brief set kernel's input tensor.
  ///
  /// \param[in] in_tensor define the input tensor.
  /// \param[in] index define the index of the input tensor.
  virtual void set_input(mindspore::MSTensor in_tensor, int index) { this->inputs_[index] = in_tensor; }
  /// \brief set kernel's output tensors.
  ///
  /// \param[in] out_tensors define the output tensors.
  virtual void set_outputs(const std::vector<mindspore::MSTensor> &out_tensors) { this->outputs_ = out_tensors; }
  /// \brief set kernel's output tensor.
  ///
  /// \param[in] out_tensor define the output tensor.
  /// \param[in] index define the index of the output tensor.
  virtual void set_output(mindspore::MSTensor out_tensor, int index) { this->outputs_[index] = out_tensor; }
  /// \brief obtain kernel's input tensors.
  ///
  /// \return input tensors.
  virtual const std::vector<mindspore::MSTensor> &inputs() { return this->inputs_; }
  /// \brief obtain kernel's output tensors.
  ///
  /// \return output tensors.
  virtual const std::vector<mindspore::MSTensor> &outputs() { return this->outputs_; }
  /// \brief obtain kernel's name.
  ///
  /// \return kernel's name.
  virtual std::string name() const { return this->name_; }
  /// \brief set kernel's name.
  ///
  /// \param[in] name define the kernel's name.
  void set_name(const std::string &name) { this->name_ = name; }
  /// \brief obtain kernel's context.
  ///
  /// \return kernel's context.
  const mindspore::Context *context() const { return this->context_; }

  /// \brief get kernel's attribute.
  ///
  /// \param[in] key define the kernel's attribute key.
  std::string GetAttr(const std::string &key) const {
    auto iter = attrs_.find(key);
    if (iter != attrs_.end()) {
      return iter->second;
    }
    return "";
  }

  /// \brief set kernel's config.
  ///
  /// \param[in] config define the kernel's config.
  void SetConfig(const std::map<std::string, std::map<std::string, std::string>> *config) { config_ = config; }
  /// \brief set kernel's config.
  ///
  /// \param[in] section define the section of the kernel's config.
  std::map<std::string, std::string> GetConfig(const std::string &section) const {
    if (config_ == nullptr) {
      return std::map<std::string, std::string>();
    }
    auto iter = config_->find(section);
    if (iter != config_->end()) {
      return iter->second;
    }
    return std::map<std::string, std::string>();
  }

 protected:
  /// \brief set kernel's attribute
  ///
  /// \param[in] key define the kernel's attribute key.
  /// \param[in] value define the kernel's attribute value.
  void SetAttr(const std::string &key, const std::string &value) { attrs_[key] = value; }

  std::string name_;
  const mindspore::Context *context_ = nullptr;
  std::vector<mindspore::MSTensor> inputs_;
  std::vector<mindspore::MSTensor> outputs_;
  std::map<std::string, std::string> attrs_;
  const std::map<std::string, std::map<std::string, std::string>> *config_ = nullptr;
};

/// \brief The Kernel class is used to define a MindSpore Kernel with specific primitive.
template <typename Primitive>
class MS_API IKernel : public MSKernel {
 public:
  IKernel() = default;
  /// \brief Constructor.
  ///
  /// \param[in] inputs define the input tensors for kernel.
  /// \param[in] outputs define the output tensors for kernel.
  /// \param[in] primitive define the primitive of kernel.
  /// \param[in] ctx define the context for kernel.
  IKernel(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs,
          const Primitive *primitive, const mindspore::Context *ctx)
      : MSKernel(inputs, outputs, ctx), primitive_(primitive) {}
  /// \brief Destructor.
  ~IKernel() override = default;
  /// \brief get the primitive of kernel.
  ///
  /// \return the primitive of kernel generated by flatbuffers.
  const Primitive *primitive() const { return this->primitive_; }

 protected:
  const Primitive *primitive_ = nullptr;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_KERNEL_API_H
