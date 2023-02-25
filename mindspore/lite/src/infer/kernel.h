/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_INFER_KERNEL_H_
#define MINDSPORE_LITE_INFER_KERNEL_H_

#include <string>
#include <memory>

#include "infer/tensor.h"
#include "litert/kernel_exec.h"

namespace mindspore::infer::abstract {
using Kernel = mindspore::kernel::KernelExec;
// class Kernel : public std::enable_shared_from_this<Kernel> {
//  public:
//   virtual ~Kernel() = default;

//   /// \brief Execute Kernel with inner inputs and outputs.
//   ///
//   /// \return int.
//   virtual int Execute() = 0;

//   /// \brief Prepare Kernel Execution.
//   ///
//   /// \return int.
//   virtual int Prepare() = 0;

//   /// \brief Resize Kernel Resource.
//   ///
//   /// \return int.
//   virtual int ReSize() = 0;

//   /// \brief Get Kernel name.
//   ///
//   /// \return name of kernel.
//   virtual std::string name() const = 0;

//   /// \brief Set Kernel name.
//   ///
//   /// \return void.
//   virtual void set_name(const std::string &name) = 0;

//   /// \brief Train Kernel.
//   ///
//   /// \return result of train.
//   virtual int Train() = 0;

//   /// \brief Is Kernel Train.
//   ///
//   /// \return is kernel trained.
//   virtual bool IsTrain() const = 0;

//   /// \brief Eval Kernel.
//   ///
//   /// \return int.
//   virtual int Eval() = 0;

//   /// \brief If the Kernel is Eval.
//   ///
//   /// \return bool.
//   virtual bool IsEval() const = 0;

//   /// \brief Set Kernel can be train.
//   ///
//   /// \param trainable is kernel can train
//   ///
//   /// \return void.
//   virtual void SetTrainable(bool trainable = true) = 0;

//   /// \brief Is Kernel can be train.
//   ///
//   /// \return bool.
//   virtual bool IsTrainable() const = 0;

//   /// \brief Set if kernel output is model output.
//   ///
//   /// \param is_model_output kernel output is model output
//   ///
//   /// \return void.
//   virtual void set_is_model_output(bool is_model_output) = 0;

//   /// \brief If kernel output is model output.
//   ///
//   /// \return bool.
//   virtual bool is_model_output() const = 0;

//   /// \brief If kernel finish infer shape.
//   ///
//   /// \return bool.
//   virtual bool InferShapeDone() const = 0;

//   /// \brief kernel op tyep.
//   ///
//   /// \return string of op type.
//   virtual std::string type_str() = 0;

//   /// \brief Set Input Tensors For Kernel.
//   ///
//   ///\param[in] in_tensors Abstract Input Tensor list for Kernel.
//   ///
//   /// \return void.
//   virtual void set_in_tensors(const std::vector<Tensor *> &in_tensors) = 0;

//   /// \brief Set Input Tensor For Kernel.
//   ///
//   ///\param[in] in_tensor Abstract Input Tensor for Kernel.
//   ///\param[in] index     Tensor Index for Kernel.
//   ///
//   /// \return void.
//   virtual void set_in_tensor(Tensor *in_tensor, size_t index) = 0;

//   /// \brief Set Output Tensors For Kernel.
//   ///
//   ///\param[in] out_tensors Abstract Output Tensor list for Kernel.
//   ///
//   /// \return void.
//   virtual void set_out_tensors(const std::vector<Tensor *> &out_tensors) = 0;

//   /// \brief Set Output Tensor For Kernel.
//   ///
//   ///\param[in] out_tensor Abstract Output Tensor for Kernel.
//   ///\param[in] index     Tensor Index for Kernel.
//   ///
//   /// \return void.
//   virtual void set_out_tensor(Tensor *out_tensor, size_t index) = 0;

//   /// \brief Get Input Tensor List Of Kernel.
//   ///
//   /// \return Tensor List.
//   virtual const std::vector<Tensor *> &in_tensors() const = 0;

//   /// \brief Get Output Tensor List Of Kernel.
//   ///
//   /// \return Tensor List.
//   virtual const std::vector<Tensor *> &out_tensors() const = 0;
// };
}  // namespace mindspore::infer::abstract

#endif  // MINDSPORE_LITE_INFER_KERNEL_H_
