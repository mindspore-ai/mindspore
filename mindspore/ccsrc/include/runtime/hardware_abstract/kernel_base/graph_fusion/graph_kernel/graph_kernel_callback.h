/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_GRAPH_KERNEL_CALLBACK_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_GRAPH_KERNEL_CALLBACK_H_
#include <string>
#include <memory>
#include <vector>
#include <functional>

#include "ir/anf.h"
#include "ir/dtype/type_id.h"
#include "utils/shape_utils.h"
#include "runtime/hardware_abstract/visible.h"

namespace mindspore::graphkernel {
using DFormat = std::string;
using DShape = ShapeVector;

namespace inner {
struct RUNTIME_HARDWARE_EXPORT NodeBase {
  DShape shape;
  TypeId type;
  DFormat format;
  ListSymbolPtr symbolic_shape{nullptr};
};
}  // namespace inner

class Callback;
using CallbackPtr = std::shared_ptr<Callback>;
class RUNTIME_HARDWARE_EXPORT Callback {
 public:
  virtual ~Callback() = default;
  static CallbackPtr Instance() { return instance_; }

  /**
   * @brief Get the real input shape of the `node`.
   *
   * @param node the AnfNodePtr
   * @param i    the input index, start from 0
   */
  virtual ShapeVector GetInputShape(const AnfNodePtr &node, size_t i) = 0;

  /**
   * @brief Get the real output shape of the `node`.
   *
   * @param node the AnfNodePtr
   * @param i    the output index, start from 0
   */
  virtual ShapeVector GetOutputShape(const AnfNodePtr &node, size_t i) = 0;

  /**
   * @brief Get the inferred input shape of the `node`.
   *
   * @param node the AnfNodePtr
   * @param i    the input index, start from 0
   */
  virtual ShapeVector GetInputInferShape(const AnfNodePtr &node, size_t i) = 0;

  /**
   * @brief Get the inferred output shape of the `node`.
   *
   * @param node the AnfNodePtr
   * @param i    the output index, start from 0
   */
  virtual ShapeVector GetOutputInferShape(const AnfNodePtr &node, size_t i) = 0;

  /**
   * @brief Get the real input data type of the `node`.
   *
   * @param node the AnfNodePtr
   * @param i    the input index, start from 0
   */
  virtual TypeId GetInputType(const AnfNodePtr &node, size_t i) = 0;

  /**
   * @brief Get the real output data type of the `node`.
   *
   * @param node the AnfNodePtr
   * @param i    the output index, start from 0
   */
  virtual TypeId GetOutputType(const AnfNodePtr &node, size_t i) = 0;

  /**
   * @brief Get the inferred input data type of the `node`.
   *
   * @param node the AnfNodePtr
   * @param i    the input index, start from 0
   */
  virtual TypeId GetInputInferType(const AnfNodePtr &node, size_t i) = 0;

  /**
   * @brief Get the inferred output data type of the `node`.
   *
   * @param node the AnfNodePtr
   * @param i    the output index, start from 0
   */
  virtual TypeId GetOutputInferType(const AnfNodePtr &node, size_t i) = 0;

  /**
   * @brief Get the input data format of the `node`.
   *
   * @param node the AnfNodePtr
   * @param i    the input index, start from 0
   */
  virtual std::string GetInputFormat(const AnfNodePtr &node, size_t i) = 0;

  /**
   * @brief Get the output data format of the `node`.
   *
   * @param node the AnfNodePtr
   * @param i    the output index, start from 0
   */
  virtual std::string GetOutputFormat(const AnfNodePtr &node, size_t i) = 0;

  /**
   * @brief Get the processor of the `node`.
   *
   * @param node the AnfNodePtr
   */
  virtual std::string GetProcessor(const AnfNodePtr &node) = 0;

  /**
   * @brief Get the backend target from context.
   *
   * @param detail if false(default), only "Ascend/GPU/CPU" is returned. otherwise target like "Ascend910" is returned.
   */
  std::string GetTargetFromContext(bool detail = false) { return GetTargetFromContextImpl(detail); }

  /**
   * @brief Set KernelInfo for a GraphKernel node, the info is extract from its inputs/outputs.
   *
   * @param[in] node the GraphKernel CNode.
   */
  virtual void SetGraphKernelNodeKernelInfo(const AnfNodePtr &node) = 0;

  /**
   * @brief Set KernelInfo for a basic node.
   *
   * @param node         the AnfNodePtr
   * @param outputs_info the output info list
   */
  virtual void SetBasicNodeKernelInfo(const AnfNodePtr &node, const std::vector<inner::NodeBase> &outputs_info) = 0;

  /**
   * @brief Set empty KernelInfo.
   *
   * @param node the AnfNodePtr
   */
  virtual void SetEmptyKernelInfo(const AnfNodePtr &node) = 0;

  /**
   * @brief Reset KernelInfo on different platforms.
   *
   * @param node the AnfNodePtr
   */
  virtual void ResetKernelInfo(const AnfNodePtr &node) = 0;

  /**
   * @brief Reset KernelInfo input msg for convert attr and input.
   *
   * @param node the AnfNodePtr
   * @param overwrite if true, override all inputs kernel info, if false, use the original kernel info saved in node
   */
  virtual void ResetKernelInfoInputs(const AnfNodePtr &node, const std::vector<size_t> &indices) = 0;

  /**
   * @brief The Callback implementation use nodes' device info.
   */
  virtual bool IsUseDeviceInfo() { return true; }

  static void RegImpl(const CallbackPtr &cb) { instance_ = cb; }

 private:
  // to avoid the default argument in virtual function.
  virtual std::string GetTargetFromContextImpl(bool detail) = 0;

  friend class CallbackImplRegister;
#ifndef _MSC_VER
  RUNTIME_HARDWARE_EXPORT inline static CallbackPtr instance_{nullptr};
#else
  inline static CallbackPtr instance_{nullptr};
#endif
};

class CallbackImplRegister {
 public:
  explicit CallbackImplRegister(const std::function<CallbackPtr()> &fn) noexcept { Callback::RegImpl(fn()); }
  ~CallbackImplRegister() = default;

 protected:
  // for pclint-plus
  bool rev_{false};
};

#define GRAPH_KERNEL_CALLBACK_REGISTER(cls) \
  const CallbackImplRegister callback(      \
    []() noexcept { return std::static_pointer_cast<Callback>(std::make_shared<cls>()); })
}  // namespace mindspore::graphkernel
#endif
