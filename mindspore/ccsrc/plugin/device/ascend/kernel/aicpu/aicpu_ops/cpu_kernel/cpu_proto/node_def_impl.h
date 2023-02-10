/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef AICPU_CONTEXT_CPU_PROTO_NODE_DEF_IMPL_H
#define AICPU_CONTEXT_CPU_PROTO_NODE_DEF_IMPL_H
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "cpu_kernel/inc/cpu_attr_value.h"
#include "cpu_kernel/inc/cpu_tensor.h"
#include "proto/cpu_node_def.pb.h"

namespace aicpu {
class NodeDefImpl {
  friend class CpuKernelUtils;

 public:
  NodeDefImpl(
    aicpuops::NodeDef *nodedef, std::function<void(aicpuops::NodeDef *)> del_func = [](aicpuops::NodeDef *p) {})
      : nodedef_(nodedef, del_func) {}

  ~NodeDefImpl() = default;
  NodeDefImpl(const NodeDefImpl &) = delete;
  NodeDefImpl(NodeDefImpl &&) = delete;
  NodeDefImpl &operator=(const NodeDefImpl &) = delete;
  NodeDefImpl &operator=(NodeDefImpl &&) = delete;

  /*
   * parse parameter from string.
   * @return bool: true->success, false->failed
   */
  bool ParseFromString(const std::string &str);

  /*
   * serialize string to node def.
   * @return bool: true->success, false->failed
   */
  bool SerializeToString(std::string &str) const;

  /*
   * set op type to node def.
   * @param op: op type
   */
  void SetOpType(const std::string &op);

  /*
   * get op type of node def.
   * @return string: op type
   */
  std::string GetOpType() const;

  /*
   * add input tensor to node def.
   * @return shared_ptr<Tensor>: not null->success, null->failed
   */
  std::shared_ptr<Tensor> AddInputs();

  /*
   * add output tensor to node def.
   * @return shared_ptr<Tensor>: not null->success, null->failed
   */
  std::shared_ptr<Tensor> AddOutputs();

  /*
   * add attr to node def.
   * @param name: attr name
   * @param attr: attr need to add
   * @return bool: true->success, false->failed
   */
  bool AddAttrs(const std::string &name, const AttrValue *attr);

  /*
   * get input tensor size of node def.
   * @return int32_t: input tensor size of node def
   */
  int32_t InputsSize() const;

  /*
   * get output tensor size of node def.
   * @return int32_t: input tensor size of node def
   */
  int32_t OutputsSize() const;

  /*
   * get input tensor of node def.
   * @param index: index of input tensor
   * @return shared_ptr<Tensor>: input tensor ptr of node def
   */
  std::shared_ptr<Tensor> MutableInputs(int32_t index) const;

  /*
   * get output tensor of node def.
   * @param index: index of output tensor
   * @return shared_ptr<Tensor>: output tensor ptr of node def
   */
  std::shared_ptr<Tensor> MutableOutputs(int32_t index) const;

  /*
   * get attr of node def.
   * @return std::unordered_map<std::string, std::shared_ptr<AttrValue>>: attrs
   * of node def
   */
  std::unordered_map<std::string, std::shared_ptr<AttrValue> > Attrs() const;

 private:
  std::shared_ptr<aicpuops::NodeDef> nodedef_{nullptr};
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_CPU_PROTO_NODE_DEF_IMPL_H
