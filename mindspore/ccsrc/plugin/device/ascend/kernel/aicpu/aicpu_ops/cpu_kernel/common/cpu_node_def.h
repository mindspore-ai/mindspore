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
#ifndef AICPU_CONTEXT_INC_NODE_DEF_H_
#define AICPU_CONTEXT_INC_NODE_DEF_H_
#include <memory>
#include <string>
#include <unordered_map>

#include "cpu_kernel/inc/cpu_attr_value.h"
#include "cpu_kernel/inc/cpu_tensor.h"

namespace aicpu {
class NodeDefImpl;
class AICPU_VISIBILITY NodeDef {
  friend class CpuKernelUtils;

 public:
  NodeDef() = delete;
  ~NodeDef() = default;

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
   * @return unordered_map<std::string, std::shared_ptr<AttrValue>>: attrs of
   * node def
   */
  std::unordered_map<std::string, std::shared_ptr<AttrValue> > Attrs() const;

 private:
  explicit NodeDef(NodeDefImpl *impl);

 private:
  std::shared_ptr<NodeDefImpl> impl_{nullptr};
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_INC_NODE_DEF_H_
