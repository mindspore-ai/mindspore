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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_OP_NODE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_OP_NODE_H_

#include <memory>
#include <algorithm>
#include <sstream>
#include <string>

#include "backend/optimizer/graph_kernel/model/node.h"

namespace mindspore {
namespace opt {
namespace graphkernel {
class PrimOp : public Node {
 public:
  enum ComputeType {
    RESHAPE,
    ELEMWISE,
    BROADCAST,
    REDUCE,
    OPAQUE,
  };

  PrimOp(const std::string &op, const std::string &node_name, ComputeType compute)
      : Node({{}, TypeId::kNumberTypeBegin, kOpFormat_DEFAULT}, node_name), op_(op), compute_type_(compute) {}

  virtual void Infer(const NodePtrList &inputs, const DAttrs &attrs);
  void Dump(std::ostringstream &os) const override;
  NType NodeType() override { return NType::Primitive; }

  const std::string &op() const { return op_; }
  ComputeType compute_type() const { return compute_type_; }
  virtual NodePtr InferValue(const NodePtrList &inputs, const DAttrs &attrs, const std::string &op);

 protected:
  std::string op_;
  ComputeType compute_type_;
  virtual DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) { return inputs[0]->shape; }
  virtual TypeId InferType(const NodePtrList &inputs, const DAttrs &attrs) { return inputs[0]->type; }
  virtual DFormat InferFormat(const NodePtrList &inputs, const DAttrs &attrs) { return inputs[0]->format; }
};
using PrimOpPtr = std::shared_ptr<PrimOp>;

class ElemwiseOp : public PrimOp {
 public:
  ElemwiseOp(const std::string &op, const std::string &node_name) : PrimOp(op, node_name, ELEMWISE) {}
  void Infer(const NodePtrList &inputs, const DAttrs &attrs) override;
  // TODO(dayschan) rewrite InferShape/InferFormat
};

class ReshapeOp : public PrimOp {
 public:
  ReshapeOp(const std::string &op, const std::string &node_name) : PrimOp(op, node_name, RESHAPE) {}

 protected:
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class BroadcastToOp : public PrimOp {
 public:
  BroadcastToOp(const std::string &op, const std::string &node_name) : PrimOp(op, node_name, BROADCAST) {}

 protected:
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class ReduceOp : public PrimOp {
 public:
  ReduceOp(const std::string &op, const std::string &node_name) : PrimOp(op, node_name, REDUCE) {}

 protected:
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class OpaqueOp : public PrimOp {
 public:
  OpaqueOp(const std::string &op, const std::string &node_name) : PrimOp(op, node_name, OPAQUE) {}
};

class Conv2dOp : public OpaqueOp {
 public:
  Conv2dOp(const std::string &op, const std::string &node_name) : OpaqueOp("Conv2D", node_name) {}
};
}  // namespace graphkernel
}  // namespace opt
}  // namespace mindspore
#endif
