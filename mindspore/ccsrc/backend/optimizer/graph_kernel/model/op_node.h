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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_OP_NODE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_OP_NODE_H_

#include <memory>
#include <string>

#include "backend/optimizer/graph_kernel/model/node.h"
#include "ir/dtype/type.h"

namespace mindspore::graphkernel::inner {
#define CHECK_ATTR(attrs, attr_name)                                                              \
  do {                                                                                            \
    if (attrs.count(attr_name) == 0) {                                                            \
      MS_LOG(EXCEPTION) << "The attr [" << attr_name << "] does not exist in [" << #attrs << "]"; \
    }                                                                                             \
  } while (0)

class PrimOp : public Node {
 public:
  enum class ComputeType : int {
    VIRTUAL = 0,
    RESHAPE = 1,
    ELEMWISE = 2,
    BROADCAST = 3,
    REDUCE = 4,
    OPAQUE = 5,
  };

  PrimOp(const std::string &op, ComputeType compute)
      : Node({{}, TypeId::kNumberTypeBegin, kOpFormat_DEFAULT}), op_(op), compute_type_(compute) {}
  ~PrimOp() = default;

  virtual NodeBase Infer(const NodePtrList &inputs, const DAttrs &attrs);
  virtual NodePtr InferValue(const NodePtrList &inputs, const DAttrs &attrs, const std::string &op);

  std::string ToString() const override;
  NType NodeType() override { return NType::Primitive; }

  const std::string &op() const { return op_; }
  ComputeType compute_type() const { return compute_type_; }

 protected:
  virtual void Check(const NodePtrList &inputs, const DAttrs &attrs);
  virtual void CheckShape(const NodePtrList &inputs, const DAttrs &attrs) {}
  virtual void CheckType(const NodePtrList &inputs, const DAttrs &attrs);
  virtual void CheckFormat(const NodePtrList &inputs, const DAttrs &attrs);

  virtual DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) { return inputs[0]->shape; }
  virtual TypeId InferType(const NodePtrList &inputs, const DAttrs &attrs) { return inputs[0]->type; }
  virtual DFormat InferFormat(const NodePtrList &inputs, const DAttrs &attrs) { return inputs[0]->format; }

  std::string op_;
  ComputeType compute_type_;
};
using PrimOpPtr = std::shared_ptr<PrimOp>;

class ReshapeOp : public PrimOp {
 public:
  explicit ReshapeOp(const std::string &op) : PrimOp(op, ComputeType::RESHAPE) {}
  ~ReshapeOp() = default;

 protected:
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  DFormat InferFormat(const NodePtrList &inputs, const DAttrs &attrs) override {
    return attrs.find("format") == attrs.end() ? kOpFormat_DEFAULT
                                               : GetValue<std::string>(attrs.find("format")->second);
  }
};

class ElemwiseOp : public PrimOp {
 public:
  explicit ElemwiseOp(const std::string &op) : PrimOp(op, ComputeType::ELEMWISE) {}
  ~ElemwiseOp() = default;

 protected:
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  DFormat InferFormat(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class BroadcastOp : public PrimOp {
 public:
  explicit BroadcastOp(const std::string &op) : PrimOp(op, ComputeType::BROADCAST) {}
  ~BroadcastOp() = default;
};

class BroadcastToOp : public BroadcastOp {
 public:
  explicit BroadcastToOp(const std::string &op) : BroadcastOp(op) {}
  ~BroadcastToOp() = default;

 protected:
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class ReduceOp : public PrimOp {
 public:
  explicit ReduceOp(const std::string &op) : PrimOp(op, ComputeType::REDUCE) {}
  ~ReduceOp() = default;

 protected:
  void Check(const NodePtrList &inputs, const DAttrs &attrs) override;
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  DFormat InferFormat(const NodePtrList &inputs, const DAttrs &attrs) override { return kOpFormat_DEFAULT; };
};

class OpaqueOp : public PrimOp {
 public:
  explicit OpaqueOp(const std::string &op) : PrimOp(op, ComputeType::OPAQUE) {}
  ~OpaqueOp() = default;
};

class VirtualOp : public PrimOp {
 public:
  explicit VirtualOp(const std::string &op) : PrimOp(op, ComputeType::VIRTUAL) {}
  ~VirtualOp() = default;
};

class CastOp : public ElemwiseOp {
 public:
  explicit CastOp(const std::string &op) : ElemwiseOp("Cast") {}
  ~CastOp() = default;

 protected:
  TypeId InferType(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class SelectOp : public ElemwiseOp {
 public:
  explicit SelectOp(const std::string &op) : ElemwiseOp("Select") {}
  ~SelectOp() = default;

 protected:
  void CheckType(const NodePtrList &inputs, const DAttrs &attrs) override;
  TypeId InferType(const NodePtrList &inputs, const DAttrs &attrs) override { return inputs[1]->type; }
};

class CompareOp : public ElemwiseOp {
 public:
  explicit CompareOp(const std::string &op) : ElemwiseOp(op) {}
  ~CompareOp() = default;

 protected:
  TypeId InferType(const NodePtrList &inputs, const DAttrs &attrs) override { return TypeId::kNumberTypeBool; }
};

class Conv2dOp : public OpaqueOp {
 public:
  explicit Conv2dOp(const std::string &op) : OpaqueOp("Conv2D") {}
  ~Conv2dOp() = default;

 protected:
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  TypeId InferType(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class TransposeOp : public OpaqueOp {
 public:
  explicit TransposeOp(const std::string &op) : OpaqueOp("Transpose") {}
  ~TransposeOp() = default;

 protected:
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  DFormat InferFormat(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class MatMulOp : public OpaqueOp {
 public:
  explicit MatMulOp(const std::string &op) : OpaqueOp("MatMul") {}
  ~MatMulOp() = default;

 protected:
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  TypeId InferType(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class PadAkgOp : public OpaqueOp {
 public:
  explicit PadAkgOp(const std::string &op) : OpaqueOp("PadAkg") {}
  ~PadAkgOp() = default;

 protected:
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class UnPadAkgOp : public OpaqueOp {
 public:
  explicit UnPadAkgOp(const std::string &op) : OpaqueOp("UnPadAkg") {}
  ~UnPadAkgOp() = default;

 protected:
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class CImagOp : public ElemwiseOp {
 public:
  explicit CImagOp(const std::string &op) : ElemwiseOp("CImag") {}
  ~CImagOp() = default;

 protected:
  void CheckType(const NodePtrList &inputs, const DAttrs &attrs) override {
    if (inputs[0]->type != TypeId::kNumberTypeComplex64) {
      MS_LOG(EXCEPTION) << "CImag's input[0] should be complex64";
    }
  };

  TypeId InferType(const NodePtrList &inputs, const DAttrs &attrs) override { return TypeId::kNumberTypeFloat32; }
};

class CRealOp : public ElemwiseOp {
 public:
  explicit CRealOp(const std::string &op) : ElemwiseOp("CReal") {}
  ~CRealOp() = default;

 protected:
  void CheckType(const NodePtrList &inputs, const DAttrs &attrs) override {
    if (inputs[0]->type != TypeId::kNumberTypeComplex64) {
      MS_LOG(EXCEPTION) << "CReal's input[0] should be complex64";
    }
  };

  TypeId InferType(const NodePtrList &inputs, const DAttrs &attrs) override { return TypeId::kNumberTypeFloat32; }
};

class ComplexOp : public ElemwiseOp {
 public:
  explicit ComplexOp(const std::string &op) : ElemwiseOp("Complex") {}
  ~ComplexOp() = default;

 protected:
  void CheckType(const NodePtrList &inputs, const DAttrs &attrs) override;
  TypeId InferType(const NodePtrList &inputs, const DAttrs &attrs) override { return TypeId::kNumberTypeComplex64; }
};

class StandardNormalOp : public OpaqueOp {
 public:
  explicit StandardNormalOp(const std::string &op) : OpaqueOp("StandardNormal") {}
  ~StandardNormalOp() = default;

 protected:
  DShape InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  TypeId InferType(const NodePtrList &inputs, const DAttrs &attrs) override { return TypeId::kNumberTypeFloat32; }
  DFormat InferFormat(const NodePtrList &inputs, const DAttrs &attrs) override { return kOpFormat_DEFAULT; }
};
}  // namespace mindspore::graphkernel::inner
#endif
