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
#include <vector>
#include <utility>

#include "ops/primitive_c.h"
#include "backend/common/graph_kernel/model/node.h"
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

  NodeBaseList Infer(const NodePtrList &inputs, const DAttrs &attrs);

  std::string ToString() const override;
  NType NodeType() override { return NType::Primitive; }

  const std::string &op() const { return op_; }
  ComputeType compute_type() const { return compute_type_; }
  // infer output value when all inputs are constant
  virtual NodePtr InferValue(const NodePtrList &inputs, const DAttrs &attrs);

 protected:
  // Check node info before inference the shape/type/format.
  virtual void Check(const NodePtrList &, const DAttrs &) {}

  // Infer format. assume all outputs have the same format.
  virtual DFormat InferFormat(const NodePtrList &inputs, const DAttrs &) { return inputs[0]->format; }

  // Infer shape. returning an empty vector means using PrimitiveC's infer_shape function.
  virtual std::vector<DShape> InferShape(const NodePtrList &, const DAttrs &);

  // Infer type. returning an empty vector means using PrimitiveC's infer_type function.
  virtual std::vector<TypeId> InferType(const NodePtrList &, const DAttrs &);

  // calculate const inputs, used for InferValue
  template <typename TM>
  tensor::TensorPtr CalcByOperator(const NodePtrList &inputs, const DAttrs &);

  // Gen PrimitiveC and abstract list to call PrimitiveC's inference function.
  std::pair<PrimitivePtr, AbstractBasePtrList> GenPrimAndAbstract(const NodePtrList &inputs, const DAttrs &attrs) const;

  // rectify abstract before calling PrimitiveC's inference function.
  virtual void RectifyAbstract(const PrimitivePtr &, AbstractBasePtrList *) {}

  // set abstracts from attrs.
  void SetAbastractsFromAttrs(const PrimitivePtr &primitive, const mindspore::HashSet<size_t> &convert_input_list,
                              AbstractBasePtrList *inputs_abstract, std::vector<std::string> input_names_vec);

  std::string op_;
  ComputeType compute_type_;
};
using PrimOpPtr = std::shared_ptr<PrimOp>;

class ReshapeOp : public PrimOp {
 public:
  explicit ReshapeOp(const std::string &op) : PrimOp(op, ComputeType::RESHAPE) {}
  ~ReshapeOp() = default;

 protected:
  void RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list);
  DFormat InferFormat(const NodePtrList &, const DAttrs &attrs) override {
    return attrs.find("format") == attrs.end() ? kOpFormat_DEFAULT
                                               : GetValue<std::string>(attrs.find("format")->second);
  }
};

class ElemwiseOp : public PrimOp {
 public:
  explicit ElemwiseOp(const std::string &op) : PrimOp(op, ComputeType::ELEMWISE) {}
  ~ElemwiseOp() = default;

 protected:
  std::vector<DShape> InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  DFormat InferFormat(const NodePtrList &inputs, const DAttrs &) override;
};

class BroadcastOp : public PrimOp {
 public:
  explicit BroadcastOp(const std::string &op) : PrimOp(op, ComputeType::BROADCAST) {}
  ~BroadcastOp() = default;
};

class TileOp : public BroadcastOp {
 public:
  explicit TileOp(const std::string &op) : BroadcastOp(op) {}
  ~TileOp() = default;

 protected:
  void RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) override;
};

class ReduceOp : public PrimOp {
 public:
  explicit ReduceOp(const std::string &op) : PrimOp(op, ComputeType::REDUCE) {}
  ~ReduceOp() = default;

 protected:
  void RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) override;
  DFormat InferFormat(const NodePtrList &, const DAttrs &) override { return kOpFormat_DEFAULT; };
};

class ArgReduceOp : public ReduceOp {
 public:
  explicit ArgReduceOp(const std::string &op) : ReduceOp(op) {}
  ~ArgReduceOp() = default;

 protected:
  std::vector<DShape> InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  std::vector<TypeId> InferType(const NodePtrList &, const DAttrs &attrs) override;
};

class OpaqueOp : public PrimOp {
 public:
  explicit OpaqueOp(const std::string &op) : PrimOp(op, ComputeType::OPAQUE) {}
  ~OpaqueOp() = default;

 protected:
  // for pclint warning: 1790 public base symbol of symbol has no non-destructor virtual functions
  virtual void DoNothing() {}
};

class VirtualOp : public PrimOp {
 public:
  explicit VirtualOp(const std::string &op) : PrimOp(op, ComputeType::VIRTUAL) {}
  ~VirtualOp() = default;
};

class TransposeOp : public OpaqueOp {
 public:
  explicit TransposeOp(const std::string &op) : OpaqueOp(op) {}
  ~TransposeOp() = default;

 protected:
  void RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) override;
  DFormat InferFormat(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class OneHotOp : public OpaqueOp {
 public:
  explicit OneHotOp(const std::string &op) : OpaqueOp(op) {}
  ~OneHotOp() = default;

 protected:
  void RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) override;
};
class LayoutTransformOp : public OpaqueOp {
 public:
  explicit LayoutTransformOp(const std::string &op) : OpaqueOp(op) {}
  ~LayoutTransformOp() = default;

 protected:
  std::vector<DShape> InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  std::vector<TypeId> InferType(const NodePtrList &inputs, const DAttrs &) override { return {inputs[0]->type}; }
  DFormat InferFormat(const NodePtrList &, const DAttrs &attrs) override {
    return GetValue<std::string>(attrs.find("dst_format")->second);
  }
};

class ElemAnyOp : public OpaqueOp {
 public:
  explicit ElemAnyOp(const std::string &op) : OpaqueOp(op) {}
  ~ElemAnyOp() = default;

 protected:
  std::vector<DShape> InferShape(const NodePtrList &, const DAttrs &) override { return {{1}}; }
  std::vector<TypeId> InferType(const NodePtrList &, const DAttrs &) override { return {TypeId::kNumberTypeFloat32}; }
};

class ShapeOp : public OpaqueOp {
 public:
  explicit ShapeOp(const std::string &op) : OpaqueOp(op) {}
  ~ShapeOp() = default;
  NodePtr InferValue(const NodePtrList &inputs, const DAttrs &) override;

 protected:
  std::vector<DShape> InferShape(const NodePtrList &inputs, const DAttrs &) override {
    return {{SizeToLong(inputs[0]->shape.size())}};
  }
  std::vector<TypeId> InferType(const NodePtrList &, const DAttrs &) override { return {TypeId::kNumberTypeInt32}; }
  DFormat InferFormat(const NodePtrList &, const DAttrs &) override { return kOpFormat_DEFAULT; };
};

class ConstantOfShapeOp : public OpaqueOp {
 public:
  explicit ConstantOfShapeOp(const std::string &op) : OpaqueOp(op) {}
  ~ConstantOfShapeOp() = default;

  NodePtr InferValue(const NodePtrList &inputs, const DAttrs &attrs) override;

 protected:
  std::vector<DShape> InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  std::vector<TypeId> InferType(const NodePtrList &, const DAttrs &attrs) override {
    return {static_cast<TypeId>(GetValue<int64_t>(attrs.find("data_type")->second))};
  }
  DFormat InferFormat(const NodePtrList &, const DAttrs &) override { return kOpFormat_DEFAULT; }
};

class PadAkgOp : public OpaqueOp {
 public:
  explicit PadAkgOp(const std::string &op) : OpaqueOp(op) {}
  ~PadAkgOp() = default;

 protected:
  std::vector<DShape> InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  std::vector<TypeId> InferType(const NodePtrList &inputs, const DAttrs &) override { return {inputs[0]->type}; }
};

class UnPadAkgOp : public OpaqueOp {
 public:
  explicit UnPadAkgOp(const std::string &op) : OpaqueOp(op) {}
  ~UnPadAkgOp() = default;

 protected:
  std::vector<DShape> InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  std::vector<TypeId> InferType(const NodePtrList &inputs, const DAttrs &) override { return {inputs[0]->type}; }
};

class Conv2dOp : public OpaqueOp {
 public:
  explicit Conv2dOp(const std::string &op) : OpaqueOp(op) {}
  ~Conv2dOp() = default;

 protected:
  std::vector<DShape> InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  std::vector<TypeId> InferType(const NodePtrList &inputs, const DAttrs &attrs) override;
  DFormat InferFormat(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class GatherOp : public OpaqueOp {
 public:
  explicit GatherOp(const std::string &op) : OpaqueOp(op) {}
  ~GatherOp() = default;
  NodePtr InferValue(const NodePtrList &inputs, const DAttrs &attrs) override;

 protected:
  void RectifyAbstract(const PrimitivePtr &primitive, AbstractBasePtrList *input_abstract_ptr) override;
  template <typename TM>
  tensor::TensorPtr CalcGather(const NodePtrList &inputs, const DAttrs &attrs);
  DFormat InferFormat(const NodePtrList &, const DAttrs &) override { return kOpFormat_DEFAULT; };
};

class ConcatOp : public OpaqueOp {
 public:
  explicit ConcatOp(const std::string &op) : OpaqueOp(op) {}
  ~ConcatOp() = default;
  NodePtr InferValue(const NodePtrList &inputs, const DAttrs &attrs) override;

 protected:
  void RectifyAbstract(const PrimitivePtr &, AbstractBasePtrList *input_abstract_ptr) override;
  template <typename TM>
  tensor::TensorPtr CalcConcat(const NodePtrList &inputs, const DAttrs &attrs);
  DFormat InferFormat(const NodePtrList &, const DAttrs &) override { return kOpFormat_DEFAULT; };
};

class CImagRealOp : public ElemwiseOp {
 public:
  explicit CImagRealOp(const std::string &op) : ElemwiseOp(op) {}
  ~CImagRealOp() = default;

 protected:
  void Check(const NodePtrList &inputs, const DAttrs &) override {
    if (inputs[0]->type != TypeId::kNumberTypeComplex64) {
      MS_LOG(EXCEPTION) << op_ << "'s input[0] should be complex64, but got " << TypeIdToString(inputs[0]->type, true);
    }
  };

  std::vector<DShape> InferShape(const NodePtrList &inputs, const DAttrs &) override { return {inputs[0]->shape}; }
  std::vector<TypeId> InferType(const NodePtrList &, const DAttrs &) override { return {TypeId::kNumberTypeFloat32}; }
};

class Pool2DOp : public OpaqueOp {
 public:
  explicit Pool2DOp(const std::string &op) : OpaqueOp(op) {}
  ~Pool2DOp() = default;

 protected:
  std::vector<DShape> InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  std::vector<TypeId> InferType(const NodePtrList &inputs, const DAttrs &) override { return {inputs[0]->type}; }
};

class ComplexOp : public ElemwiseOp {
 public:
  explicit ComplexOp(const std::string &op) : ElemwiseOp(op) {}
  ~ComplexOp() = default;

 protected:
  void Check(const NodePtrList &inputs, const DAttrs &) override;
  std::vector<DShape> InferShape(const NodePtrList &inputs, const DAttrs &) override { return {inputs[0]->shape}; }
  std::vector<TypeId> InferType(const NodePtrList &, const DAttrs &) override { return {TypeId::kNumberTypeComplex64}; }
};

class StandardNormalOp : public OpaqueOp {
 public:
  explicit StandardNormalOp(const std::string &op) : OpaqueOp(op) {}
  ~StandardNormalOp() = default;

 protected:
  std::vector<DShape> InferShape(const NodePtrList &, const DAttrs &attrs) override;
  std::vector<TypeId> InferType(const NodePtrList &, const DAttrs &) override { return {TypeId::kNumberTypeFloat32}; }
  DFormat InferFormat(const NodePtrList &, const DAttrs &) override { return kOpFormat_DEFAULT; }
};

class StridedSliceOp : public OpaqueOp {
 public:
  explicit StridedSliceOp(const std::string &op) : OpaqueOp(op) {}
  ~StridedSliceOp() = default;

 protected:
  void RectifyAbstract(const PrimitivePtr &primitive, AbstractBasePtrList *inputs_abstract) override;
};

class StridedSliceOnnxOp : public OpaqueOp {
 public:
  explicit StridedSliceOnnxOp(const std::string &op) : OpaqueOp(op) {}
  ~StridedSliceOnnxOp() = default;
  NodePtr InferValue(const NodePtrList &inputs, const DAttrs &attrs) override;

 protected:
  template <typename TM>
  tensor::TensorPtr CalcStridedSliceOnnx(const NodePtrList &inputs, const DAttrs &);
  std::vector<DShape> InferShape(const NodePtrList &, const DAttrs &attrs) override {
    return GetValue<std::vector<DShape>>(attrs.find("output_shape")->second);
  }
  std::vector<TypeId> InferType(const NodePtrList &inputs, const DAttrs &) override { return {inputs[0]->type}; }
  DFormat InferFormat(const NodePtrList &, const DAttrs &) override { return kOpFormat_DEFAULT; }
};

class MatMulOp : public OpaqueOp {
 public:
  explicit MatMulOp(const std::string &op) : OpaqueOp(op) {}
  ~MatMulOp() = default;

 protected:
  std::vector<DShape> InferShape(const NodePtrList &inputs, const DAttrs &attrs) override;
  std::vector<TypeId> InferType(const NodePtrList &inputs, const DAttrs &attrs) override;
};

class TupleGetItemOp : public VirtualOp {
 public:
  using VirtualOp::VirtualOp;
  ~TupleGetItemOp() = default;

 protected:
  void RectifyAbstract(const PrimitivePtr &prim, AbstractBasePtrList *abs_list) override;
};
}  // namespace mindspore::graphkernel::inner
#endif
