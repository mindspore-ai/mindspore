/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain x copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops/reshape_ext.h"

#include <string>
#include "ops/scalar_graph_holder.h"
#include "ops/sequence_ops.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/array_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kReshapeExtInputsNum = 3;
typedef int64_t (*ARITHMETIC)(const int64_t &x, const int64_t &y);
int64_t Add(const int64_t &x, const int64_t &y) { return x + y; }
int64_t Sub(const int64_t &x, const int64_t &y) { return x - y; }
int64_t Mul(const int64_t &x, const int64_t &y) { return x * y; }
int64_t Div(const int64_t &x, const int64_t &y) { return x / y; }
int64_t FloorDiv(const int64_t &x, const int64_t &y) { return floor(float(x) / y); }

std::map<PrimitivePtr, ARITHMETIC> arith_func_map = {
  {prim::kPrimScalarAdd, Add}, {prim::kPrimScalarSub, Sub},           {prim::kPrimScalarMul, Mul},
  {prim::kPrimScalarDiv, Div}, {prim::kPrimScalarFloorDiv, FloorDiv},
};

void SetScalarValueForNode(const AnfNodePtr &node, const ScalarGraphHolderPtr &graph,
                           const std::vector<AbstractBasePtr> &input_args) {
  auto cnode = node->cast<CNodePtr>();
  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  if (IsPrimitiveEquals(prim, prim::kPrimShape)) {
    auto input_index = graph->GetShapeIndex(node);
    graph->SetScalarValue(node, input_args[input_index - 1]->GetShape()->GetShapeVector());
  } else if (IsPrimitiveEquals(prim, prim::kPrimTupleGetItem) || IsPrimitiveEquals(prim, prim::kPrimRealTupleGetItem)) {
    auto get_item_input = cnode->inputs().at(kIndex1);
    auto get_item_index = cnode->inputs().at(kIndex2);
    auto input_value = graph->GetScalarValue(get_item_input);
    auto index_value = LongToSize(graph->GetScalarValue(get_item_index).at(0));
    graph->SetScalarValue(node, {input_value[index_value]});
  } else if (IsPrimitiveEquals(prim, prim::kPrimScalarAdd) || IsPrimitiveEquals(prim, prim::kPrimScalarSub) ||
             IsPrimitiveEquals(prim, prim::kPrimScalarMul) || IsPrimitiveEquals(prim, prim::kPrimScalarDiv) ||
             IsPrimitiveEquals(prim, prim::kPrimScalarFloorDiv)) {
    auto x = graph->GetScalarValue(cnode->inputs().at(kIndex1)).at(0);
    auto y = graph->GetScalarValue(cnode->inputs().at(kIndex2)).at(0);
    for (const auto &itr : arith_func_map) {
      if (IsPrimitiveEquals(prim, itr.first)) {
        auto arith_func = itr.second;
        graph->SetScalarValue(node, {arith_func(x, y)});

      } else {
        MS_LOG_EXCEPTION << "Can't find the function for scalar arithmetic operator.";
      }
    }
  } else if (IsPrimitiveEquals(prim, prim::kPrimMakeTuple) || IsPrimitiveEquals(prim, prim::kPrimRealMakeTuple)) {
    std::vector<int64_t> tuple;
    for (size_t i = 1; i < cnode->size(); i++) {
      auto in = cnode->inputs().at(i);
      tuple.push_back(graph->GetScalarValue(in).at(0));
    }
    graph->SetScalarValue(node, tuple);
  } else if (IsPrimitiveEquals(prim, prim::kPrimReshape)) {
    auto shape = cnode->inputs().at(kIndex2);
    graph->SetScalarValue(node, graph->GetScalarValue(shape));
  } else {
    MS_LOG_EXCEPTION
      << "The CNode in ReshapeExt graph should in the whitelist. Please check the ShapeReshapeFusion pass.";
  }
}

abstract::ShapePtr ReshapeExtInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto attr = primitive->GetAttr("graph");
  MS_EXCEPTION_IF_NULL(attr);
  auto graph = attr->cast<ScalarGraphHolderPtr>();

  for (size_t i = 0; i < graph->GetNodeSize(); ++i) {
    auto node = graph->GetAnfNode(i);
    if (node->isa<ValueNode>()) {
      continue;
    } else if (node->isa<CNode>()) {
      SetScalarValueForNode(node, graph, input_args);
    } else {
      MS_LOG_EXCEPTION
        << "The node in ReshapeExt graph should be ValueNode or CNode. Please check the ShapeReshapeFusion pass.";
    }
  }

  // The last node is Reshape.
  auto reshape_node = graph->GetAnfNode(graph->GetNodeSize() - 1);
  if (!IsPrimitiveCNode(reshape_node, prim::kPrimReshape)) {
    MS_LOG_EXCEPTION
      << "The last node in ReshapeExt graph should be Reshape. Please check the ShapeReshapeFusion pass.";
  }
  auto key_shape = std::make_shared<abstract::Shape>(graph->GetScalarValue(reshape_node));
  return key_shape;  // output shape
}

TypePtr ReshapeExtInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto input_type = input_args[1]->GetType();
  return input_type;  // output type
}
}  // namespace

AbstractBasePtr ReshapeExtInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto ordinary_input_num = CheckAndConvertUtils::GetRemoveUMonadAbsNum(input_args);
  (void)CheckAndConvertUtils::CheckInteger("inputs num", SizeToLong(ordinary_input_num), kEqual, kReshapeExtInputsNum,
                                           prim_name);
  auto infer_type = ReshapeExtInferType(primitive, input_args);
  auto infer_shape = ReshapeExtInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(ReshapeExt, BaseOperator);

// AG means auto generated
class MIND_API AGReshapeExtInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ReshapeExtInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ReshapeExtInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ReshapeExtInfer(engine, primitive, input_args);
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ReshapeExt, prim::kPrimReshapeExt, AGReshapeExtInfer, false);
}  // namespace ops
}  // namespace mindspore
