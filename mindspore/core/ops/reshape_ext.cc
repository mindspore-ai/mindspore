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

std::map<OpType, ARITHMETIC> arith_func_map = {
  {OpType::SCALAR_ADD, Add},
  {OpType::SCALAR_SUB, Sub},
  {OpType::SCALAR_MUL, Mul},
  {OpType::SCALAR_DIV, Div},
  {OpType::SCALAR_FLOOR_DIV, FloorDiv},
};

void CalScalarValueForGraph(const ScalarGraphHolderPtr &graph, const std::vector<AbstractBasePtr> &input_args) {
  size_t shape_index = 0;
  for (size_t i = 0; i < graph->GetNodeSize(); ++i) {
    auto node = graph->GetScalarNode(i);
    switch (node->type_) {
      case OpType::VALUE:
        break;
      case OpType::SHAPE: {
        auto index = graph->GetShapeIndex().at(shape_index);
        shape_index++;
        graph->SetScalarValue(i, input_args[index - 1]->GetShape()->GetShapeVector());
        break;
      }
      case OpType::RESHAPE: {
        auto index = node->in_index_.at(kIndex1);
        graph->SetScalarValue(i, graph->GetScalarValue(index));
        break;
      }
      case OpType::TUPLE_GET_ITEM: {
        auto get_item_input = node->in_index_.at(kIndex0);
        auto get_item_index = node->in_index_.at(kIndex1);
        auto input_value = graph->GetScalarValue(get_item_input);
        auto index_value = LongToSize(graph->GetScalarValue(get_item_index).at(0));
        graph->SetScalarValue(i, {input_value[index_value]});
        break;
      }
      case OpType::MAKE_TUPLE: {
        std::vector<int64_t> tuple;
        for (size_t j = 0; j < node->in_index_.size(); ++j) {
          tuple.push_back(graph->GetScalarValue(node->in_index_.at(j)).at(0));
        }
        graph->SetScalarValue(i, tuple);
        break;
      }
      case OpType::SCALAR_ADD:
      case OpType::SCALAR_SUB:
      case OpType::SCALAR_MUL:
      case OpType::SCALAR_DIV:
      case OpType::SCALAR_FLOOR_DIV: {
        auto x = graph->GetScalarValue(node->in_index_.at(kIndex0)).at(0);
        auto y = graph->GetScalarValue(node->in_index_.at(kIndex1)).at(0);
        auto itr = arith_func_map.find(node->type_);
        if (itr != arith_func_map.end()) {
          auto arith_func = itr->second;
          graph->SetScalarValue(i, {arith_func(x, y)});
        } else {
          MS_LOG_EXCEPTION << "Can't find the function for scalar arithmetic operator.";
        }
        break;
      }
      default:
        MS_LOG_EXCEPTION
          << "The Node in ReshapeExt graph should in the whitelist. Please check the ShapeReshapeFusion pass.";
        break;
    }
  }
}

abstract::ShapePtr ReshapeExtInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto attr = primitive->GetAttr("graph");
  MS_EXCEPTION_IF_NULL(attr);
  auto graph = attr->cast<ScalarGraphHolderPtr>();

  CalScalarValueForGraph(graph, input_args);

  // The last node is Reshape.
  auto reshape_node = graph->GetScalarNode(graph->GetNodeSize() - 1);
  if (reshape_node->type_ != OpType::RESHAPE) {
    MS_LOG_EXCEPTION
      << "The last node in ReshapeExt graph should be Reshape. Please check the ShapeReshapeFusion pass.";
  }
  auto key_shape = std::make_shared<abstract::Shape>(graph->GetScalarValue(graph->GetNodeSize() - 1));
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
