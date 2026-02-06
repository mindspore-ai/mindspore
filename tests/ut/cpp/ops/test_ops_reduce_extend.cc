/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <memory>
#include "common/common_test.h"
#include "infer/ops_func_impl/mean_ext.h"
#include "infer/ops_func_impl/sum_ext.h"
#include "infer/ops_func_impl/prod_ext.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "ir/tensor_new.h"

namespace mindspore {
namespace ops {
namespace {
struct ReduceExtendParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ShapeVector output_shape;
  TypePtr output_type;
  AbstractBasePtr axis;
  AbstractBasePtr keep_dims;
  AbstractBasePtr dtype;
};

static auto value_any = mindspore::kValueAny->ToAbstract();
static auto value_none = mindspore::kNone->ToAbstract();
static auto keep_dims_true = std::make_shared<BoolImm>(true)->ToAbstract();
static auto keep_dims_false = std::make_shared<BoolImm>(false)->ToAbstract();
static auto dtype_float64 = std::make_shared<Int64Imm>(kNumberTypeFloat64)->ToAbstract();
static auto dtype_int32 = std::make_shared<Int64Imm>(kNumberTypeInt32)->ToAbstract();
static auto dtype_int16 = std::make_shared<Int64Imm>(kNumberTypeInt16)->ToAbstract();
static auto dtype_int8 = std::make_shared<Int64Imm>(kNumberTypeInt8)->ToAbstract();
static auto dtype_uint8 = std::make_shared<Int64Imm>(kNumberTypeUInt8)->ToAbstract();
static auto dtype_bool = std::make_shared<Int64Imm>(kNumberTypeBool)->ToAbstract();

AbstractBasePtr CreateInt(const int &value) { return CreatePyInt(value)->ToAbstract(); }

AbstractBasePtr CreateIntTuple(const std::vector<NumberContainer> &value) {
  return CreatePyIntTuple(value)->ToAbstract();
}

template <typename T>
tensor::TensorPtr CreateTensor(const ShapeVector &shape, const TypeId &dtype, std::vector<T> value) {
  void *data_ptr = &value[0];
  auto tensor = tensor::from_buffer(dtype, shape, data_ptr, dtype);
  return tensor;
}

static std::map<std::string, OpFuncImplPtr> reduce_extand_func_impl = {
  {kNameMeanExt, std::make_shared<MeanExtFuncImpl>()},
  {kNameSumExt, std::make_shared<SumExtFuncImpl>()},
  {kNameProdExt, std::make_shared<ProdExtFuncImpl>()},
};
}  // namespace

class TestReduceExtend : public TestOps,
                         public testing::WithParamInterface<std::tuple<const char *, ReduceExtendParams>> {};

TEST_P(TestReduceExtend, dyn_shape) {
  const auto &op_name = std::get<0>(GetParam());
  const auto &param = std::get<1>(GetParam());
  ASSERT_TRUE(reduce_extand_func_impl.find(op_name) != reduce_extand_func_impl.end());
  auto op_impl = reduce_extand_func_impl[op_name];
  ASSERT_NE(op_impl, nullptr);

  auto prim = std::make_shared<Primitive>(op_name);
  ASSERT_NE(prim, nullptr);
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  ASSERT_NE(input, nullptr);
  auto input_args = std::vector<AbstractBasePtr>{input, param.axis, param.keep_dims, param.dtype};

  auto expect_shape = std::make_shared<abstract::TensorShape>(param.output_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = std::make_shared<TensorType>(param.output_type);
  ASSERT_NE(expect_type, nullptr);

  auto out_shape = op_impl->InferShape(prim, input_args);
  auto out_type = op_impl->InferType(prim, input_args);

  ShapeCompare(out_shape, expect_shape);
  TypeCompare(out_type, expect_type);
}

class TestReduceExtendSimpleInfer : public TestOps, public testing::WithParamInterface<std::tuple<const char *, ReduceExtendParams>> {};

TEST_P(TestReduceExtendSimpleInfer, simple_infer) {
  const auto &op_name = std::get<0>(GetParam());
  const auto &param = std::get<1>(GetParam());
  ASSERT_TRUE(reduce_extand_func_impl.find(op_name) != reduce_extand_func_impl.end());
  auto op_impl = reduce_extand_func_impl[op_name];
  ASSERT_NE(op_impl, nullptr);

  auto prim = std::make_shared<Primitive>(op_name);
  ASSERT_NE(prim, nullptr);
  auto input = tensor::from_spec(param.input_type->type_id(), param.input_shape, device::DeviceType::kCPU);
  ASSERT_NE(input, nullptr);
  ValuePtrList input_values;
  input_values.push_back(std::move(input));
  input_values.push_back(std::move(param.axis->GetValue()));
  input_values.push_back(std::move(param.keep_dims->GetValue()));
  input_values.push_back(std::move(param.dtype->GetValue()));

  auto expect_shape = ShapeArray{param.output_shape};
  auto expect_type = TypePtrList{param.output_type};

  auto output_shape = op_impl->InferShape(prim, input_values);
  auto output_type = op_impl->InferType(prim, input_values);

  ShapeCompare(output_shape, expect_shape);
  TypeCompare(output_type, expect_type);
}

auto ReduceExtendTestCase = testing::ValuesIn(
  {ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 1, 4}, kFloat32, CreateIntTuple({1}), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 4}, kFloat32, CreateIntTuple({1}), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {1, 1, 4}, kFloat32, CreateIntTuple({0, 1}), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {4}, kFloat32, CreateIntTuple({0, 1}), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 3, 1}, kFloat32, CreateIntTuple({-1}), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 4}, kFloat32, CreateIntTuple({-2}), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 1, 1}, kFloat32, CreateIntTuple({-1, -2}), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {4}, kFloat32, CreateIntTuple({-2, -3}), keep_dims_false, value_none},
   ReduceExtendParams{
     {2, 3, 4}, kFloat32, {-1, 1, -1}, kFloat32, CreateIntTuple({kValueAny, 1}), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {-1}, kFloat32, CreateIntTuple({kValueAny, 1}), keep_dims_false, value_none},
   ReduceExtendParams{
     {2, 3, 4}, kFloat32, {-1, -1, -1}, kFloat32, CreateIntTuple({kValueAny, kValueAny}), keep_dims_true, value_none},
   ReduceExtendParams{
     {2, 3, 4}, kFloat32, {-1}, kFloat32, CreateIntTuple({kValueAny, kValueAny}), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {-1, -1, -1}, kFloat32, value_any, keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {-2}, kFloat32, value_any, keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {}, kFloat32, CreateIntTuple({}), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {-2}, kFloat32, CreateIntTuple({1}), value_any, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {-2}, kFloat32, CreateIntTuple({1, 2}), value_any, value_none},
   ReduceExtendParams{{-1, -1, 4}, kFloat32, {-1, 1, 4}, kFloat32, CreateIntTuple({1}), keep_dims_true, value_none},
   ReduceExtendParams{{-1, -1, 4}, kFloat32, {-1, 4}, kFloat32, CreateIntTuple({1}), keep_dims_false, value_none},
   ReduceExtendParams{{-1, 3, 4}, kFloat32, {1, 3, 1}, kFloat32, CreateIntTuple({0, 2}), keep_dims_true, value_none},
   ReduceExtendParams{{-1, 3, 4}, kFloat32, {3}, kFloat32, CreateIntTuple({0, 2}), keep_dims_false, value_none},
   ReduceExtendParams{
     {-1, -1, 4}, kFloat32, {-1, 1, -1}, kFloat32, CreateIntTuple({kValueAny, 1}), keep_dims_true, value_none},
   ReduceExtendParams{
     {-1, -1, 4}, kFloat32, {-1}, kFloat32, CreateIntTuple({kValueAny, 1}), keep_dims_false, value_none},
   ReduceExtendParams{
     {-1, -1, 4}, kFloat32, {-1, -1, -1}, kFloat32, CreateIntTuple({kValueAny, kValueAny}), keep_dims_true, value_none},
   ReduceExtendParams{
     {-1, -1, 4}, kFloat32, {-1}, kFloat32, CreateIntTuple({kValueAny, kValueAny}), keep_dims_false, value_none},
   ReduceExtendParams{{-1, -1, 4}, kFloat32, {-1, -1, -1}, kFloat32, value_any, keep_dims_true, value_none},
   ReduceExtendParams{{-1, -1, 4}, kFloat32, {-2}, kFloat32, value_any, keep_dims_false, value_none},
   ReduceExtendParams{{-1, -1, 4}, kFloat32, {}, kFloat32, CreateIntTuple({}), keep_dims_false, value_none},
   ReduceExtendParams{{-1, -1, 4}, kFloat32, {-2}, kFloat32, CreateIntTuple({1}), value_any, value_none},
   ReduceExtendParams{{-1, -1, 4}, kFloat32, {-2}, kFloat32, CreateIntTuple({1, 2}), value_any, value_none},
   ReduceExtendParams{{-2}, kFloat32, {-2}, kFloat32, CreateIntTuple({1}), keep_dims_true, value_none},
   ReduceExtendParams{{-2}, kFloat32, {-2}, kFloat32, CreateIntTuple({0, 2}), keep_dims_false, value_none},
   ReduceExtendParams{{-2}, kFloat32, {-2}, kFloat32, CreateIntTuple({kValueAny, 1}), keep_dims_true, value_none},
   ReduceExtendParams{{-2}, kFloat32, {-2}, kFloat32, value_any, keep_dims_true, value_none},
   ReduceExtendParams{{-2}, kFloat32, {}, kFloat32, CreateIntTuple({}), keep_dims_false, value_none},
   ReduceExtendParams{{-1, -1, -1}, kFloat32, {1, 1, 1}, kFloat64, value_none, keep_dims_true, dtype_float64},
   ReduceExtendParams{{-1, -1, -1}, kFloat32, {}, kFloat64, value_none, keep_dims_false, dtype_float64},
   ReduceExtendParams{{-2}, kFloat32, {-2}, kFloat64, value_none, keep_dims_true, dtype_float64},
   ReduceExtendParams{{-2}, kFloat32, {}, kFloat64, value_none, keep_dims_false, dtype_float64},
   ReduceExtendParams{{-1, -1, -1}, kFloat32, {-2}, kFloat64, value_none, value_any, dtype_float64},
   ReduceExtendParams{{-2}, kFloat32, {-2}, kFloat64, value_none, value_any, dtype_float64},
   ReduceExtendParams{{}, kFloat32, {-2}, kFloat32, CreateIntTuple({0}), value_any, value_none},
   ReduceExtendParams{{}, kFloat32, {}, kFloat32, CreateIntTuple({0}), keep_dims_true, value_none},
   ReduceExtendParams{{}, kFloat32, {}, kFloat32, CreateIntTuple({0}), keep_dims_false, value_none},
   ReduceExtendParams{{}, kFloat32, {-2}, kFloat32, value_any, value_any, value_none},
   ReduceExtendParams{{}, kFloat32, {}, kFloat32, value_any, keep_dims_true, value_none},
   ReduceExtendParams{{}, kFloat32, {-2}, kFloat32, value_any, keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {1, 1, 1}, kFloat32, value_none, keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {1, 1, 1}, kFloat32, CreateIntTuple({}), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {}, kFloat32, value_none, keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {}, kFloat32, CreateIntTuple({}), keep_dims_false, value_none}});

auto ReduceExtendTestCase_ProdExt = testing::ValuesIn(
  {ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 1, 4}, kFloat32, CreateInt(1), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 4}, kFloat32, CreateInt(1), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 3, 1}, kFloat32, CreateInt(-1), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 4}, kFloat32, CreateInt(-2), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {-1, -1, -1}, kFloat32, value_any, keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {-1, -1}, kFloat32, value_any, keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {-2}, kFloat32, CreateInt(1), value_any, value_none},
   ReduceExtendParams{{-1, -1, 4}, kFloat32, {-1, 1, 4}, kFloat32, CreateInt(1), keep_dims_true, value_none},
   ReduceExtendParams{{-1, -1, 4}, kFloat32, {-1, 4}, kFloat32, CreateInt(1), keep_dims_false, value_none},
   ReduceExtendParams{{-1, -1, 4}, kFloat32, {-1, -1, -1}, kFloat32, value_any, keep_dims_true, value_none},
   ReduceExtendParams{{-1, -1, 4}, kFloat32, {-1, -1}, kFloat32, value_any, keep_dims_false, value_none},
   ReduceExtendParams{{-1, -1, 4}, kFloat32, {-2}, kFloat32, CreateInt(1), value_any, value_none},
   ReduceExtendParams{{-2}, kFloat32, {-2}, kFloat32, CreateInt(1), keep_dims_true, value_none},
   ReduceExtendParams{{-2}, kFloat32, {-2}, kFloat32, value_any, keep_dims_true, value_none},
   ReduceExtendParams{{-1, -1, -1}, kFloat32, {}, kFloat64, value_none, keep_dims_false, dtype_float64},
   ReduceExtendParams{{-2}, kFloat32, {}, kFloat64, value_none, keep_dims_false, dtype_float64},
   ReduceExtendParams{{-1, -1, -1}, kFloat32, {-2}, kFloat64, value_none, value_any, dtype_float64},
   ReduceExtendParams{{-2}, kFloat32, {-2}, kFloat64, value_none, value_any, dtype_float64},
   ReduceExtendParams{{}, kFloat32, {-2}, kFloat32, CreateInt(0), value_any, value_none},
   ReduceExtendParams{{}, kFloat32, {}, kFloat32, CreateInt(0), keep_dims_true, value_none},
   ReduceExtendParams{{}, kFloat32, {}, kFloat32, CreateInt(0), keep_dims_false, value_none},
   ReduceExtendParams{{}, kFloat32, {-2}, kFloat32, value_any, value_any, value_none},
   ReduceExtendParams{{}, kFloat32, {}, kFloat32, value_any, keep_dims_true, value_none},
   ReduceExtendParams{{}, kFloat32, {}, kFloat32, value_any, keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {}, kFloat32, value_none, keep_dims_false, value_none}});

auto ReduceExtendTestCase_ExtraDtype = testing::ValuesIn(
  {ReduceExtendParams{{}, kFloat32, {}, kFloat32, value_none, keep_dims_false, value_none},
   ReduceExtendParams{{}, kComplex64, {}, kComplex64, value_none, keep_dims_false, value_none},
   ReduceExtendParams{{}, kInt32, {}, kInt64, value_none, keep_dims_false, value_none},
   ReduceExtendParams{{}, kInt16, {}, kInt64, value_none, keep_dims_false, value_none},
   ReduceExtendParams{{}, kInt8, {}, kInt64, value_none, keep_dims_false, value_none},
   ReduceExtendParams{{}, kUInt8, {}, kInt64, value_none, keep_dims_false, value_none},
   ReduceExtendParams{{}, kBool, {}, kInt64, value_none, keep_dims_false, value_none},
   ReduceExtendParams{{}, kInt32, {}, kBool, value_none, keep_dims_false, dtype_bool},
   ReduceExtendParams{{}, kInt16, {}, kUInt8, value_none, keep_dims_false, dtype_uint8},
   ReduceExtendParams{{}, kInt8, {}, kBool, value_none, keep_dims_false, dtype_bool},
   ReduceExtendParams{{}, kUInt8, {}, kInt16, value_none, keep_dims_false, dtype_int16},
   ReduceExtendParams{{}, kBool, {}, kInt32, value_none, keep_dims_false, dtype_int32}});

INSTANTIATE_TEST_CASE_P(TestMeanExtGroup, TestReduceExtend,
                        testing::Combine(testing::ValuesIn({kNameMeanExt}), ReduceExtendTestCase));
INSTANTIATE_TEST_CASE_P(TestSumExtGroup, TestReduceExtend,
                        testing::Combine(testing::ValuesIn({kNameSumExt}), ReduceExtendTestCase));
INSTANTIATE_TEST_CASE_P(TestSumExtGroup_ExtraDtype, TestReduceExtend,
                        testing::Combine(testing::ValuesIn({kNameSumExt}), ReduceExtendTestCase_ExtraDtype));
INSTANTIATE_TEST_CASE_P(TestProdExtGroup, TestReduceExtend,
                        testing::Combine(testing::ValuesIn({kNameProdExt}), ReduceExtendTestCase_ProdExt));
INSTANTIATE_TEST_CASE_P(TestProdExtGroup_ExtraDtype, TestReduceExtend,
                        testing::Combine(testing::ValuesIn({kNameProdExt}), ReduceExtendTestCase_ExtraDtype));

auto ReduceExtendSimpleInferTestCase = testing::ValuesIn(
  {ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 1, 4}, kFloat32, CreateIntTuple({1}), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 4}, kFloat32, CreateIntTuple({1}), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {1, 1, 4}, kFloat32, CreateIntTuple({0, 1}), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {4}, kFloat32, CreateIntTuple({0, 1}), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 3, 1}, kFloat32, CreateIntTuple({-1}), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 4}, kFloat32, CreateIntTuple({-2}), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 1, 1}, kFloat32, CreateIntTuple({-1, -2}), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {4}, kFloat32, CreateIntTuple({-2, -3}), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {}, kFloat32, CreateIntTuple({}), keep_dims_false, value_none},
   ReduceExtendParams{{}, kFloat32, {}, kFloat32, CreateIntTuple({0}), keep_dims_true, value_none},
   ReduceExtendParams{{}, kFloat32, {}, kFloat32, CreateIntTuple({0}), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {1, 1, 1}, kFloat32, value_none, keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {1, 1, 1}, kFloat32, CreateIntTuple({}), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {}, kFloat32, value_none, keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {}, kFloat32, CreateIntTuple({}), keep_dims_false, value_none}});

auto ReduceExtendSimpleInferTestCase_ProdExt = testing::ValuesIn(
  {ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 1, 4}, kFloat32, CreateInt(1), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 4}, kFloat32, CreateInt(1), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 3, 1}, kFloat32, CreateInt(-1), keep_dims_true, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {2, 4}, kFloat32, CreateInt(-2), keep_dims_false, value_none},
   ReduceExtendParams{{}, kFloat32, {}, kFloat32, CreateInt(0), keep_dims_true, value_none},
   ReduceExtendParams{{}, kFloat32, {}, kFloat32, CreateInt(0), keep_dims_false, value_none},
   ReduceExtendParams{{2, 3, 4}, kFloat32, {}, kFloat32, value_none, keep_dims_false, value_none}});

INSTANTIATE_TEST_CASE_P(TestMeanExtGroup, TestReduceExtendSimpleInfer,
                        testing::Combine(testing::ValuesIn({kNameMeanExt}), ReduceExtendSimpleInferTestCase));
INSTANTIATE_TEST_CASE_P(TestSumExtGroup, TestReduceExtendSimpleInfer,
                        testing::Combine(testing::ValuesIn({kNameSumExt}), ReduceExtendSimpleInferTestCase));
INSTANTIATE_TEST_CASE_P(TestSumExtGroup_ExtraDtype, TestReduceExtendSimpleInfer,
                        testing::Combine(testing::ValuesIn({kNameSumExt}), ReduceExtendTestCase_ExtraDtype));
INSTANTIATE_TEST_CASE_P(TestProdExtGroup, TestReduceExtendSimpleInfer,
                        testing::Combine(testing::ValuesIn({kNameProdExt}), ReduceExtendSimpleInferTestCase_ProdExt));
INSTANTIATE_TEST_CASE_P(TestProdExtGroup_ExtraDtype, TestReduceExtendSimpleInfer,
                        testing::Combine(testing::ValuesIn({kNameProdExt}), ReduceExtendTestCase_ExtraDtype));

struct ReduceExtendInferValueParams {
  tensor::TensorPtr input;
  AbstractBasePtr axis;
  AbstractBasePtr keep_dims;
  AbstractBasePtr dtype;
  tensor::TensorPtr out;
};

class TestReduceExtendInferValue
    : public TestOps,
      public testing::WithParamInterface<std::tuple<const char *, ReduceExtendInferValueParams>> {};

TEST_P(TestReduceExtendInferValue, dyn_shape_infer_value) {
  const auto &op_name = std::get<0>(GetParam());
  const auto &param = std::get<1>(GetParam());

  auto primitive = std::make_shared<Primitive>(op_name);
  ASSERT_NE(primitive, nullptr);

  ASSERT_NE(param.input, nullptr);
  auto input = param.input->ToAbstract();
  ASSERT_NE(input, nullptr);

  auto input_args = abstract::AbstractBasePtrList{input, param.axis, param.keep_dims, param.dtype};
  auto value_opt = abstract::InferValueByFuncImpl(primitive, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << op_name << " have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << op_name << " can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

auto ReduceExtendInferValueTestCase_MeanExt = testing::ValuesIn(
  {ReduceExtendInferValueParams{
     CreateTensor<float>(ShapeVector{2, 2}, kNumberTypeFloat32, std::vector<float>{2, 3, 4, 5}), CreateIntTuple({0}),
     keep_dims_false, dtype_float64,
     CreateTensor<double>(ShapeVector{2}, kNumberTypeFloat64, std::vector<double>{3, 4})},
   ReduceExtendInferValueParams{
     CreateTensor<float>(ShapeVector{2, 2}, kNumberTypeFloat32, std::vector<float>{2, 3, 4, 5}), value_none,
     keep_dims_false, dtype_float64, CreateTensor<double>(ShapeVector{}, kNumberTypeFloat64, std::vector<double>{3.5})},
   ReduceExtendInferValueParams{
     CreateTensor<float>(ShapeVector{2, 2}, kNumberTypeFloat32, std::vector<float>{2, 3, 4, 5}), CreateIntTuple({0}),
     keep_dims_true, value_none,
     CreateTensor<float>(ShapeVector{1, 2}, kNumberTypeFloat32, std::vector<float>{3, 4})}});

auto ReduceExtendInferValueTestCase_SumExt = testing::ValuesIn(
  {ReduceExtendInferValueParams{
     CreateTensor<float>(ShapeVector{2, 2}, kNumberTypeFloat32, std::vector<float>{2, 3, 4, 5}), CreateIntTuple({0}),
     keep_dims_false, dtype_float64,
     CreateTensor<double>(ShapeVector{2}, kNumberTypeFloat64, std::vector<double>{6, 8})},
   ReduceExtendInferValueParams{
     CreateTensor<float>(ShapeVector{2, 2}, kNumberTypeFloat32, std::vector<float>{2, 3, 4, 5}), value_none,
     keep_dims_false, dtype_float64, CreateTensor<double>(ShapeVector{}, kNumberTypeFloat64, std::vector<double>{14})},
   ReduceExtendInferValueParams{
     CreateTensor<float>(ShapeVector{2, 2}, kNumberTypeFloat32, std::vector<float>{2, 3, 4, 5}), CreateIntTuple({0}),
     keep_dims_true, value_none,
     CreateTensor<float>(ShapeVector{1, 2}, kNumberTypeFloat32, std::vector<float>{6, 8})}});

auto ReduceExtendInferValueTestCase_ProdExt = testing::ValuesIn(
  {ReduceExtendInferValueParams{
     CreateTensor<float>(ShapeVector{2, 2}, kNumberTypeFloat32, std::vector<float>{2, 3, 4, 5}), CreateIntTuple({0}),
     keep_dims_false, dtype_float64,
     CreateTensor<double>(ShapeVector{2}, kNumberTypeFloat64, std::vector<double>{8, 15})},
   ReduceExtendInferValueParams{
     CreateTensor<float>(ShapeVector{2, 2}, kNumberTypeFloat32, std::vector<float>{2, 3, 4, 5}), value_none,
     keep_dims_false, dtype_float64, CreateTensor<double>(ShapeVector{}, kNumberTypeFloat64, std::vector<double>{120})},
   ReduceExtendInferValueParams{
     CreateTensor<float>(ShapeVector{2, 2}, kNumberTypeFloat32, std::vector<float>{2, 3, 4, 5}), CreateIntTuple({0}),
     keep_dims_true, value_none,
     CreateTensor<float>(ShapeVector{1, 2}, kNumberTypeFloat32, std::vector<float>{8, 15})}});

INSTANTIATE_TEST_CASE_P(TestMeanExtInferValueGroup, TestReduceExtendInferValue,
                        testing::Combine(testing::ValuesIn({kNameMeanExt}), ReduceExtendInferValueTestCase_MeanExt));
INSTANTIATE_TEST_CASE_P(TestSumExtInferValueGroup, TestReduceExtendInferValue,
                        testing::Combine(testing::ValuesIn({kNameSumExt}), ReduceExtendInferValueTestCase_SumExt));
INSTANTIATE_TEST_CASE_P(TestProdExtInferValueGroup, TestReduceExtendInferValue,
                        testing::Combine(testing::ValuesIn({kNameProdExt}), ReduceExtendInferValueTestCase_ProdExt));
}  // namespace ops
}  // namespace mindspore
