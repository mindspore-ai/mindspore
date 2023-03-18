/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <memory>

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "frontend/operator/ops.h"
#include "abstract/utils.h"

namespace mindspore {
namespace abstract {

class TestData : public UT::Common {
 public:
  void SetUp();
  void TearDown();
};

void TestData::SetUp() { UT::InitPythonPath(); }

void TestData::TearDown() {
  // destroy resource
}

TEST_F(TestData, test_build_value) {
  // assert build_value(S(1)) == 1
  AbstractScalar s1 = AbstractScalar(static_cast<int64_t>(1));
  ASSERT_EQ(1, s1.BuildValue()->cast<Int64ImmPtr>()->value());
  // assert build_value(S(t=ty.Int[64]), default=ANY) is ANY
  s1 = AbstractScalar(kValueAny, kInt64);
  ASSERT_TRUE(s1.BuildValue()->isa<ValueAny>());
  ASSERT_TRUE(s1.BuildValue()->isa<ValueAny>());

  // assert build_value(T([S(1), S(2)])) == (1, 2)
  AbstractBasePtr base1 = std::make_shared<AbstractScalar>(static_cast<int64_t>(1));
  AbstractBasePtr base2 = std::make_shared<AbstractScalar>(static_cast<int64_t>(2));
  AbstractBasePtrList base_list = {base1, base2};
  AbstractTuple t1 = AbstractTuple(base_list);

  std::vector<ValuePtr> value_list = {MakeValue(static_cast<int64_t>(1)), MakeValue(static_cast<int64_t>(2))};
  auto tup = t1.BuildValue()->cast<ValueTuplePtr>()->value();

  ASSERT_TRUE(tup.size() == value_list.size());
  for (int i = 0; i < value_list.size(); i++) {
    ASSERT_EQ(*tup[i], *value_list[i]);
  }

  // BuildValue(AbstractFunction) should return kValueAny.
  AbstractBasePtr abs_f1 = FromValue(prim::kPrimReturn, false);
  ValuePtr abs_f1_built = abs_f1->BuildValue();
  ASSERT_EQ(abs_f1_built, prim::kPrimReturn);

  FuncGraphPtr fg1 = std::make_shared<FuncGraph>();
  AbstractBasePtr abs_fg1 = FromValue(fg1, false);
  ValuePtr abs_fg1_built = abs_fg1->BuildValue();
  ASSERT_EQ(abs_fg1_built, kValueAny);

  // BuildValue(Tuple(AbstractFunction)) should return kValueAny;
  AbstractBasePtr abs_f2 = FromValue(prim::kPrimScalarAdd, false);
  AbstractBasePtr abs_func_tuple = std::make_shared<AbstractTuple>(AbstractBasePtrList({abs_f1, abs_f2}));
  ValuePtr func_tuple_built = abs_func_tuple->BuildValue();
  ASSERT_EQ(*func_tuple_built, ValueTuple(std::vector<ValuePtr>{prim::kPrimReturn, prim::kPrimScalarAdd}));

  // BuildValue(List(AbstractFunction)) should return kValueAny;
  AbstractBasePtr abs_func_list = std::make_shared<AbstractList>(AbstractBasePtrList({abs_f1, abs_f2}));
  ValuePtr func_list_built = abs_func_list->BuildValue();
  ASSERT_EQ(*func_list_built, ValueList(std::vector<ValuePtr>{prim::kPrimReturn, prim::kPrimScalarAdd}));

  // BuildValue(Tuple(AnyAbstractBase, AbstractFunction)) should return kValueAny
  abs_func_tuple = std::make_shared<AbstractTuple>(AbstractBasePtrList({base1, abs_f2}));
  func_tuple_built = abs_func_tuple->BuildValue();
  ASSERT_EQ(*func_tuple_built, ValueTuple(std::vector<ValuePtr>{std::make_shared<Int64Imm>(1), prim::kPrimScalarAdd}));
}

TEST_F(TestData, test_build_type) {
  AbstractBasePtr s1 = FromValue(static_cast<int64_t>(1), false);
  AbstractBasePtr s2 = FromValue(static_cast<int64_t>(2), false);
  ASSERT_TRUE(Int(64) == *s1->BuildType());

  AbstractFunctionPtr f1 = std::make_shared<PrimitiveAbstractClosure>(nullptr, nullptr);
  ASSERT_TRUE(Function() == *f1->BuildType());

  AbstractList l1 = AbstractList({s1, s2});
  ASSERT_TRUE(List({std::make_shared<Int>(64), std::make_shared<Int>(64)}) == *l1.BuildType());
}

TEST_F(TestData, test_build_shape) {
  AbstractBasePtr s1 = FromValue(static_cast<int64_t>(1), false);
  AbstractBasePtr s2 = FromValue(static_cast<int64_t>(2), false);
  ASSERT_TRUE(NoShape() == *s1->BuildShape());

  AbstractFunctionPtr f1 = std::make_shared<PrimitiveAbstractClosure>(nullptr, nullptr);
  ASSERT_TRUE(NoShape() == *f1->BuildShape());

  AbstractList l1 = AbstractList({s1, s2});
  auto lshape = l1.BuildShape();
  ASSERT_TRUE(lshape);

  std::vector<int64_t> weight1_dims = {2, 20, 5, 5};
  std::vector<int64_t> weight2_dims = {2, 2, 5, 5};
  tensor::TensorPtr weight1 = std::make_shared<tensor::Tensor>(kNumberTypeInt64, weight1_dims);
  tensor::TensorPtr weight2 = std::make_shared<tensor::Tensor>(kNumberTypeInt64, weight2_dims);

  AbstractBasePtr abstract_weight1 = FromValue(weight1, true);
  AbstractBasePtr abstract_weight2 = FromValue(weight2, true);
  ShapePtr shape_weight = dyn_cast<Shape>(abstract_weight1->BuildShape());
  ASSERT_TRUE(shape_weight);
  ASSERT_EQ(weight1_dims, shape_weight->shape());

  std::vector<ValuePtr> vec({weight1, weight2});
  AbstractBasePtr abstract_tup = FromValue(vec, true);
  std::shared_ptr<TupleShape> shape_tuple = dyn_cast<TupleShape>(abstract_tup->BuildShape());
  ASSERT_TRUE(shape_tuple);
  const std::vector<BaseShapePtr> &ptr_vec = shape_tuple->shape();
  ASSERT_EQ(ptr_vec.size(), 2);

  ShapePtr shape1 = dyn_cast<Shape>(ptr_vec[0]);
  ASSERT_TRUE(shape1);
  ASSERT_EQ(weight1_dims, shape1->shape());

  ShapePtr shape2 = dyn_cast<Shape>(ptr_vec[1]);
  ASSERT_TRUE(shape2);
  ASSERT_EQ(weight2_dims, shape2->shape());
}

TEST_F(TestData, test_join) {
  int64_t int1 = 1;
  AbstractBasePtr s1 = FromValue(int1, false);
  AbstractBasePtr s2 = s1->Broaden();

  std::vector<AbstractBasePtr> xx = {s1, s2};
  AbstractListPtr l1 = std::make_shared<AbstractList>(xx);
  AbstractListPtr l2 = std::make_shared<AbstractList>(xx);
  l1->Join(l2);
}

TEST_F(TestData, test_broaden) {
  int64_t int1 = 1;
  AbstractBasePtr s1 = FromValue(int1, false);
  AbstractBasePtr s2 = s1->Broaden();
  ASSERT_TRUE(*s1->GetTypeTrack() == *s2->GetTypeTrack());
  ASSERT_TRUE(*s1->GetValueTrack() == *MakeValue(int1));
  ASSERT_TRUE(s2->GetValueTrack()->isa<Int64Imm>());

  AbstractFunctionPtr f1 =
    std::make_shared<FuncGraphAbstractClosure>(std::make_shared<FuncGraph>(), AnalysisContext::DummyContext());
  AbstractBasePtr f2 = f1->Broaden();
  ASSERT_TRUE(f2 == f1);

  AbstractList l1 = AbstractList({s1, s2});
  AbstractBasePtr l2 = l1.Broaden();
  AbstractList *l2_cast = dynamic_cast<AbstractList *>(l2.get());
  ASSERT_TRUE(l2_cast != nullptr);
  AbstractBasePtr csr = AbstractJoin(l2_cast->elements());
  ASSERT_TRUE(csr->GetValueTrack()->isa<Int64Imm>());
}

}  // namespace abstract
}  // namespace mindspore
