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
#include <vector>

#include "common/common_test.h"
#include "ir/value.h"
#include "pybind_api/ir/primitive_py.h"
#include "pipeline/jit/parse/parse_base.h"
#include "include/common/utils/python_adapter.h"
#include "frontend/operator/ops.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace prim {

class TestOps : public UT::Common {
 public:
  TestOps() {}
  virtual void SetUp() {}
};

// Arithmetic
TEST_F(TestOps, ScalarAddTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarAdd);
  ASSERT_EQ(prim->name(), kPrimScalarAdd->name());
}

TEST_F(TestOps, ScalarSubTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarSub);
  ASSERT_EQ(prim->name(), kPrimScalarSub->name());
}

TEST_F(TestOps, ScalarMulTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarMul);
  ASSERT_EQ(prim->name(), kPrimScalarMul->name());
}

TEST_F(TestOps, ScalarDivTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarDiv);
  ASSERT_EQ(prim->name(), kPrimScalarDiv->name());
}

TEST_F(TestOps, ScalarModTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarMod);
  ASSERT_EQ(prim->name(), kPrimScalarMod->name());
}

TEST_F(TestOps, ScalarPowTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarPow);
  ASSERT_EQ(prim->name(), kPrimScalarPow->name());
}

TEST_F(TestOps, ScalarTruncTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarTrunc);
  ASSERT_EQ(prim->name(), kPrimScalarTrunc->name());
}

TEST_F(TestOps, ScalarFloorTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarFloor);
  ASSERT_EQ(prim->name(), kPrimScalarFloor->name());
}

TEST_F(TestOps, ScalarUaddTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarUadd);
  ASSERT_EQ(prim->name(), kPrimScalarUadd->name());
}

TEST_F(TestOps, ScalarUsubTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarUsub);
  ASSERT_EQ(prim->name(), kPrimScalarUsub->name());
}

TEST_F(TestOps, ScalarExpTest) {
  auto prim = std::make_shared<Primitive>("scalar_exp");
  ASSERT_EQ(prim->name(), kPrimScalarExp->name());
}

TEST_F(TestOps, ScalarLogTest) {
  auto prim = std::make_shared<Primitive>("scalar_log");
  ASSERT_EQ(prim->name(), kPrimScalarLog->name());
}

TEST_F(TestOps, ScalarSinTest) {
  auto prim = std::make_shared<Primitive>("scalar_sin");
  ASSERT_EQ(prim->name(), kPrimScalarSin->name());
}

TEST_F(TestOps, ScalarCosTest) {
  auto prim = std::make_shared<Primitive>("scalar_cos");
  ASSERT_EQ(prim->name(), kPrimScalarCos->name());
}

TEST_F(TestOps, ScalarTanTest) {
  auto prim = std::make_shared<Primitive>("scalar_tan");
  ASSERT_EQ(prim->name(), kPrimScalarTan->name());
}

// Comparisons
TEST_F(TestOps, ScalarEqTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarEq);
  ASSERT_EQ(prim->name(), kPrimScalarEq->name());
}

TEST_F(TestOps, ScalarLtTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarLt);
  ASSERT_EQ(prim->name(), kPrimScalarLt->name());
}

TEST_F(TestOps, ScalarGtTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarGt);
  ASSERT_EQ(prim->name(), kPrimScalarGt->name());
}

TEST_F(TestOps, ScalarNeTest) {
  auto prim = std::make_shared<Primitive>("scalar_ne");
  ASSERT_EQ(prim->name(), kPrimScalarNe->name());
}

TEST_F(TestOps, ScalarLeTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarLe);
  ASSERT_EQ(prim->name(), kPrimScalarLe->name());
}

TEST_F(TestOps, ScalarGeTest) {
  auto prim = std::make_shared<Primitive>(prim::kScalarGe);
  ASSERT_EQ(prim->name(), kPrimScalarGe->name());
}

TEST_F(TestOps, BoolNotTest) {
  auto prim = std::make_shared<Primitive>("bool_not");
  ASSERT_EQ(prim->name(), kPrimBoolNot->name());
}

TEST_F(TestOps, BoolAndTest) {
  auto prim = std::make_shared<Primitive>("bool_and");
  ASSERT_EQ(prim->name(), kPrimBoolAnd->name());
}

TEST_F(TestOps, BoolOrTest) {
  auto prim = std::make_shared<Primitive>("bool_or");
  ASSERT_EQ(prim->name(), kPrimBoolOr->name());
}

TEST_F(TestOps, BoolEqTest) {
  auto prim = std::make_shared<Primitive>("bool_eq");
  ASSERT_EQ(prim->name(), kPrimBoolEq->name());
}

// Type introspection
TEST_F(TestOps, TypeOfTest) {
  auto prim = std::make_shared<Primitive>("typeof");
  ASSERT_EQ(prim->name(), kPrimTypeOf->name());
}

TEST_F(TestOps, HasTypeTest) {
  auto prim = std::make_shared<Primitive>("hastype");
  ASSERT_EQ(prim->name(), kPrimHasType->name());
}

// Data structures
TEST_F(TestOps, MakeTupleTest) {
  auto prim = std::make_shared<Primitive>("MakeTuple");
  ASSERT_EQ(prim->name(), kPrimMakeTuple->name());
}

TEST_F(TestOps, MakeListTest) {
  auto prim = std::make_shared<Primitive>("make_list");
  ASSERT_EQ(prim->name(), kPrimMakeList->name());
}

TEST_F(TestOps, TupleGetItemTest) {
  auto prim = std::make_shared<Primitive>(kTupleGetItem);
  ASSERT_EQ(prim->name(), kPrimTupleGetItem->name());
}

TEST_F(TestOps, ListGetItemTest) {
  auto prim = std::make_shared<Primitive>("list_getitem");
  ASSERT_EQ(prim->name(), kPrimListGetItem->name());
}

TEST_F(TestOps, ArrayGetItemTest) {
  auto prim = std::make_shared<Primitive>("array_getitem");
  ASSERT_EQ(prim->name(), kPrimArrayGetItem->name());
}

TEST_F(TestOps, TupleSetItemTest) {
  auto prim = std::make_shared<Primitive>("tuple_setitem");
  ASSERT_EQ(prim->name(), kPrimTupleSetItem->name());
}

TEST_F(TestOps, ListSetItemTest) {
  auto prim = std::make_shared<Primitive>("list_setitem");
  ASSERT_EQ(prim->name(), kPrimListSetItem->name());
}

TEST_F(TestOps, ArraySetItemTest) {
  auto prim = std::make_shared<Primitive>("array_setitem");
  ASSERT_EQ(prim->name(), kPrimArraySetItem->name());
}

TEST_F(TestOps, ListAppendTest) {
  auto prim = std::make_shared<Primitive>("ListAppend");
  ASSERT_EQ(prim->name(), kPrimListAppend->name());
}

/// Feature: Generate primitive.
/// Description: Generate primitive.
/// Expectation: No exception.
TEST_F(TestOps, SequenceAddTest) {
  auto prim = std::make_shared<Primitive>("SequenceAdd");
  ASSERT_EQ(prim->name(), kPrimSequenceAdd->name());
}

/// Feature: Generate primitive.
/// Description: Generate primitive.
/// Expectation: No exception.
TEST_F(TestOps, SequenceCountTest) {
  auto prim = std::make_shared<Primitive>("SequenceCount");
  ASSERT_EQ(prim->name(), kPrimSequenceCount->name());
}

TEST_F(TestOps, GetAttrTest) {
  auto prim = std::make_shared<Primitive>("getattr");
  ASSERT_EQ(prim->name(), kPrimGetAttr->name());
}

/// Feature: Generate primitive.
/// Description: Generate primitive.
/// Expectation: No exception.
TEST_F(TestOps, SequenceLenTest) {
  auto prim = std::make_shared<Primitive>("sequence_len");
  ASSERT_EQ(prim->name(), kPrimSequenceLen->name());
}

TEST_F(TestOps, ArrayLenTest) {
  auto prim = std::make_shared<Primitive>("array_len");
  ASSERT_EQ(prim->name(), kPrimArrayLen->name());
}

TEST_F(TestOps, ListReduceTest) {
  auto prim = std::make_shared<Primitive>("list_reduce");
  ASSERT_EQ(prim->name(), kPrimListReduce->name());
}

// Arrays
TEST_F(TestOps, ArrayToScalarTest) {
  auto prim = std::make_shared<Primitive>("array_to_scalar");
  ASSERT_EQ(prim->name(), kPrimArrayToScalar->name());
}

TEST_F(TestOps, BroadCastShapeTest) {
  auto prim = std::make_shared<Primitive>("broadcast_shape");
  ASSERT_EQ(prim->name(), kPrimBroadcastShape->name());
}

TEST_F(TestOps, ArrayMapTest) {
  auto prim = std::make_shared<Primitive>("array_map");
  ASSERT_EQ(prim->name(), kPrimArrayMap->name());
}

TEST_F(TestOps, ArrayReduceTest) {
  auto prim = std::make_shared<Primitive>("array_reduce");
  ASSERT_EQ(prim->name(), kPrimArrayReduce->name());
}

TEST_F(TestOps, DistributeTest) {
  auto prim = std::make_shared<Primitive>("distribute");
  ASSERT_EQ(prim->name(), kPrimDistribute->name());
}

TEST_F(TestOps, TransposeTest) {
  auto prim = std::make_shared<Primitive>("Transpose");
  ASSERT_EQ(prim->name(), kPrimTranspose->name());
}

TEST_F(TestOps, Im2ColTest) {
  auto prim = std::make_shared<Primitive>("Im2Col");
  ASSERT_EQ(prim->name(), kPrimIm2Col->name());
}

TEST_F(TestOps, Col2ImTest) {
  auto prim = std::make_shared<Primitive>("Col2Im");
  ASSERT_EQ(prim->name(), kPrimCol2Im->name());
}

TEST_F(TestOps, Im2ColV1Test) {
  auto prim = std::make_shared<Primitive>("im2col_v1");
  ASSERT_EQ(prim->name(), kPrimIm2ColV1->name());
}

TEST_F(TestOps, Col2ImV1Test) {
  auto prim = std::make_shared<Primitive>("col2im_v1");
  ASSERT_EQ(prim->name(), kPrimCol2ImV1->name());
}

// Statements
TEST_F(TestOps, SwitchTest) {
  auto prim = std::make_shared<Primitive>("Switch");
  ASSERT_EQ(prim->name(), kPrimSwitch->name());
}

TEST_F(TestOps, ReturnTest) {
  auto prim = std::make_shared<Primitive>("Return");
  ASSERT_EQ(prim->name(), kPrimReturn->name());
}

// Miscellaneous

TEST_F(TestOps, IdentityTest) {
  auto prim = std::make_shared<Primitive>("identity");
  ASSERT_EQ(prim->name(), kPrimIdentity->name());
}

TEST_F(TestOps, ResolveTest) {
  auto prim = std::make_shared<Primitive>("resolve");
  ASSERT_EQ(prim->name(), kPrimResolve->name());
}

TEST_F(TestOps, PartialTest) {
  auto prim = std::make_shared<Primitive>("Partial");
  ASSERT_EQ(prim->name(), kPrimPartial->name());
}

TEST_F(TestOps, JTest) {
  auto prim = std::make_shared<Primitive>("J");
  ASSERT_EQ(prim->name(), kPrimJ->name());
}

TEST_F(TestOps, EmbedTest) {
  auto prim = std::make_shared<Primitive>("embed");
  ASSERT_EQ(prim->name(), kPrimEmbed->name());
}

/// Feature: Check primitive name equivalence
/// Description: EnvironSet primitive name equivalence
/// Expectation: Equal
TEST_F(TestOps, EnvironSetTest) {
  auto prim = std::make_shared<Primitive>("EnvironSet");
  ASSERT_EQ(prim->name(), kPrimEnvironSet->name());
}

/// Feature: Check primitive name equivalence
/// Description: EnvironGet primitive name equivalence
/// Expectation: Equal
TEST_F(TestOps, EnvironGetTest) {
  auto prim = std::make_shared<Primitive>("EnvironGet");
  ASSERT_EQ(prim->name(), kPrimEnvironGet->name());
}

/// Feature: Check primitive name equivalence
/// Description: EnvironAdd primitive name equivalence
/// Expectation: Equal
TEST_F(TestOps, EnvironAddTest) {
  auto prim = std::make_shared<Primitive>("EnvironAdd");
  ASSERT_EQ(prim->name(), kPrimEnvironAdd->name());
}

// Neural Network
TEST_F(TestOps, Conv2dTest) {
  auto prim = std::make_shared<Primitive>("Conv2D");
  ASSERT_EQ(prim->name(), kPrimConv2D->name());
}

TEST_F(TestOps, Conv2dAttrTest) {
  Primitive prim("Conv2D");
  prim.SetAttrs({
    {"stride", MakeValue(static_cast<int64_t>(3))},
    {"pad", MakeValue(static_cast<int64_t>(1))},
  });
  ASSERT_EQ(prim.name(), kPrimConv2D->name());

  Int64Imm stride(3);
  Int64Imm pad(1);
  ASSERT_EQ(*prim.GetAttr("stride"), stride);
  ASSERT_EQ(*prim.GetAttr("pad"), pad);
}

TEST_F(TestOps, CustomOpAttrTest) {
  Primitive prim("CustomOp", true, kPrimTypePyInfer);
  prim.SetAttrs({
    {"attr1", MakeValue(static_cast<int64_t>(3))},
    {"attr2", MakeValue(static_cast<int64_t>(1))},
  });
  ASSERT_EQ(prim.name(), std::string("CustomOp"));
  ASSERT_EQ(prim.prim_type(), kPrimTypePyInfer);

  auto attrs = prim.attrs();
  for (auto attr : attrs) {
    std::string prim_name = attr.first;
    auto prim_value = attr.second;
    std::cout << prim_name << std::endl;
    std::cout << prim_value << std::endl;
  }
}

TEST_F(TestOps, Conv2dBackpropInputTest) {
  auto prim = std::make_shared<Primitive>("Conv2DBackpropInput");
  ASSERT_EQ(prim->name(), kPrimConv2DBackpropInput->name());
}

TEST_F(TestOps, Conv2dBackpropFilterTest) {
  auto prim = std::make_shared<Primitive>("Conv2DBackpropFilter");
  ASSERT_EQ(prim->name(), kPrimConv2DBackpropFilter->name());
}

TEST_F(TestOps, ReluTest) {
  auto prim = std::make_shared<Primitive>("ReLU");
  ASSERT_EQ(prim->name(), kPrimReLU->name());
}

TEST_F(TestOps, PoolingTest) {
  auto prim = std::make_shared<Primitive>("Pooling");
  ASSERT_EQ(prim->name(), kPrimPooling->name());
}

TEST_F(TestOps, GetConv2DPrimPyTest) {
  auto conv2d_prim = prim::GetPythonOps("conv2d_prim", "gtest_input.pynative");
  ASSERT_TRUE(conv2d_prim);
  PrimitivePyPtr conv2d_ptr = dyn_cast<PrimitivePy>(conv2d_prim);
  ASSERT_TRUE(conv2d_ptr);
  if (nullptr != conv2d_ptr) {
    MS_LOG(INFO) << "Get PrimitivePyPtr: " << conv2d_ptr->name();
    if(!conv2d_ptr->HasComputeFunction()){
      MS_LOG(EXCEPTION) << "" << conv2d_ptr->name() << "'s compute function is not implemented";
    }

    py::object conv2d_pyobj = python_adapter::GetPyFn("gtest_input.pynative", "conv2d_prim");
    py::dict opAttrs = py::getattr(conv2d_pyobj, "attrs");
    mindspore::HashMap<std::string, ValuePtr> attrs{};
    for (auto item : opAttrs) {
      if (!py::isinstance<py::str>(item.first)) {
        MS_LOG(EXCEPTION) << "type error in py dict convert";
      }
      std::string name = py::cast<std::string>(item.first);
      MS_LOG(INFO) << "Attr name: " << name;

      ValuePtr converted_ret;
      parse::ConvertData(py::cast<py::object>(item.second), &converted_ret);
      MS_LOG(INFO) << "Attr value: " << converted_ret->ToString();
      attrs.emplace(name, converted_ret);
    }
  }

  MS_LOG(INFO) << "Finish GetPyFnTest!";
}

}  // namespace prim
}  // namespace mindspore
