/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "infer/ops_func_impl/detach.h"
#include "ops/test_value_utils.h"


namespace mindspore{
namespace ops{

// input and output params
struct DetachParams {
    // input
    ShapeVector x_shape;
    TypePtr x_type;
    ValuePtr sync;
    // output
    ShapeVector output_shape;
    TypePtr output_type;
};

class TestDetach : public TestOps, public testing::WithParamInterface<DetachParams> {};

TEST_P(TestDetach, dyn_shape) {
    // get params
    const auto &param = GetParam();
    auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
    auto sync = param.sync->ToAbstract();

    // build expected struct of outputs (BaseShape, Type)
    auto expect_shape = 
        std::make_shared<abstract::Shape>(param.output_shape);
    auto expect_type =
        std::make_shared<TensorType>(param.output_type);
    DetachFuncImpl detach_func_impl;
    auto prim = std::make_shared<Primitive>("Detach");

    // get real shape & output and compare real VS expected
    auto real_type = detach_func_impl.InferType(prim, {x, sync});
    ASSERT_TRUE(*real_type == *expect_type);
    if (*real_type == *expect_type){
        MS_LOG(ERROR) << "real_type == expect_type";
    }
    
    auto real_shape = detach_func_impl.InferShape(prim, {x, sync});
    ASSERT_TRUE(*real_shape == *expect_shape);
    if (*real_shape == *expect_shape){
        MS_LOG(ERROR) << "real_shape == expect_shape";
    }
}

//  测试用例批量测试参数
INSTANTIATE_TEST_CASE_P(
  TestDetach, TestDetach,
  testing::Values(
    DetachParams{{1}              , kFloat32, CreateScalar<bool>(true), {1},                kFloat32},
    DetachParams{{6}              , kFloat32, CreateScalar<bool>(true), {6},                kFloat32},
    DetachParams{{6, 7}           , kFloat32, CreateScalar<bool>(true), {6, 7},             kFloat32},
    DetachParams{{6, 7, 8}        , kFloat32, CreateScalar<bool>(true), {6, 7, 8},          kFloat32},
    DetachParams{{6, 7, 8, 9}     , kFloat32, CreateScalar<bool>(true), {6, 7, 8, 9},       kFloat32},
    DetachParams{{6, 7, 8, 9, 10} , kFloat32, CreateScalar<bool>(true), {6, 7, 8, 9, 10},   kFloat32},

    DetachParams{{1}              , kFloat32, CreateScalar<bool>(false), {1},                kFloat32},
    DetachParams{{6}              , kFloat32, CreateScalar<bool>(false), {6},                kFloat32},
    DetachParams{{6, 7}           , kFloat32, CreateScalar<bool>(false), {6, 7},             kFloat32},
    DetachParams{{6, 7, 8}        , kFloat32, CreateScalar<bool>(false), {6, 7, 8},          kFloat32},
    DetachParams{{6, 7, 8, 9}     , kFloat32, CreateScalar<bool>(false), {6, 7, 8, 9},       kFloat32},
    DetachParams{{6, 7, 8, 9, 10} , kFloat32, CreateScalar<bool>(false), {6, 7, 8, 9, 10},   kFloat32},

    DetachParams{{6, 7, 8, 9}, kFloat32, CreateScalar<bool>(true), {6, 7, 8, 9}, kFloat32},
    DetachParams{{6, 7, 8, 9}, kFloat32, CreateScalar<bool>(false), {6, 7, 8, 9}, kFloat32},
    DetachParams{{6, 7, 8, 9}, kFloat32, CreateScalar<bool>(true), {6, 7, 8, 9}, kFloat32},
    DetachParams{{6, 7, 8, 9}, kInt16, CreateScalar<bool>(true), {6, 7, 8, 9}, kInt16}
  ));
}
}
