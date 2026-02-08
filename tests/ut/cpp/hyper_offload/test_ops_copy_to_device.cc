/**
Copyright 2023 Huawei Technologies Co., Ltd
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
#include "ops/test_value_utils.h"
#include "infer/ops_func_impl/copy_to_device.h"
namespace mindspore{
namespace ops{

struct CopyToDeviceParams {
    ShapeVector x_shape;
    TypePtr x_type;
    ShapeVector output_shape;   
    TypePtr output_type;
};

class TestCopyToDevice : public TestOps, public testing::WithParamInterface<CopyToDeviceParams> {};

TEST_P(TestCopyToDevice, dyn_shape) {
    const auto &param = GetParam();
    auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);

    auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
    auto expect_type = std::make_shared<TensorType>(param.output_type);

    CopyToDeviceFuncImpl copy_to_device_func_impl;
    auto prim = std::make_shared<Primitive>("CopyToDevice");

    auto real_dtype = copy_to_device_func_impl.InferType(prim, {x});
    ASSERT_TRUE(*real_dtype == *expect_type);
    auto real_shape = copy_to_device_func_impl.InferShape(prim, {x});
    ASSERT_TRUE(*real_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(
    TestCopyToDevice,TestCopyToDevice,
    testing::Values(CopyToDeviceParams{{3,4,5},kFloat32,{3,4,5},kFloat32},
    CopyToDeviceParams{{2,3,4},kFloat16,{2,3,4},kFloat16},
    CopyToDeviceParams{{-3,4,5},kBFloat16,{-3,4,5},kBFloat16},
    CopyToDeviceParams{{1,2,3,4,5},kFloat32,{1,2,3,4,5},kFloat32},
    CopyToDeviceParams{{-1,-2,-3,4,5},kFloat16,{-1,-2,-3,4,5},kFloat16},
    CopyToDeviceParams{{3,4},kBFloat16,{3,4},kBFloat16}));
}//namespace ops
}//namespace mindspore