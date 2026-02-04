/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "ir/tensor_new.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "infer/ops_func_impl/arange.h"
#include "ops/utils/general_infer_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace ops {
OP_FUNC_IMPL_TEST_DECLARE(Arange, MultiInputOpParams);

OP_FUNC_IMPL_TEST_CASES(
  Arange,
  testing::Values(
    MultiInputOpParams{
      {{}, {}, {}}, {kInt32, kInt32, kInt32}, {{-1}}, {kInt64}, {CreateScalar<int64_t>(kNumberTypeInt64)}},
    MultiInputOpParams{
      {{}, {}, {}}, {kInt64, kInt64, kInt64}, {{-1}}, {kFloat32}, {CreateScalar<int64_t>(kNumberTypeFloat32)}},
    MultiInputOpParams{{{-1}, {-1}, {-1}}, {kFloat32, kFloat32, kFloat32}, {{-1}}, {kFloat32}, {mindspore::kNone}},
    MultiInputOpParams{{{-2}, {-2}, {-2}}, {kFloat64, kFloat64, kFloat64}, {{-1}}, {kFloat64}, {mindspore::kNone}}));

    template <typename T>
    tensor::TensorPtr CreateArangeTensor(const TypeId &type, const ShapeVector &shape, std::vector<T> value) {
      void *data_ptr = &value[0];
      auto tensor = tensor::from_buffer(type, shape, data_ptr, type);
      return tensor;
    }
    
    struct ArangeInferValueParams {
      ValuePtr start_value;
      ValuePtr end_value;
      ValuePtr step_value;
      AbstractBasePtr dtype_value;
      tensor::TensorPtr out;
    };
    
    static auto arange_dtype_int64 = std::make_shared<Int64Imm>(kNumberTypeInt64)->ToAbstract();

    class TestArangeInferValue : public TestOps, public testing::WithParamInterface<ArangeInferValueParams> {};
    
    TEST_P(TestArangeInferValue, dyn_shape_infer_value) {
      const auto param = GetParam();
      auto start = param.start_value->ToAbstract();
      auto end = param.end_value->ToAbstract();
      auto step = param.step_value->ToAbstract();
      ASSERT_NE(start, nullptr);
      ASSERT_NE(end, nullptr);
      ASSERT_NE(step, nullptr);
      auto input_args = abstract::AbstractBasePtrList{start, end, step, param.dtype_value};
      auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimArange, input_args);
      if (!value_opt.has_value()) {
        MS_LOG(ERROR) << "Log have no infer value implement!";
        ASSERT_TRUE(false);
      }
      auto infer_out = value_opt.value();
      if (infer_out == nullptr) {
        MS_LOG(ERROR) << "Log can not infer value with inputs: " << input_args;
        ASSERT_TRUE(false);
      }
      auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
      ASSERT_NE(infer_tensor, nullptr);
      ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
    }
    
    INSTANTIATE_TEST_CASE_P(
      TestArangeInferValue, TestArangeInferValue,
      testing::Values(ArangeInferValueParams{CreateScalar<int64_t>(1), CreateScalar<int64_t>(5), CreateScalar<int64_t>(1), arange_dtype_int64,
                                             CreateArangeTensor<int64_t>(kNumberTypeInt64, ShapeVector{4}, std::vector<int64_t>{1, 2, 3, 4})}));
    
}  // namespace ops
}  // namespace mindspore
