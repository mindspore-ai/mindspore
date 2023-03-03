/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "backend/common/graph_kernel/expanders/utils.h"
#include "backend/common/graph_kernel/expanders/custom_op_utils.h"

namespace mindspore::graphkernel::expanders {
class LU : public OpDesc {
 public:
  LU() {}
  ~LU() = default;

 protected:
  std::vector<int64_t> GetIndicesValue(int64_t u_left, int64_t u_right, int64_t v_left, int64_t v_right) {
    std::vector<int64_t> indices_value;
    for (int64_t u = u_left; u < u_right; ++u) {
      for (int64_t v = v_left; v < v_right; ++v) {
        (void)indices_value.emplace_back(u);
        (void)indices_value.emplace_back(v);
      }
    }
    return indices_value;
  }
  std::vector<int32_t> GetEyesValue(int64_t num) {
    std::vector<int32_t> eyes_value;
    for (int64_t i = 0; i < num; ++i) {
      for (int64_t j = 0; j < num; ++j) {
        if (i == j) {
          (void)eyes_value.emplace_back(1);
        } else {
          (void)eyes_value.emplace_back(0);
        }
      }
    }
    return eyes_value;
  }
  std::vector<int32_t> GetRangeValue(int64_t num) {
    std::vector<int32_t> range_value;
    for (int i = 0; i < static_cast<int>(num); ++i) {
      (void)range_value.emplace_back(i);
    }
    return range_value;
  }
  NodePtrList Expand(const NodePtrList &inputs) override {
    auto input_x = inputs[0];
    auto num = input_x->shape[0];
    auto loop_count = static_cast<int64_t>(num / kBlock);
    std::vector<int64_t> strides{1, 1};

    // lu dsl  implementation
    for (int64_t i = 0; i < loop_count; ++i) {
      std::vector<int64_t> begin_1{i * kBlock, i * kBlock};
      std::vector<int64_t> end_1{(i + 1) * kBlock, (i + 1) * kBlock};
      auto stride_1 = gb.StridedSlice(input_x, begin_1, end_1, strides);
      std::string lu_func_type = kFuncType;
      std::string lu_func_name = kLUName;
      // get the name of lu via its attrs
      auto lu_iter = kLUFuncStrMap.find(lu_func_name);
      std::string lu_func_source_str = lu_iter->second;
      size_t lu_inplace_assign_output = 0;
      std::string lu_func_compile_attrs = kLUAttrs;

      auto custom_lu_decomp_result =
        gb.Custom({stride_1}, {stride_1->shape, stride_1->type, stride_1->format}, lu_func_name, lu_func_type,
                  lu_func_source_str, lu_inplace_assign_output, lu_func_compile_attrs);
      ShapeVector ind_shape{kBlock, kBlock, 2};
      std::vector<int64_t> ind_value = GetIndicesValue(i * kBlock, (i + 1) * kBlock, i * kBlock, (i + 1) * kBlock);

      auto ind_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, ind_shape, &ind_value[0], kNumberTypeInt64);
      auto first_indices = gb.Value(ind_tensor);
      input_x = gb.Emit("ScatterNdUpdate", {input_x, first_indices, custom_lu_decomp_result},
                        {{"use_locking", MakeValue(false)}});
      if (i < loop_count - 1) {
        std::vector<int64_t> begin_2{i * kBlock, (i + 1) * kBlock};
        std::vector<int64_t> end_2{(i + 1) * kBlock, num};
        auto stride_2 = gb.StridedSlice(input_x, begin_1, end_1, strides);
        auto stride_3 = gb.StridedSlice(input_x, begin_2, end_2, strides);
        std::string trsmL_off_diag_func_type = kFuncType;
        std::string trsmL_off_diag_func_name = kTrsmName;
        // get the name of trsmL_off_diag via its attrs
        trsmL_off_diag_func_name = trsmL_off_diag_func_name + "L_N_U";
        auto trsmL_off_diag_iter = kTrsmFuncStrMap.find(trsmL_off_diag_func_name);
        std::string trsmL_off_diag_source_str = trsmL_off_diag_iter->second;
        size_t trsmL_off_diag_inplace_assign_output = 1;
        std::string trsmL_off_diag_compile_attrs = kTrsmLAttrs;
        auto custom_trsmL_off_diag_result =
          gb.Custom({stride_2, stride_3}, {stride_3->shape, stride_3->type, stride_3->format}, trsmL_off_diag_func_name,
                    trsmL_off_diag_func_type, trsmL_off_diag_source_str, trsmL_off_diag_inplace_assign_output,
                    trsmL_off_diag_compile_attrs);
        ShapeVector sec_indices_shape{kBlock, num - (i + 1) * kBlock, 2};
        std::vector<int64_t> sec_indicse_value = GetIndicesValue(i * kBlock, (i + 1) * kBlock, (i + 1) * kBlock, num);

        auto sec_indices_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, sec_indices_shape,
                                                                   &sec_indicse_value[0], kNumberTypeInt64);
        auto sec_indices = gb.Value(sec_indices_tensor);
        input_x = gb.Emit("ScatterNdUpdate", {input_x, sec_indices, custom_trsmL_off_diag_result},
                          {{"use_locking", MakeValue(false)}});

        std::vector<int64_t> begin_3{(i + 1) * kBlock, i * kBlock};
        std::vector<int64_t> end_3{num, (i + 1) * kBlock};
        auto stride_4 = gb.StridedSlice(input_x, begin_1, end_1, strides);
        auto stride_5 = gb.StridedSlice(input_x, begin_3, end_3, strides);
        std::string trsmUT_func_type = kFuncType;
        std::string trsmUT_func_name = kTrsmName;
        // get the name of trsmUT via its attrs
        trsmUT_func_name = trsmUT_func_name + "U_T";
        auto trsmUT_iter = kTrsmFuncStrMap.find(trsmUT_func_name);
        std::string trsmUT_source_str = trsmUT_iter->second;
        size_t trsmUT_inplace_assign_output = 1;
        std::string trsmUT_compile_attrs = kTrsmLAttrs;
        auto custom_trsmUT_result =
          gb.Custom({stride_4, stride_5}, {stride_5->shape, stride_5->type, stride_5->format}, trsmUT_func_name,
                    trsmUT_func_type, trsmUT_source_str, trsmUT_inplace_assign_output, trsmUT_compile_attrs);
        ShapeVector third_indices_shape{num - (i + 1) * kBlock, kBlock, 2};
        std::vector<int64_t> thi_indices_v = GetIndicesValue((i + 1) * kBlock, num, i * kBlock, (i + 1) * kBlock);

        auto thi_indices_tensor =
          std::make_shared<tensor::Tensor>(kNumberTypeInt64, third_indices_shape, &thi_indices_v[0], kNumberTypeInt64);
        auto thi_indices = gb.Value(thi_indices_tensor);
        input_x =
          gb.Emit("ScatterNdUpdate", {input_x, thi_indices, custom_trsmUT_result}, {{"use_locking", MakeValue(false)}});

        std::vector<int64_t> begin_4{(i + 1) * kBlock, (i + 1) * kBlock};
        std::vector<int64_t> end_4{num, num};
        std::vector<int64_t> begin_5{i * kBlock, (i + 1) * kBlock};
        std::vector<int64_t> end_5{(i + 1) * kBlock, num};
        auto stride_6 = gb.StridedSlice(input_x, begin_4, end_4, strides);
        auto stride_7 = gb.StridedSlice(input_x, begin_3, end_3, strides);
        auto stride_8 = gb.StridedSlice(input_x, begin_5, end_5, strides);
        // on ascend, matmul's inputs must be fp16
        stride_7 = gb.Cast(stride_7, kNumberTypeFloat16);
        stride_8 = gb.Cast(stride_8, kNumberTypeFloat16);
        auto matmul_stride_7_8 = gb.MatMul(stride_7, stride_8);
        matmul_stride_7_8 = gb.Cast(matmul_stride_7_8, kNumberTypeFloat32);
        auto final_update = gb.Sub(stride_6, matmul_stride_7_8);
        ShapeVector final_indices_shape{num - (i + 1) * kBlock, num - (i + 1) * kBlock, 2};
        std::vector<int64_t> final_indicse_value = GetIndicesValue((i + 1) * kBlock, num, (i + 1) * kBlock, num);

        auto final_indices_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, final_indices_shape,
                                                                     &final_indicse_value[0], kNumberTypeInt64);
        auto f_indices = gb.Value(final_indices_tensor);
        input_x = gb.Emit("ScatterNdUpdate", {input_x, f_indices, final_update}, {{"use_locking", MakeValue(false)}});
      }
    }
    ShapeVector eyes_shape{num, num};
    std::vector<int32_t> eyes_value = GetEyesValue(num);
    auto eyes_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, eyes_shape, &eyes_value[0], kNumberTypeInt32);
    auto eyes = gb.Value(eyes_tensor);
    auto eyes_cnode = gb.Reshape(eyes, eyes_shape);

    ShapeVector r_shape{num};
    std::vector<int32_t> r_value = GetRangeValue(num);
    auto range_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, r_shape, &r_value[0], kNumberTypeInt32);
    auto range = gb.Value(range_tensor);
    auto range_cnode = gb.Reshape(range, r_shape);
    return {input_x, range_cnode, eyes_cnode};
  }
};
EXPANDER_OP_DESC_REGISTER("LU", LU);
}  // namespace mindspore::graphkernel::expanders
