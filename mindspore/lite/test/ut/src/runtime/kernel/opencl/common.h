/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TEST_UT_SRC_RUNTIME_KERNEL_OPENCL_COMMON_H_
#define MINDSPORE_LITE_TEST_UT_SRC_RUNTIME_KERNEL_OPENCL_COMMON_H_

#include <string>
#include <iostream>
#include <vector>
#include <tuple>
#include <map>
#include <memory>
#include "nnacl/op_base.h"
#include "ir/dtype/type_id.h"
#include "src/tensor.h"
#include "src/common/file_utils.h"
#include "common/common_test.h"

using Tensor = mindspore::lite::Tensor;
using ArgsTuple = std::tuple<std::vector<int>, void *, Tensor::Category>;
using ArgsTupleOut = std::tuple<std::vector<int>, void *>;
using ArgsTupleWithDtype = std::tuple<std::vector<int>, void *, Tensor::Category, mindspore::TypeId>;
constexpr Tensor::Category VAR = Tensor::VAR;
constexpr Tensor::Category CONST_TENSOR = Tensor::Category::CONST_TENSOR;
constexpr Tensor::Category CONST_SCALAR = Tensor::Category::CONST_SCALAR;

namespace mindspore::lite::opencl::test {

template <typename T>
void CompareOutput(void *output, void *expect, size_t elem_num, T atol, float rtol = 1e-9, bool print_data = false) {
  T *output_data = reinterpret_cast<T *>(output);
  T *expect_data = reinterpret_cast<T *>(expect);

  if (print_data) {
    for (int i = 0; i < elem_num; ++i) {
      printf("%d: expect=%.3f output=%.3f\n", i, expect_data[i], output_data[i]);
    }
  }

  int mismatch_num = 0;
  int first_err_idx = -1;
  for (int i = 0; i < elem_num; ++i) {
    auto delta = static_cast<float>(std::fabs(output_data[i] - expect_data[i]));
    auto tolerance = static_cast<float>(atol + rtol * std::fabs(expect_data[i]));
    if (delta > tolerance) {
      mismatch_num++;
      if (first_err_idx == -1) {
        first_err_idx = i;
      }
    }
  }
  if (mismatch_num > 0) {
    printf("(mismatch %4.1f%%)\n", 100 * static_cast<float>(mismatch_num) / elem_num);
    printf("Not equal to tolerance atol=%.0e, rtol=%.0e\n", atol, rtol);
    printf("first error at idx=%d expect=%.1f output=%.1f\n", first_err_idx, expect_data[first_err_idx],
           output_data[first_err_idx]);
    FAIL();
  }
}

template <typename T>
void CompareOutput(Tensor *output_tensor, const std::string &file_path, float atol, float rtol = 1e-9) {
  size_t output_size;
  auto expect_data = lite::ReadFile(file_path.c_str(), &output_size);
  CompareOutput<T>(output_tensor->data_c(), expect_data, output_tensor->ElementsNum(), atol, rtol);
}

template <typename T>
T *CreateParameter(schema::PrimitiveType type) {
  auto *param = static_cast<T *>(malloc(sizeof(T)));
  if (param == nullptr) {
    MS_LOG(ERROR) << std::string("create Parameter failed for ") + schema::EnumNamePrimitiveType(type) << std::endl;
    return nullptr;
  }
  memset(param, 0, sizeof(T));
  (reinterpret_cast<OpParameter *>(param))->type_ = type;
  return param;
}

void TestMain(const std::vector<ArgsTupleWithDtype> &input_infos, const std::vector<ArgsTupleOut> &output_info,
              OpParameter *op_parameter, bool fp16_enable = false, float atol = 1e-9, float rtol = 1e-9,
              bool print_output = false);

void TestMain(const std::vector<ArgsTuple> &input_infos, const std::vector<ArgsTupleOut> &output_info,
              OpParameter *op_parameter, bool fp16_enable = false, float atol = 1e-9, float rtol = 1e-9,
              bool print_output = false);

void TestMain(const std::vector<ArgsTupleWithDtype> &input_infos, std::tuple<std::vector<int>, float *> output_info,
              OpParameter *op_parameter, bool fp16_enable = false, float atol = 1e-9, float rtol = 1e-9,
              bool print_output = false);
void TestMain(const std::vector<ArgsTuple> &input_infos, std::tuple<std::vector<int>, float *> output_info,
              OpParameter *op_parameter, bool fp16_enable = false, float atol = 1e-9, float rtol = 1e-9,
              bool print_output = false);

}  // namespace mindspore::lite::opencl::test

#endif  // MINDSPORE_LITE_TEST_UT_SRC_RUNTIME_KERNEL_OPENCL_COMMON_H_
