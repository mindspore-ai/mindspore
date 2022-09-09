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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_CONVERTER_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_CONVERTER_UTILS_H_
#include <string>
#include <memory>
#include <vector>
#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif
#include "schema/ops_generated.h"
#include "include/graph/tensor.h"
#include "include/graph/op/array_defs.h"
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "include/graph/op/all_ops.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
enum NCHW_SHAPE { NCHW_INVALID = -1, NCHW_N = 0, NCHW_C = 1, NCHW_H = 2, NCHW_W = 3 };
enum NHWC_SHAPE { NHWC_N = 0, NHWC_H = 1, NHWC_W = 2, NHWC_C = 3 };
inline const std::vector<int> NHWC2NCHW_PERM = {0, 3, 1, 2};
inline const std::vector<int> NCHW2NHWC_PERM = {0, 2, 3, 1};

enum NPU_ACTIVATION_MODE {
  ACTIVATION_INVALID = -1,
  SIGMOID = 0,
  RELU = 1,
  TANH = 2,
  CLIPPED_RELU = 3,
  ELU = 4,
  P_RELU = 5,
  ABS = 6,
  RELU1 = 7,
  SOFTSIGN = 8,
  SOFTPLUS = 9,
  HARD_SIGMOID = 10,
  THRESHOLD_RELU = 11,
  SELU = 12,
  LINEAR = 13,
  RELU6 = 14,
  GELU = 15,
};

enum PAD {
  PAD_UP = 0,
  PAD_DOWN = 1,
  PAD_LEFT = 2,
  PAD_RIGHT = 3,
};

enum NPU_PAD_MODE {
  PAD_VALID = 5,
  PAD_SAME = 6,
};

#ifdef ENABLE_ARM
void Float32ToFloat16(const float *__restrict input, float16_t *__restrict output, int number);

void Float16ToFloat32(const float16_t *__restrict input, float *__restrict output, int number);
#endif

std::shared_ptr<ge::Tensor> ConverterToNPUTensor(mindspore::MSTensor src, bool is_expand_4d = false);

hiai::op::Data *ConverterToNPUData(const mindspore::MSTensor &src, const std::string &name);

ge::Format ConverterToNPUFormat(schema::Format format);

ge::DataType ConverterToNPUDataType(DataType type_id);

ge::Shape ConverterToNPUShape(const std::vector<int64_t> &src_shape, bool is_expand_4d = false);

int ConverterToNPUEltwiseMode(schema::EltwiseMode mode);

int ConverterToNPUActivationMode(schema::ActivationType type);

int TransFormAxis(int axis);

template <typename T>
hiai::op::Const *GetNPUConst(const uint8_t *const_data, const std::vector<int64_t> &shape, const ge::DataType data_type,
                             std::string name = "const", bool is_expand_4d = false) {
  MS_CHECK_TRUE_MSG(const_data != nullptr, nullptr, "Const data can not be nullptr.");
  int element_num = 1;
  if (!shape.empty()) {
    for (size_t i = 0; i < shape.size(); i++) {
      MS_CHECK_GT(shape.at(i), 0, nullptr);
      MS_CHECK_INT_MUL_NOT_OVERFLOW(element_num, shape.at(i), nullptr);
      element_num *= shape.at(i);
    }
  }
  ge::TensorDesc const_tensor_desc(ConverterToNPUShape(shape, is_expand_4d), ge::FORMAT_NCHW, data_type);
  ge::TensorPtr const_tensor = std::make_shared<hiai::Tensor>(const_tensor_desc);
  const_tensor->SetData(const_data, element_num * sizeof(T));
  auto const_op = new (std::nothrow) hiai::op::Const(name);
  if (const_op == nullptr) {
    MS_LOG(ERROR) << "New Const op failed.";
    return const_op;
  }
  const_op->set_attr_value(const_tensor);
  return const_op;
}
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_CONVERTER_UTILS_H_
