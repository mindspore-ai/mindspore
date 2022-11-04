/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "checker/reshape_checker.h"
#include <vector>
#include <algorithm>
#include <string>
#include "common/anf_util.h"
#include "common/op_enum.h"
#include "common/fetch_content.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr int kMaxReshapeInputW = 65536;
}  // namespace
bool ReshapeChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  std::vector<int64_t> input_shape;
  if (GetInputShapeFromCNode(op, kInputIndex1, &input_shape) == RET_OK && !input_shape.empty()) {
    int64_t input_w;
    if (GetWidth(input_shape, format, &input_w) != RET_OK) {
      MS_LOG(ERROR) << "get input_w failed";
      return false;
    }

    if (input_shape.size() == kDims4 && input_w > kMaxReshapeInputW) {
      return false;
    }
  }
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }

  if (op->inputs().size() < kInputIndex3) {
    MS_LOG(ERROR) << "There should be at least 2 inputs, but is " << (op->inputs().size() - 1);
    return false;
  }

  DataInfo data_info;
  std::vector<int64_t> shape_data;
  auto shape_ptr = primitive->GetAttr(ops::kShape);
  if (op->inputs().size() > kInputIndex2 && FetchDataFromParameterNode(op, kInputIndex2, &data_info) == lite::RET_OK) {
    if (data_info.data_type_ != static_cast<int>(kNumberTypeInt32)) {
      MS_LOG(ERROR) << "data_type not correct";
      return false;
    }
    auto data = reinterpret_cast<int32_t *>(data_info.data_.data());
    if (data == nullptr) {
      MS_LOG(ERROR) << "data is nullptr. " << op->fullname_with_scope();
      return false;
    }
    int data_size;
    if (GetDataSizeFromTensor(&data_info, &data_size) != RET_OK) {
      MS_LOG(ERROR) << "get data size from tensor failed.";
      return false;
    }
    (void)std::transform(data, data + data_size, std::back_inserter(shape_data),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
  } else if (shape_ptr != nullptr) {
    shape_data = api::GetValue<std::vector<int64_t>>(shape_ptr);
  } else {
    MS_LOG(ERROR) << "can't get shape value. " << op->fullname_with_scope();
    return false;
  }
  (void)primitive->AddAttr(ops::kShape, api::MakeValue(shape_data));

  auto param_ptr = op->input(kInputIndex2)->cast<api::ParameterPtr>();
  if (param_ptr == nullptr) {
    MS_LOG(ERROR) << "param_ptr is nullptr. " << op->fullname_with_scope();
    return false;
  }
  auto param_value = param_ptr->default_param()->cast<api::TensorPtr>();
  if (param_value == nullptr) {
    MS_LOG(ERROR) << "param_value is nullptr." << op->fullname_with_scope();
    return false;
  }
  return !(static_cast<size_t>(param_value->DataSize()) < kDims1 ||
           static_cast<size_t>(param_value->DataSize()) > kDims4);
}

OpCheckerRegistrar g_ReshapeChecker("Reshape", new ReshapeChecker());
}  // namespace dpico
}  // namespace mindspore
