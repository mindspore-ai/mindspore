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

#include "checker/reduce_checker.h"
#include <vector>
#include <algorithm>
#include <set>
#include "mindapi/base/types.h"
#include "include/registry/converter_context.h"
#include "common/fetch_content.h"
#include "common/anf_util.h"
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
namespace {
STATUS GetAxesSet(const api::CNodePtr &op, const ShapeVector &input_shape, const api::PrimitivePtr &primitive,
                  std::set<int32_t> *axes_set) {
  if (axes_set == nullptr) {
    MS_LOG(ERROR) << "axes_set is nullptr. " << op->fullname_with_scope();
    return RET_ERROR;
  }
  DataInfo data_info;
  if (op->inputs().size() > kInputIndex2 && FetchDataFromParameterNode(op, kInputIndex2, &data_info) == lite::RET_OK) {
    if (data_info.data_type_ != static_cast<int>(kNumberTypeInt32)) {
      MS_LOG(ERROR) << "data_type not correct";
      return RET_ERROR;
    }
    auto data = reinterpret_cast<int32_t *>(data_info.data_.data());
    if (data == nullptr) {
      MS_LOG(ERROR) << "data is nullptr. " << op->fullname_with_scope();
      return RET_ERROR;
    }
    int data_size;
    if (GetDataSizeFromTensor(&data_info, &data_size) != RET_OK) {
      MS_LOG(ERROR) << "get data size from tensor failed.";
      return RET_ERROR;
    }
    (void)std::transform(
      data, data + data_size, std::inserter(*axes_set, axes_set->begin()),
      [input_shape](int32_t value) { return (value + static_cast<int32_t>(input_shape.size())) % input_shape.size(); });
  } else if (primitive->GetAttr(ops::kAxes) != nullptr) {
    auto axes = api::GetValue<std::vector<int64_t>>(primitive->GetAttr(ops::kAxes));
    (void)std::transform(
      axes.begin(), axes.end(), std::inserter(*axes_set, axes_set->begin()),
      [input_shape](int64_t value) { return (value + static_cast<int64_t>(input_shape.size())) % input_shape.size(); });
  }
  return RET_OK;
}

bool CheckAttr(const api::CNodePtr &op, mindspore::Format format, const api::PrimitivePtr &primitive,
               const ShapeVector &input_shape) {
  bool keep_dims = true;
  if (primitive->GetAttr(ops::kKeepDims) != nullptr) {
    keep_dims = api::GetValue<bool>(primitive->GetAttr(ops::kKeepDims));
  }
  std::set<int32_t> axes_set;
  if (GetAxesSet(op, input_shape, primitive, &axes_set) != RET_OK) {
    MS_LOG(ERROR) << "get axes set failed. " << op->fullname_with_scope();
    return false;
  }
  // special process when cnode is from caffe
  if (primitive->GetAttr(ops::kFmkType) != nullptr) {
    auto fmk_type = static_cast<converter::FmkType>(api::GetValue<int64_t>(primitive->GetAttr(ops::kFmkType)));
    if (fmk_type == converter::kFmkTypeCaffe) {
      return true;
    }
  }

  if (format == Format::NCHW && !input_shape.empty() && input_shape.at(0) == 1) {
    return (axes_set == std::set<int32_t>{kAxis3} && keep_dims) ||
           (axes_set == std::set<int32_t>{kAxis2, kAxis3} && !keep_dims);
  } else if (input_shape.size() == kDims3 && axes_set == std::set<int32_t>{kAxis2}) {
    return ((!keep_dims && input_shape.at(0) == 1) || keep_dims);
  }
  return false;
}
}  // namespace
bool ReduceChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  auto mode_ptr = primitive->GetAttr(ops::kMode);
  if (mode_ptr != nullptr) {
    auto reduce_mode = static_cast<ReduceMode>(api::GetValue<int64_t>(mode_ptr));
    if (reduce_mode < ReduceMode::Reduce_Mean || reduce_mode >= ReduceMode::Reduce_All) {
      MS_LOG(WARNING) << "unsupported reduce mode " << reduce_mode << " " << op->fullname_with_scope();
      return false;
    }
  }

  std::vector<int64_t> input_shape;
  if (GetInputShapeFromCNode(op, kInputIndex1, &input_shape) == RET_OK && !input_shape.empty()) {
    int64_t input_w;
    if (GetWidth(input_shape, format, &input_w) != RET_OK) {
      MS_LOG(ERROR) << "get input_w failed " << op->fullname_with_scope();
      return false;
    }
    if (input_shape.size() <= kDims2) {
      MS_LOG(WARNING) << "reduce op input_shape size need to be greater than 2";
      return false;
    }
    if (input_shape.size() == kDims4 && input_w > kMaxInputWOf4Dims) {
      MS_LOG(WARNING) << op->fullname_with_scope() << "'s input_w:" << input_w << " exceed the maximum limit "
                      << kMaxInputWOf4Dims;
      return false;
    }
    if (!CheckAttr(op, format, primitive, input_shape)) {
      MS_LOG(WARNING) << "axes_set or keep_dims val is unsupported by dpico. " << op->fullname_with_scope();
      return false;
    }
  }
  return true;
}

OpCheckerRegistrar g_ReduceChecker("ReduceFusion", new ReduceChecker());
}  // namespace dpico
}  // namespace mindspore
