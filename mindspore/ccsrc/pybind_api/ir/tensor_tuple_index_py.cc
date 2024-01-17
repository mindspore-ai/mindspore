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

#include "pybind_api/ir/tensor_tuple_index_py.h"

namespace py = pybind11;
namespace mindspore::tensor {
void GetItemByNoneWithView::GetItemWithView(TensorTupleIndexInfoForView *ViewInfo, py::handle obj) {
  auto slice_op_info = std::make_shared<pynative::SliceOpInfo>();
  slice_op_info->slice_op_name = prim::kPrimExpandDims->name();
  (void)slice_op_info->slice_index_inputs.emplace_back(std::make_shared<pynative::FastValue>(ViewInfo->m_dim));
  (void)slice_op_info->data_indexs.emplace_back(0);
  (void)ViewInfo->m_slice_op_infos->emplace_back(slice_op_info);
  ViewInfo->m_new_data_shape.insert(ViewInfo->m_new_data_shape.begin() + ViewInfo->m_dim, 1);
  ViewInfo->m_dim++;
}

void GetItemByEllipsisWithView::GetItemWithView(TensorTupleIndexInfoForView *ViewInfo, py::handle obj) {
  ViewInfo->CheckEllipsisCounter();
  ViewInfo->m_dim += ViewInfo->m_data_shape.size() - ViewInfo->m_specified_dimensions;
  ViewInfo->m_ellipsis_counter += 1;
}

void GetItemByIntWithView::GetItemWithView(TensorTupleIndexInfoForView *ViewInfo, py::handle obj) {
  auto index = py::cast<int64_t>(obj);
  if (index >= ViewInfo->m_new_data_shape[ViewInfo->m_dim] || index < -ViewInfo->m_new_data_shape[ViewInfo->m_dim]) {
    // Raise exception in python, because python iterator need raise IndexError to stop for loop.
    ViewInfo->m_data_transfer_types->emplace_back(static_cast<int>(ValueTransferType::kRaiseIndexError));
    ViewInfo->m_data_transfer_args->emplace_back(py::make_tuple(index, ViewInfo->m_new_data_shape[ViewInfo->m_dim]));
    return;
  }
  int64_t transformed_number = CheckRange(index, ViewInfo->m_new_data_shape[ViewInfo->m_dim]);
  auto slice_op_info = std::make_shared<pynative::SliceOpInfo>();
  slice_op_info->slice_op_name = prim::kPrimSelectView->name();
  (void)slice_op_info->slice_index_inputs.emplace_back(std::make_shared<pynative::FastValue>(transformed_number));
  (void)slice_op_info->slice_index_inputs.emplace_back(std::make_shared<pynative::FastValue>(ViewInfo->m_dim));
  (void)slice_op_info->data_indexs.emplace_back(0);
  (void)ViewInfo->m_slice_op_infos->emplace_back(slice_op_info);
  (void)ViewInfo->m_new_data_shape.erase(ViewInfo->m_new_data_shape.begin() + ViewInfo->m_dim);
}

void GetItemBySliceWithView::GetItemWithView(TensorTupleIndexInfoForView *ViewInfo, py::handle obj) {
  auto slice_info = Slice(TensorIndex(obj).slice(), ViewInfo->m_new_data_shape[ViewInfo->m_dim]);
  std::vector<int64_t> begin_info(ViewInfo->m_new_data_shape.size(), 0);
  std::vector<int64_t> end_info(ViewInfo->m_new_data_shape);
  std::vector<int64_t> step_info(ViewInfo->m_new_data_shape.size(), 1);

  if (slice_info.step() < 0) {
    ViewInfo->m_data_transfer_args->clear();
    ViewInfo->m_data_transfer_types->clear();
    ViewInfo->m_could_apply_view = false;
    return;
  }

  if (slice_info.start() == 0 && slice_info.step() == 1 && slice_info.stop() == end_info[ViewInfo->m_dim]) {
    ViewInfo->m_dim++;
    return;
  }

  ViewInfo->m_empty_strided_slice_result = (slice_info.start() >= slice_info.stop());
  begin_info[ViewInfo->m_dim] = slice_info.start();
  end_info[ViewInfo->m_dim] = slice_info.stop();
  step_info[ViewInfo->m_dim] = slice_info.step();
  auto slice_op_info = std::make_shared<pynative::SliceOpInfo>();
  slice_op_info->slice_op_name = prim::kPrimStridedSlice->name();
  (void)slice_op_info->slice_index_inputs.emplace_back(std::make_shared<pynative::FastValue>(begin_info));
  (void)slice_op_info->slice_index_inputs.emplace_back(std::make_shared<pynative::FastValue>(end_info));
  (void)slice_op_info->slice_index_inputs.emplace_back(std::make_shared<pynative::FastValue>(step_info));
  (void)slice_op_info->data_indexs.emplace_back(0);
  (void)ViewInfo->m_slice_op_infos->emplace_back(slice_op_info);
  ViewInfo->m_new_data_shape[ViewInfo->m_dim] =
    (slice_info.stop() + slice_info.step() - 1 - slice_info.start()) / slice_info.step();
  ViewInfo->m_dim++;
}
}  // namespace mindspore::tensor
