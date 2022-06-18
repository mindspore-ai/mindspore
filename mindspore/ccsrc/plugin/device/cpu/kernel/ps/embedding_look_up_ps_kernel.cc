/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/ps/embedding_look_up_ps_kernel.h"
#include <vector>
#include <memory>
#include <functional>
#include "kernel/common_utils.h"
#include "ps/util.h"

namespace mindspore {
namespace kernel {
namespace ps {
using mindspore::ps::Util;
constexpr int kAxis = 0;
constexpr size_t kEmbeddingLookUpPSInputSize = 3;

void EmbeddingLookUpPSKernelMod::InitKernel(const std::shared_ptr<std::vector<std::shared_ptr<ShapeVector>>> &shapes) {
  const std::vector<std::shared_ptr<ShapeVector>> &shape_vec = *shapes;
  if (shape_vec.size() < kEmbeddingLookUpPSInputSize) {
    MS_LOG(EXCEPTION) << "EmbeddingLookUpPSKernelMod needs " << kEmbeddingLookUpPSInputSize << " input shapes, but got "
                      << shape_vec.size();
  }
  for (auto shape : shape_vec) {
    MS_EXCEPTION_IF_NULL(shape);
  }
  auto input_shape = *(shape_vec[0]);
  if (input_shape.empty()) {
    MS_LOG(EXCEPTION) << "Input shape can not empty";
  }

  first_dim_size_ = LongToSize(input_shape[0]);
  outer_dim_size_ *= SizeOf(input_shape);
  auto indices_shape = *(shape_vec[1]);
  indices_lens_ = SizeOf(indices_shape);
  size_t output_index = 2;
  auto output_shape = *(shape_vec[output_index]);

  int64_t offset = 0;
  for (size_t i = 0; i < rank_id_; i++) {
    offset += Util::LocalShard(input_shape[kAxis], SizeToLong(i), SizeToLong(pserver_num_));
  }
  offset_ = offset;

  // input shape must be sharded after computing offset_;
  Shard(&input_shape, kAxis);

  input_shape_ = Convert2SizeT(input_shape);

  size_t output_size = sizeof(float) * SizeOf(output_shape);
  (void)output_size_list_.emplace_back(output_size);
}

void EmbeddingLookUpPSKernelMod::ReInit(const std::vector<ShapeVector> &shapes) {
  if (shapes.empty() || shapes[0].empty()) {
    MS_LOG(EXCEPTION) << "Shape can not empty";
  }
  const auto &indices_shape = shapes[0];
  indices_lens_ = LongToSize(indices_shape[0]);

  size_t output_size = sizeof(float) * indices_lens_;
  for (size_t i = kAxis + 1; i < input_shape_.size(); i++) {
    output_size *= input_shape_[i];
  }
  output_size_list_.clear();
  (void)output_size_list_.emplace_back(output_size);
}

bool EmbeddingLookUpPSKernelMod::Execute(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  return Launch(inputs, workspace, outputs);
}

void EmbeddingLookUpPSKernelMod::UpdateEmbeddings(float *embedding_table, const size_t *lookup_ids,
                                                  const float *update_vals, size_t ids_size) {
  size_t copy_len = outer_dim_size_ * sizeof(float);
  size_t dest_len = copy_len;
  for (size_t i = 0; i < ids_size; ++i) {
    int index = SizeToInt(lookup_ids[i]) - LongToInt(offset_);
    if (index < 0 || index >= SizeToInt(first_dim_size_)) {
      MS_LOG(EXCEPTION) << "UpdateEmbeddings index invalid.";
    }
    auto ret = memcpy_s(embedding_table + IntToSize(index) * outer_dim_size_, dest_len,
                        update_vals + i * outer_dim_size_, copy_len);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "LookUpTable task memcpy failed.";
    }
  }
}

const std::vector<size_t> &EmbeddingLookUpPSKernelMod::input_sizes() const { return input_shape_; }

const std::vector<size_t> &EmbeddingLookUpPSKernelMod::output_sizes() const { return GetOutputSizeList(); }

const std::vector<size_t> &EmbeddingLookUpPSKernelMod::workspace_sizes() const { return GetWorkspaceSizeList(); }

int64_t EmbeddingLookUpPSKernelMod::offset() const { return offset_; }
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore
