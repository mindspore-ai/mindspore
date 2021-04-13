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

#include "backend/kernel_compiler/cpu/ps/embedding_look_up_ps_kernel.h"
#include <vector>
#include <memory>
#include <functional>
#include "backend/kernel_compiler/common_utils.h"
#include "ps/util.h"

namespace mindspore {
namespace kernel {
namespace ps {
using mindspore::ps::Util;
constexpr int kAxis = 0;
void EmbeddingLookUpPSKernel::InitKernel(
  const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &shapes) {
  const std::vector<std::shared_ptr<std::vector<size_t>>> &shape_vec = *shapes;
  input_shape_ = *(shape_vec[0]);
  first_dim_size_ = input_shape_[0];
  for (size_t i = 1; i < input_shape_.size(); ++i) {
    outer_dim_size_ *= input_shape_[i];
  }
  auto indices_shape = *(shape_vec[1]);
  indices_lens_ = 1;
  for (auto shape : indices_shape) {
    indices_lens_ = indices_lens_ * shape;
  }
  auto output_shape = *(shape_vec[2]);

  size_t offset = 0;
  for (size_t i = 0; i < rank_id_; i++) {
    offset += Util::LocalShard(input_shape_[kAxis], i, pserver_num_);
  }
  offset_ = offset;

  // input shape should be sharded after computing offset_;
  Shard(&input_shape_, kAxis);

  size_t output_size =
    std::accumulate(output_shape.begin(), output_shape.end(), sizeof(float), std::multiplies<size_t>());
  output_size_list_.emplace_back(output_size);
}

void EmbeddingLookUpPSKernel::ReInit(const std::vector<std::vector<size_t>> &shapes) {
  const auto &indices_shape = shapes[0];
  indices_lens_ = indices_shape[0];

  size_t output_size = sizeof(float) * indices_lens_;
  for (size_t i = kAxis + 1; i < input_shape_.size(); i++) {
    output_size *= input_shape_[i];
  }
  output_size_list_.clear();
  output_size_list_.emplace_back(output_size);
}

bool EmbeddingLookUpPSKernel::Execute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  return Launch(inputs, workspace, outputs);
}

void EmbeddingLookUpPSKernel::UpdateEmbeddings(float *embedding_table, const size_t *lookup_ids,
                                               const float *update_vals, size_t ids_size) {
  size_t copy_len = outer_dim_size_ * sizeof(float);
  size_t dest_len = copy_len;
  for (size_t i = 0; i < ids_size; ++i) {
    int index = lookup_ids[i] - offset_;
    if (index >= 0 && index < SizeToInt(first_dim_size_)) {
      auto ret =
        memcpy_s(embedding_table + index * outer_dim_size_, dest_len, update_vals + i * outer_dim_size_, copy_len);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "LookUpTable task memcpy failed.";
      }
    } else {
      MS_LOG(EXCEPTION) << "UpdateEmbeddings index invalid.";
    }
  }
}

const std::vector<size_t> &EmbeddingLookUpPSKernel::input_sizes() const { return input_shape_; }

const std::vector<size_t> &EmbeddingLookUpPSKernel::output_sizes() const { return GetOutputSizeList(); }

const std::vector<size_t> &EmbeddingLookUpPSKernel::workspace_sizes() const { return GetWorkspaceSizeList(); }
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore
