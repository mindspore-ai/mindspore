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
#include "kernel/cpu/ps/embedding_look_up_proxy_kernel.h"
#include <vector>
#include "parallel/ps/worker.h"

namespace mindspore {
namespace kernel {
namespace ps {
void EmbeddingLookUpProxyKernel::InitKernel(const CNodePtr &kernel_node) {
  EmbeddingLookUpCPUKernel::InitKernel(kernel_node);

  for (auto dim : input_shape_) {
    input_dims_ *= dim;
  }

  if (mindspore::parallel::ps::Util::IsRoleOfWorker()) {
    key_ = AnfAlgo::GetNodeAttr<size_t>(kernel_node, kAttrPsKey);
  }
  std::vector<size_t> keys{key_, key_, key_};
  std::vector<size_t> values;
  values.insert(values.end(), input_shape_.begin(), input_shape_.end());
  values.insert(values.end(), indices_shape_.begin(), indices_shape_.end());
  values.insert(values.end(), output_shape_.begin(), output_shape_.end());
  std::vector<int> lens{SizeToInt(input_shape_.size()), SizeToInt(indices_shape_.size()),
                        SizeToInt(output_shape_.size())};
  const char *env_role = getenv(mindspore::parallel::ps::kEnvRole);
  if (env_role != nullptr && strcmp(env_role, mindspore::parallel::ps::kEnvRoleOfWorker) == 0) {
    parallel::ps::Worker<float>::GetInstance().AddEmbeddingTable(key_, input_shape_[axis_]);
    parallel::ps::Worker<float>::GetInstance().InitPSEmbeddingTable(keys, values, lens);
  }
}

bool EmbeddingLookUpProxyKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> & /*workspace*/,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  auto indices_addr = reinterpret_cast<int *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  size_t input_size = inputs[1]->size;
  size_t output_size = outputs[0]->size;

  size_t size = input_size / sizeof(float);
  ::ps::SArray<float> lookup_ids(size, 0);
  ::ps::SArray<int> lengths{size};
  ::ps::SArray<float> lookup_result;

  auto ret = memcpy_s(lookup_ids.data(), input_size, indices_addr, input_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Lookup id memcpy failed.";
  }
  parallel::ps::Worker<float>::GetInstance().DoPSEmbeddingLookup({key_}, lookup_ids, lengths, lookup_result,
                                                                 parallel::ps::kEmbeddingLookupCmd);

  auto ret2 = memcpy_s(output_addr, output_size, lookup_result.data(), output_size);
  if (ret2 != EOK) {
    MS_LOG(EXCEPTION) << "Lookup result memcpy failed.";
  }
  return true;
}
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore
