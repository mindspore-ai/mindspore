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

#include "fl/armour/cipher/cipher_unmask.h"
#include "fl/server/common.h"
#include "fl/server/local_meta_store.h"
#include "fl/armour/cipher/cipher_meta_storage.h"

namespace mindspore {
namespace armour {
bool CipherUnmask::UnMask(const std::map<std::string, AddressPtr> &data) {
  MS_LOG(INFO) << "CipherMgr::UnMask START";
  clock_t start_time = clock();
  std::vector<float> noise;

  bool ret = cipher_init_->cipher_meta_storage_.GetClientNoisesFromServer(fl::server::kCtxClientNoises, &noise);
  if (!ret || noise.size() != cipher_init_->featuremap_) {
    MS_LOG(WARNING) << "Client noises is not ready";
    return false;
  }

  size_t data_size = fl::server::LocalMetaStore::GetInstance().value<size_t>(fl::server::kCtxFedAvgTotalDataSize);
  if (data_size == 0) {
    MS_LOG(ERROR) << "FedAvgTotalDataSize equals to 0";
    return false;
  }
  int sum_size = 0;
  for (auto iter = data.begin(); iter != data.end(); ++iter) {
    if (iter->second == nullptr) {
      MS_LOG(ERROR) << "AddressPtr is nullptr";
      return false;
    }
    size_t size_data = iter->second->size / sizeof(float);
    float *in_data = reinterpret_cast<float *>(iter->second->addr);
    for (size_t i = 0; i < size_data; ++i) {
      in_data[i] = in_data[i] + noise[i + IntToSize(sum_size)] / data_size;
    }
    sum_size += IntToSize(size_data);
    for (size_t i = 0; i < data.size(); ++i) {
      MS_LOG(INFO) << " index : " << i << " in_data unmask: " << in_data[i] * data_size;
    }
  }
  MS_LOG(INFO) << "CipherMgr::UnMask sum_size : " << sum_size;
  MS_LOG(INFO) << "CipherMgr::UnMask feature_map : " << cipher_init_->featuremap_;
  clock_t end_time = clock();
  double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
  MS_LOG(INFO) << "Unmask success time is : " << duration;
  return true;
}
}  // namespace armour
}  // namespace mindspore
