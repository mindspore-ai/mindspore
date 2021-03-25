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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_BUCKET_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_BUCKET_H_

#include <memory>
#include "runtime/device/bucket.h"

namespace mindspore::device::ascend {
class AscendBucket : public Bucket {
 public:
  AscendBucket(uint32_t id, uint32_t bucket_size) : Bucket(id, bucket_size) {}
  ~AscendBucket() override = default;

  void Init() override;

 private:
  void AllocateAllReduceAddr() override;
  void FreeAllDeviceMem() override;
  void FreeDeviceMem(void *dev_ptr) override;
  void CopyTensorToContiguousMemory() override;
  void LaunchAllReduce() override;
  std::shared_ptr<LaunchKernel> CreateLaunchMul() override;
  std::shared_ptr<LaunchKernel> CreateLaunchAtomicClean();
  void CleanAllReduceInputAddr();
};
}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_BUCKET_H_
