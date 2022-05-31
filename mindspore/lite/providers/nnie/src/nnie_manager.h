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

#ifndef MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_MANAGER_H_
#define MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_MANAGER_H_
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "include/errorcode.h"
#include "include/api/types.h"
#include "include/api/allocator.h"
#include "src/nnie_common.h"
#include "src/nnie_cfg_parser.h"

namespace mindspore {
namespace nnie {
class NNIEManager {
 public:
  static NNIEManager *GetInstance(const void *model_buf) {
    static std::map<const void *, NNIEManager *> managers_;
    auto iter = managers_.find(model_buf);
    if (iter != managers_.end()) {
      return iter->second;
    } else {
      auto manager = new (std::nothrow) NNIEManager();
      if (manager == nullptr) {
        return manager;
      } else {
        managers_[model_buf] = manager;
        return manager;
      }
    }
  }

  NNIEManager() {}

  ~NNIEManager() {}

  int Init(char *model_buf, int size, const std::vector<mindspore::MSTensor> &inputs);

  int CfgInit(const Flags &flags, int max_seg_id);

  void SetInputNum(int max_input_num);

  int SetAllocatorInputs(std::vector<mindspore::MSTensor> *inputs, bool run_box, std::shared_ptr<Allocator> allocator,
                         unsigned int seg_id);

  int SetAllocatorOutputs(std::vector<mindspore::MSTensor> *outputs, bool run_box, std::shared_ptr<Allocator> allocator,
                          unsigned int seg_id);

  int SetAllocator(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                   std::shared_ptr<Allocator> allocator, unsigned int seg_id);

  int FillData(std::vector<mindspore::MSTensor> *inputs, unsigned int seg_id);

  int Run(std::vector<mindspore::MSTensor> *outputs, unsigned int seg_id,
          const std::vector<std::vector<int64_t>> &outputs_shape);

  void Release(bool resize_flag);

  int LoadInputs(std::vector<mindspore::MSTensor> *inputs, std::shared_ptr<Allocator> allocator);

  int LoadOutputs(std::vector<mindspore::MSTensor> *outputs, std::shared_ptr<Allocator> allocator);

  int SetBlobAddr(SVP_SRC_BLOB_S *blob, HI_U64 virt, mindspore::MSTensor *tensor, std::shared_ptr<Allocator> allocator);

  void SetMaxSegId(int max_id) {
    if (max_id > max_seg_id_) {
      max_seg_id_ = max_id;
    }
  }

  inline int GetMaxSegId() { return max_seg_id_; }

  inline Flags *GetFlags() { return &flags_; }

  inline bool GetLoadModel() { return load_model_; }

  void SetLoadModel(bool flag) { load_model_ = flag; }

 private:
  int SetAllocatorTensor(mindspore::MSTensor *tensor, SVP_SRC_BLOB_S *blob, std::shared_ptr<Allocator> allocator);

  int GetOutputData(std::vector<mindspore::MSTensor> *outputs, const std::vector<std::vector<int64_t>> &outputs_shape,
                    bool run_box = false);

  int MallocBlobData(SVP_SRC_BLOB_S *blob, mindspore::MSTensor *tensor, HI_U32 blob_size);

  int FillRoiPooling(mindspore::MSTensor *input);
  char *wk_model_ = nullptr;

  int model_size_ = 0;

  NnieRunCfg nnie_cfg_;
  int max_seg_id_ = 0;
  Flags flags_;
  bool load_model_ = false;
  std::vector<SVP_SRC_BLOB_S *> blobs_;
  std::vector<mindspore::MSTensor *> tensors_;
};
}  // namespace nnie
}  // namespace mindspore
#endif  // MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_MANAGER_H_
