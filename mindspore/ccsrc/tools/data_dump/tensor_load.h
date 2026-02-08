/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_TENSOR_LOAD_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_TENSOR_LOAD_H_

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ir/tensor_new.h"
#include "tools/data_dump/dump_json_parser.h"
#include "tools/data_dump/dump_utils.h"
#include "tools/tensor_data.h"

namespace mindspore {
class TensorLoader {
 public:
  TensorLoader() = default;

  ~TensorLoader() { EmptyCurrentTensor(); }

  bool TensorExistsInCurrent(const std::string &tensor_name) const {
    return tensor_list_map_.find(tensor_name) != tensor_list_map_.end();
  }

  /*
   * Feature group: Dump, Online debugger and Offline debugger.
   * Target device group: Ascend, GPU.
   * Runtime category: Old runtime, MindRT.
   * Description: Load new tensor into tensor_list_map_ (debugger backend cache). In offline debugger, add ":prev" to
   * the previous tensor's name to avoid segfault caused by wrongly evicting the tensor when memory limit is enabled.
   */
  bool LoadNewTensor(const std::shared_ptr<TensorData> &tensor, bool keep_prev) {
    std::lock_guard<std::mutex> lg(lock_);
    auto tensor_name = tensor->GetName();
    std::string key_name = tensor_name;
    tensor_list_map_[key_name] = tensor;  // use [] instead of insert to ensure latest value
    return true;
  }

  std::shared_ptr<TensorData> GetTensor(const std::string &tensor_name) const {
    auto iter = tensor_list_map_.find(tensor_name);
    if (iter != tensor_list_map_.end()) {
      return iter->second;
    }
    return nullptr;
  }

  void EmptyCurrentTensor() { tensor_list_map_.clear(); }

  /*
   * Feature group: Dump.
   * Target device group: GPU, Ascend.
   * Runtime category: Old runtime, MindRT.
   * Description: Load tensor data from debugger backend cache (tensor_list_map_) and dump to file in npy format,
   *              used for GPU and Ascend KernelByKernel mode.
   */
  bool DumpTensorToFile(const std::string &filepath, const std::string &tensor_name, size_t slot) {
    if (filepath.empty()) {
      MS_LOG(ERROR) << "Dump file path is null!";
      return false;
    }

    std::string tensor_loader_name = tensor_name + ":" + std::to_string(slot);
    std::map<std::string, std::shared_ptr<TensorData>>::const_iterator iter = tensor_list_map_.find(tensor_loader_name);
    if (iter != tensor_list_map_.cend()) {
      std::shared_ptr<TensorData> node = iter->second;
      std::string path = filepath + '.' + node->GetFormat() + '.' + node->GetTypeString();
      if (node->GetByteSize() == 0) {
        MS_LOG(INFO) << "The byte size is 0 for tensor: " << tensor_loader_name;
        return false;
      }
      auto type_string = node->GetTypeString();
      if (type_string == "bfloat16") {
        std::shared_ptr<tensor::Tensor> bfloat16_tensor =
          tensor::from_buffer(TypeId::kNumberTypeBFloat16, node->GetShape(),
                              static_cast<void *>(const_cast<char *>(node->GetDataPtr())), node->GetByteSize());
        std::shared_ptr<tensor::Tensor> float32_tensor =
          std::make_shared<tensor::Tensor>(*bfloat16_tensor, TypeId::kNumberTypeFloat32);
        return DumpJsonParser::DumpToFile(path, float32_tensor->data_c(), float32_tensor->Size(),
                                          float32_tensor->shape_c(),
                                          static_cast<TypeId>(float32_tensor->data_type_c()));
      } else if (type_string == "int4") {
        auto int8_tensor = tensor::from_spec(TypeId::kNumberTypeInt8, node->GetShape(), device::DeviceType::kCPU);
        bool split_succeed =
          SplitInt8ToInt4x2(node->GetDataPtr(), node->GetByteSize(), int8_tensor->data_c(), int8_tensor->DataSize());
        if (!split_succeed) {
          return false;
        }
        return DumpJsonParser::DumpToFile(path, int8_tensor->data_c(), int8_tensor->Size(), int8_tensor->shape_c(),
                                          static_cast<TypeId>(int8_tensor->data_type_c()));
      }
      return DumpJsonParser::DumpToFile(path, node->GetDataPtr(), node->GetByteSize(), node->GetShape(),
                                        StringToTypeId(node->GetTypeString()));
    }
    MS_LOG(INFO) << "Tensor name:" << tensor_name << " not found in tensor_list_map_";
    return false;
  }

 private:
  // the pair is (device_id, iteration)
  std::map<std::string, std::shared_ptr<TensorData>> tensor_list_map_;
  std::mutex lock_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_TOOLS_DATA_DUMP_TENSOR_LOAD_H_
