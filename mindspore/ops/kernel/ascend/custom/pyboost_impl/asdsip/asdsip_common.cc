/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/kernel/ascend/custom/pyboost_impl/asdsip_common.h"

namespace ms::pynative {
constexpr size_t kDefaultCacheSize = 50;
using FFTHandlePair = std::pair<uint64_t, asdFftHandle>;

class FFTCache {
 public:
  static FFTCache &GetInstance() {
    static FFTCache instance(kDefaultCacheSize);
    return instance;
  }
  asdFftHandle Get(const FFTParam &param) {
    uint64_t hash = HashFFTParam(param);
    std::lock_guard<std::mutex> lock(mtx_);
    auto iter = fft_map_.find(hash);
    if (iter != fft_map_.end()) {
      fft_cache_.splice(fft_cache_.begin(), fft_cache_, iter->second);
      return fft_cache_.front().second;
    } else {
      asdFftHandle handle = CreateHandle(param);
      fft_cache_.push_front({hash, handle});
      fft_map_[hash] = fft_cache_.begin();
      if (fft_cache_.size() > capaticy_) {
        DestrotyHandle(fft_cache_.back().second);
        fft_map_.erase(fft_cache_.back().first);
        fft_cache_.pop_back();
      }
      return handle;
    }
  }
  size_t GetSize() { return fft_cache_.size(); }
  void SetSize(size_t capaticy) {
    capaticy_ = capaticy;
    std::lock_guard<std::mutex> lock(mtx_);
    while (fft_cache_.size() > capaticy_) {
      DestrotyHandle(fft_cache_.back().second);
      fft_map_.erase(fft_cache_.back().first);
      fft_cache_.pop_back();
    }
  }
  void Clear() {
    for (auto &item : fft_cache_) {
      DestrotyHandle(item.second);
    }
    fft_map_.clear();
    fft_cache_.clear();
  }

 private:
  explicit FFTCache(size_t capaticy) : capaticy_(capaticy) {}
  size_t capaticy_;
  std::list<FFTHandlePair> fft_cache_;
  std::unordered_map<uint64_t, std::list<FFTHandlePair>::iterator> fft_map_;
  std::mutex mtx_;
};

size_t AsdSipFFTOpRunner::cache_capaticy_ = kDefaultCacheSize;
bool AsdSipFFTOpRunner::cache_set_flag_ = false;
void AsdSipFFTOpRunner::SetCacheSize(size_t capaticy) { cache_capaticy_ = capaticy; }

void AsdSipFFTOpRunner::Init(const FFTParam &param) {
  if (!cache_set_flag_) {
    FFTCache::GetInstance().SetSize(cache_capaticy_);
    cache_set_flag_ = true;
  }
  _device_context_->device_res_manager_->BindDeviceToCurrentThread(false);
  asd_fft_handle_ = FFTCache::GetInstance().Get(param);
}

void AsdSipFFTOpRunner::ProcessWithWorkspace() {
  AsdFftSetWorkSpace(asd_fft_handle_, _workspace_ptr_);
  AsdFftSetStream(asd_fft_handle_, stream());
  if (_inputs_[0].is_defined() && !_inputs_[0].is_contiguous()) {
    input_tensor_ = mindspore::device::ascend::ConvertType(_inputs_[0].contiguous().tensor());
  } else {
    input_tensor_ = mindspore::device::ascend::ConvertType(_inputs_[0].tensor());
  }
  output_tensor_ = mindspore::device::ascend::ConvertType(_outputs_[0].tensor());
}

size_t AsdSipFFTOpRunner::CalcWorkspace() {
  AsdFftGetWorkSpaceSize(asd_fft_handle_, workspace_size_);
  return workspace_size_;
}

void AsdSipFFTOpRunner::_Run() { PyboostRunner::_Run(); }

void AsdSipFFTOpRunner::LaunchKernel() {
  static std::unordered_map<std::string, AsdFftExecFunc> asd_fft_exec_map;
  auto iter = asd_fft_exec_map.find(_op_name_);
  AsdFftExecFunc asd_fft_exec = nullptr;
  if (iter == asd_fft_exec_map.end()) {
    auto asd_fft_exec_symbol = GetAsdSipApiFuncAddr(_op_name_.c_str());
    asd_fft_exec = reinterpret_cast<AsdFftExecFunc>(asd_fft_exec_symbol);
    asd_fft_exec_map[_op_name_] = asd_fft_exec;
  } else {
    asd_fft_exec = iter->second;
  }
  if (asd_fft_exec == nullptr) {
    MS_LOG(EXCEPTION) << "Get asdFftExec " << _op_name_ << "failed";
  }
  auto ret = asd_fft_exec(asd_fft_handle_, input_tensor_, output_tensor_);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Execute " << _op_name_ << " failed, ret: " << ret;
  }
  mindspore::device::ascend::Release(input_tensor_);
  mindspore::device::ascend::Release(output_tensor_);
}
}  // namespace ms::pynative
