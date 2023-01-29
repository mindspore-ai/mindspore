/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/max_pool_with_argmax_cpu_kernel.h"
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/max_pool_with_argmax.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxPoolWithArgmaxInputsNum = 1;
constexpr size_t kMaxPoolWithArgmaxOutputsNum = 2;
constexpr size_t kInputRank = 4;
constexpr size_t kIndex3 = 3;
constexpr int kPadHalf = 2;
}  // namespace

bool MaxPoolWithArgmaxCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MaxPoolWithArgmax>(base_operator);
  data_format_ = kernel_ptr->get_format();
  auto kernel_size = kernel_ptr->get_kernel_size();
  auto strides = kernel_ptr->get_strides();
  if (kernel_size.size() < kIndex3 || strides.size() < kIndex3) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than 3, but got "
                      << strides.size();
  }

  window_height_ = LongToInt(kernel_size[kDim1]);
  window_width_ = LongToInt(kernel_size[kDim2]);
  if (window_height_ < 1 || window_width_ < 1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', expected kernel_size to be Union[int, tuple[int]] with value no "
                                "less than 1 "
                                "but got the window height: "
                             << window_height_ << ", and the window width: " << window_width_;
    return false;
  }
  stride_height_ = LongToInt(strides[kDim1]);
  stride_width_ = LongToInt(strides[kDim2]);
  if (stride_height_ < 1 || stride_width_ < 1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', expected strides to be Union[int, tuple[int]] with value no "
                                "less than 1 "
                                "but got the window height: "
                             << window_height_ << ", and the window width: " << window_width_;
  }
  pad_mode_ = kernel_ptr->get_pad_mode();
  if (pad_mode_ == PadMode::SAME) {
    int tmp_height = (input_height_ / stride_height_) * stride_height_ == input_height_
                       ? (input_height_ / stride_height_)
                       : (input_height_ / stride_height_) + 1;
    pad_height_ = std::max<int>(0, (tmp_height - 1) * stride_height_ + window_height_ - input_height_);
    int tmp_width = (input_width_ / stride_width_) * stride_width_ == input_width_ ? (input_width_ / stride_width_)
                                                                                   : (input_width_ / stride_width_) + 1;
    pad_width_ = std::max<int>(0, (tmp_width - 1) * stride_width_ + window_width_ - input_width_);
    pad_top_ = pad_height_ / kPadHalf;
    pad_left_ = pad_width_ / kPadHalf;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  // pair = [is_match, index]
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;
  return true;
}

void MaxPoolWithArgmaxCpuKernelMod::ResizedInputSize(const std::vector<KernelTensorPtr> &inputs) {
  auto x_shape = inputs[kIndex0]->GetShapeVector();
  if (x_shape.size() != kInputRank) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the input 'x' must be 4-dimensional.";
  }
  for (size_t i = 0; i < x_shape.size(); i++) {
    if (x_shape[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', expected input have non-empty spatial dimensions, "
                                  "but input has sizes "
                               << x_shape[i] << " wit h dimension " << i << " being empty.";
    }
  }
  batch_ = LongToInt(x_shape[kDim0]);
  if (data_format_ == Format::NHWC) {
    channel_ = LongToInt(x_shape[kDim3]);
    input_height_ = LongToInt(x_shape[kDim1]);
    input_width_ = LongToInt(x_shape[kDim2]);
  } else {
    //  data_format_ == Format::NCHW
    channel_ = LongToInt(x_shape[kDim1]);
    input_height_ = LongToInt(x_shape[kDim2]);
    input_width_ = LongToInt(x_shape[kDim3]);
  }
}

void MaxPoolWithArgmaxCpuKernelMod::ResizedOutputSize(const std::vector<KernelTensorPtr> &outputs) {
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  output_height_ = LongToInt(output_shape[kDim2]);
  output_width_ = LongToInt(output_shape[kDim3]);
  std::vector<int64_t> mask_shape = outputs[kIndex1]->GetShapeVector();
  if (mask_shape != output_shape) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', expected output and mask have the same shape "
                                "but output has shape "
                             << output_shape << ", and mask shape: " << mask_shape;
  }
}

int MaxPoolWithArgmaxCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaxPoolWithArgmaxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaxPoolWithArgmaxOutputsNum, kernel_name_);
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MaxPoolWithArgmax>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast op from BaseOperator to MaxPoolWithArgmax failed.";
    return KRET_RESIZE_FAILED;
  }
  ResizedInputSize(inputs);
  ResizedOutputSize(outputs);
  return KRET_OK;
}

template <typename T>
bool MaxPoolWithArgmaxCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  auto *x = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  MS_EXCEPTION_IF_NULL(x);
  auto *output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  MS_EXCEPTION_IF_NULL(output);
  auto *mask = reinterpret_cast<int32_t *>(outputs.at(kIndex1)->addr);
  MS_EXCEPTION_IF_NULL(mask);
  int cWeight;
  int hWeight;
  int wWeight;
  if (data_format_ == Format::NHWC) {
    cWeight = 1;
    wWeight = channel_ * cWeight;
    hWeight = input_height_ * wWeight;
  } else {
    // data_format == NCHW
    wWeight = 1;
    hWeight = input_width_ * wWeight;
    cWeight = input_height_ * hWeight;
  }
  const int batch = this->batch_;
  const int channel = this->channel_;
  const int i_h = this->input_height_;
  const int i_w = this->input_width_;
  const int s_h = this->stride_height_;
  const int s_w = this->stride_width_;
  const int w_h = this->window_height_;
  const int w_w = this->window_width_;
  const int pad_top = this->pad_top_;
  const int pad_left = this->pad_left_;
  const int o_h = this->output_height_;
  const int o_w = this->output_width_;
  const size_t length = IntToSize(batch * channel * o_h * o_w);

  auto task = [x, output, mask, &batch, &channel, &i_h, &i_w, &s_h, &s_w, &w_h, &w_w, &pad_top, &pad_left, &o_h, &o_w,
               &wWeight, &hWeight, &cWeight](size_t start, size_t end) {
    for (int i = SizeToInt(start); i < SizeToInt(end); ++i) {
      const int posn = i / (channel * o_h * o_w);
      const int posc = i / (o_h * o_w) % channel;
      const int posh = i / o_w % o_h;
      const int posw = i % o_w;
      int hstart = posh * s_h - pad_top;
      int wstart = posw * s_w - pad_left;
      const int hend = std::min<int>(hstart + w_h, i_h);
      const int wend = std::min<int>(wstart + w_w, i_w);
      hstart = std::max<int>(hstart, 0);
      wstart = std::max<int>(wstart, 0);
      int32_t inputStart = posn * channel * i_h * i_w;
      int32_t maxIdx = posc * cWeight + hstart * hWeight + wstart * wWeight;
      T maxData = x[inputStart + maxIdx];
      for (int hcur = hstart; hcur < hend; ++hcur) {
        for (int wcur = wstart; wcur < wend; ++wcur) {
          int32_t inputIdx = posc * cWeight + hcur * hWeight + wcur * wWeight;
          T inputData = x[inputStart + inputIdx];
          if (inputData > maxData) {
            maxIdx = inputIdx;
            maxData = inputData;
          }
        }
      }
      output[i] = maxData;
      mask[i] = maxIdx;
    }
  };
  ParallelLaunchAutoSearch(task, length, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, MaxPoolWithArgmaxCpuKernelMod::MaxPoolWithArgmaxFunc>>
  MaxPoolWithArgmaxCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxCpuKernelMod::LaunchKernel<float>},
};

std::vector<KernelAttr> MaxPoolWithArgmaxCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxPoolWithArgmaxFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaxPoolWithArgmax, MaxPoolWithArgmaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
