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
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "include/api/types.h"
#include "include/dataset/lite_cv/lite_mat.h"
#include "include/dataset/lite_cv/image_process.h"
#include "include/dataset/vision_lite.h"
#include "include/dataset/execute.h"

using mindspore::dataset::Execute;
using mindspore::dataset::LDataType;
using mindspore::dataset::LiteMat;
using mindspore::dataset::PaddBorderType;
using mindspore::dataset::vision::Decode;

int main(int argc, char **argv) {
  std::ifstream ifs("test_image.jpg");

  if (!ifs.is_open() || !ifs.good()) {
    std::cout << "fail to load image, check image path" << std::endl;
    return -1;
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  mindspore::MSTensor image("file", mindspore::DataType::kNumberTypeUInt8, {static_cast<int64_t>(size)}, nullptr, size);

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(image.MutableData()), size);
  ifs.close();

  auto decode = Decode();
  auto executor = Execute(decode);
  executor(image, &image);

  constexpr int32_t image_h = 0;
  constexpr int32_t image_w = 1;
  constexpr int32_t image_c = 2;
  LiteMat lite_mat_rgb(image.Shape()[image_w], image.Shape()[image_h], image.Shape()[image_c],
                       const_cast<void *>(image.Data().get()), LDataType::UINT8);
  std::cout << "lite_mat_rgb: height=" << lite_mat_rgb.height_ << ", width=" << lite_mat_rgb.width_ << std::endl;

  LiteMat lite_mat_resize;
  constexpr int32_t target_size = 256;
  ResizeBilinear(lite_mat_rgb, lite_mat_resize, target_size, target_size);
  std::cout << "lite_mat_resize: height=" << lite_mat_resize.height_ << ", width=" << lite_mat_resize.width_
            << std::endl;

  LiteMat lite_mat_pad;
  constexpr int32_t pad_top = 30;
  constexpr int32_t pad_bottom = 30;
  constexpr int32_t pad_left = 10;
  constexpr int32_t pad_right = 10;
  constexpr int32_t pad_color = 255;
  Pad(lite_mat_resize, lite_mat_pad, pad_top, pad_bottom, pad_left, pad_right, PaddBorderType::PADD_BORDER_CONSTANT,
      pad_color, pad_color, pad_color);
  std::cout << "lite_mat_pad: height=" << lite_mat_pad.height_ << ", width=" << lite_mat_pad.width_ << std::endl;
}
