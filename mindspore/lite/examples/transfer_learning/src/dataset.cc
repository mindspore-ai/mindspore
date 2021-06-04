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

#include "src/dataset.h"
#include <dirent.h>
#include <arpa/inet.h>
#include <map>
#include <iostream>
#include <fstream>
#include <memory>
#include "src/utils.h"

#pragma pack(push, 1)

typedef struct {  // Total: 54 bytes
  uint16_t type;  // Magic identifier: 0x4d42
  uint32_t size;
  uint16_t reserved1;
  uint16_t reserved2;
  uint32_t offset;
  uint32_t dib_header_size;
  int32_t width;
  int32_t height;
  uint16_t channels;
  uint16_t bits_per_pixel;
  uint32_t compression;
  uint32_t image_size_bytes;
  int32_t x_resolution_ppm;
  int32_t y_resolution_ppm;
  uint32_t num_colors;
  uint32_t important_colors;
} bmp_header;

#pragma pack(pop)

float CH_MEAN[3] = {0.485, 0.456, 0.406};
float CH_STD[3] = {0.229, 0.224, 0.225};

using LabelId = std::map<std::string, int>;
constexpr int kClassNum = 10;
constexpr int kBGRDim = 2;
constexpr float kRGBMAX = 255.0f;
constexpr int kRGBDims = 3;

static char *ReadBitmapFile(const std::string &filename, size_t *size) {
  MS_ASSERT(size != nullptr);
  *size = 0;
  bmp_header bitmap_header;
  std::ifstream ifs(filename);
  if (!ifs.good() || !ifs.is_open()) {
    std::cerr << "file: " << filename << " does not exist or failed to open";
    return nullptr;
  }

  ifs.read(reinterpret_cast<char *>(&bitmap_header), sizeof(bmp_header));
  if (bitmap_header.type != 0x4D42) {
    std::cerr << "file: " << filename << " magic number does not match BMP";
    ifs.close();
    return nullptr;
  }

  ifs.seekg(bitmap_header.offset, std::ios::beg);

  unsigned char *bmp_image = reinterpret_cast<unsigned char *>(malloc(bitmap_header.image_size_bytes));
  if (bmp_image == nullptr) {
    ifs.close();
    return nullptr;
  }

  ifs.read(reinterpret_cast<char *>(bmp_image), bitmap_header.image_size_bytes);

  size_t buffer_size = bitmap_header.width * bitmap_header.height * kRGBDims;
  float *hwc_bin_image = new (std::nothrow) float[buffer_size];
  if (hwc_bin_image == nullptr) {
    free(bmp_image);
    ifs.close();
    return nullptr;
  }

  // swap the R and B values to get RGB (bitmap is BGR)
  // swap columns (in BMP, first pixel is lower left one...)
  const size_t channels = 3;
  const size_t hStride = channels * bitmap_header.width;
  const size_t height = bitmap_header.height;

  for (int h = 0; h < bitmap_header.height; h++) {
    for (int w = 0; w < bitmap_header.width; w++) {
      hwc_bin_image[h * hStride + w * channels + 0] =
        (((static_cast<float>(bmp_image[(height - h - 1) * hStride + w * channels + kBGRDim])) / kRGBMAX) -
         CH_MEAN[0]) /
        CH_STD[0];
      hwc_bin_image[h * hStride + w * channels + 1] =
        (((static_cast<float>(bmp_image[(height - h - 1) * hStride + w * channels + 1])) / kRGBMAX) - CH_MEAN[1]) /
        CH_STD[1];
      hwc_bin_image[h * hStride + w * channels + kBGRDim] =
        (((static_cast<float>(bmp_image[(height - h - 1) * hStride + w * channels + 0])) / kRGBMAX) -
         CH_MEAN[kBGRDim]) /
        CH_STD[kBGRDim];
    }
  }

  *size = buffer_size * sizeof(float);
  free(bmp_image);
  ifs.close();
  char *ret_buf = reinterpret_cast<char *>(hwc_bin_image);
  return ret_buf;
}

char *ReadFile(const std::string &file, size_t *size) {
  MS_ASSERT(size != nullptr);
  std::string realPath(file);
  std::ifstream ifs(realPath);
  if (!ifs.good()) {
    std::cerr << "file: " << realPath << " does not exist";
    return nullptr;
  }

  if (!ifs.is_open()) {
    std::cerr << "file: " << realPath << " open failed";
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char[]> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    std::cerr << "malloc buf failed, file: " << realPath;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), *size);
  ifs.close();

  return buf.release();
}

DataSet::~DataSet() {
  for (auto itr = train_data_.begin(); itr != train_data_.end(); ++itr) {
    auto ptr = std::get<0>(*itr);
    delete[] ptr;
  }
  for (auto itr = test_data_.begin(); itr != test_data_.end(); ++itr) {
    auto ptr = std::get<0>(*itr);
    delete[] ptr;
  }
  for (auto itr = val_data_.begin(); itr != val_data_.end(); ++itr) {
    auto ptr = std::get<0>(*itr);
    delete[] ptr;
  }
}

int DataSet::Init(const std::string &data_base_directory, database_type type) {
  InitializeBMPFoldersDatabase(data_base_directory);
  return 0;
}

void DataSet::InitializeBMPFoldersDatabase(std::string dpath) {
  size_t file_size = 0;
  const int ratio = 5;
  auto vec = ReadDir(dpath);
  int running_index = 1;
  for (const auto ft : vec) {
    int label;
    std::string file_name;
    std::tie(label, file_name) = ft;
    char *data = ReadBitmapFile(file_name, &file_size);
    DataLabelTuple data_entry = std::make_tuple(data, label);
    if ((expected_data_size_ == 0) || (file_size == expected_data_size_)) {
      if (running_index % ratio == 0) {
        val_data_.push_back(data_entry);
      } else if (running_index % ratio == 1) {
        test_data_.push_back(data_entry);
      } else {
        train_data_.push_back(data_entry);
      }
      running_index++;
    }
  }
}

std::vector<FileTuple> DataSet::ReadDir(const std::string dpath) {
  std::vector<FileTuple> vec;
  struct dirent *entry = nullptr;
  num_of_classes_ = kClassNum;
  for (int class_id = 0; class_id < num_of_classes_; class_id++) {
    std::string dirname = dpath + "/" + std::to_string(class_id);
    DIR *dp = opendir(dirname.c_str());
    if (dp != nullptr) {
      while ((entry = readdir(dp))) {
        std::string filename = dirname + "/" + entry->d_name;
        if (filename.find(".bmp") != std::string::npos) {
          FileTuple ft = make_tuple(class_id, filename);
          vec.push_back(ft);
        }
      }
      closedir(dp);
    } else {
      std::cerr << "open directory: " << dirname << " failed." << std::endl;
    }
  }
  return vec;
}
