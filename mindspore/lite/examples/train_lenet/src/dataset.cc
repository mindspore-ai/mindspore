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
#include <arpa/inet.h>
#include <map>
#include <iostream>
#include <fstream>
#include <memory>
#include "src/utils.h"

using LabelId = std::map<std::string, int>;

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
}

int DataSet::Init(const std::string &data_base_directory, database_type type) {
  InitializeMNISTDatabase(data_base_directory);
  return 0;
}

void DataSet::InitializeMNISTDatabase(std::string dpath) {
  num_of_classes_ = 10;
  ReadMNISTFile(dpath + "/train/train-images-idx3-ubyte", dpath + "/train/train-labels-idx1-ubyte", &train_data_);
  ReadMNISTFile(dpath + "/test/t10k-images-idx3-ubyte", dpath + "/test/t10k-labels-idx1-ubyte", &test_data_);
}

int DataSet::ReadMNISTFile(const std::string &ifile_name, const std::string &lfile_name,
                           std::vector<DataLabelTuple> *dataset) {
  std::ifstream lfile(lfile_name, std::ios::binary);
  if (!lfile.is_open()) {
    std::cerr << "Cannot open label file " << lfile_name << std::endl;
    return 0;
  }

  std::ifstream ifile(ifile_name, std::ios::binary);
  if (!ifile.is_open()) {
    std::cerr << "Cannot open data file " << ifile_name << std::endl;
    return 0;
  }

  int magic_number = 0;
  lfile.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
  magic_number = ntohl(magic_number);
  if (magic_number != 2049) {
    std::cout << "Invalid MNIST label file!" << std::endl;
    return 0;
  }

  int number_of_labels = 0;
  lfile.read(reinterpret_cast<char *>(&number_of_labels), sizeof(number_of_labels));
  number_of_labels = ntohl(number_of_labels);

  ifile.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
  magic_number = ntohl(magic_number);
  if (magic_number != 2051) {
    std::cout << "Invalid MNIST image file!" << std::endl;
    return 0;
  }

  int number_of_images = 0;
  ifile.read(reinterpret_cast<char *>(&number_of_images), sizeof(number_of_images));
  number_of_images = ntohl(number_of_images);

  int n_rows = 0;
  ifile.read(reinterpret_cast<char *>(&n_rows), sizeof(n_rows));
  n_rows = ntohl(n_rows);

  int n_cols = 0;
  ifile.read(reinterpret_cast<char *>(&n_cols), sizeof(n_cols));
  n_cols = ntohl(n_cols);

  if (number_of_labels != number_of_images) {
    std::cout << "number of records in labels and images files does not match" << std::endl;
    return 0;
  }

  int image_size = n_rows * n_cols;
  unsigned char labels[number_of_labels];
  unsigned char data[image_size];
  lfile.read(reinterpret_cast<char *>(labels), number_of_labels);

  for (int i = 0; i < number_of_labels; ++i) {
    std::unique_ptr<float[]> hwc_bin_image(new (std::nothrow) float[32 * 32]);
    ifile.read(reinterpret_cast<char *>(data), image_size);

    for (size_t r = 0; r < 32; r++) {
      for (size_t c = 0; c < 32; c++) {
        if (r < 2 || r > 29 || c < 2 || c > 29)
          hwc_bin_image[r * 32 + c] = 0.0;
        else
          hwc_bin_image[r * 32 + c] = (static_cast<float>(data[(r - 2) * 28 + (c - 2)])) / 255.0;
      }
    }
    DataLabelTuple data_entry = std::make_tuple(reinterpret_cast<char *>(hwc_bin_image.release()), labels[i]);
    dataset->push_back(data_entry);
  }
  return number_of_labels;
}
