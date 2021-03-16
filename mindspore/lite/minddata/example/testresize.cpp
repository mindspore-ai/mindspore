/**
 * Copyright 2021  Huawei Technologies Co., Ltd
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

#include "include/datasets.h"
#include "include/iterator.h"
#include "include/vision_lite.h"
#include "include/transforms.h"
#include "include/api/types.h"

using mindspore::dataset::Album;
using mindspore::dataset::Dataset;
using mindspore::dataset::Iterator;
using mindspore::dataset::SequentialSampler;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::vision::ResizePreserveAR;


int main(int argc, char **argv) {
  std::string folder_path = "./testAlbum/images";
  std::string schema_file = "./testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"image", "label", "id"};

  // Create a Album Dataset
  std::shared_ptr<Dataset> ds =
    Album(folder_path, schema_file, column_names, true, std::make_shared<SequentialSampler>(0, 1));
  ds = ds->SetNumWorkers(1);

  std::shared_ptr<TensorTransform> resize(new ResizePreserveAR(1000, 1000));
  ds = ds->Map({resize}, {"image"}, {"image", "ratio", "invM"});

  std::shared_ptr<Iterator> iter = ds->CreateIterator();

  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  iter->Stop();
}
