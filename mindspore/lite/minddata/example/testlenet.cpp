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

using mindspore::dataset::Dataset;
using mindspore::dataset::Iterator;
using mindspore::dataset::Mnist;
using mindspore::dataset::TensorTransform;

int main(int argc, char **argv) {
  std::string folder_path = "./testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all");

  std::shared_ptr<TensorTransform> resize(new mindspore::dataset::vision::Resize({32, 32}));
  ds = ds->Map({resize});

  ds = ds->Shuffle(2);
  ds = ds->Batch(2);

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
