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

#include <jni.h>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/util/path.h"
#if defined(__ANDROID__) || defined(ANDROID)
#include <android/log.h>
#include <android/asset_manager.h>
#endif

extern "C" JNIEXPORT jstring JNICALL Java_com_example_mindsporepredict_MainActivity_stringFromJNI(JNIEnv *env,
                                                                                                  jobject /* this */) {
  std::string hello = "Hello World!";
  MS_LOG(DEBUG) << hello;
  return env->NewStringUTF(hello.c_str());
}

using Dataset = mindspore::dataset::Dataset;
using Iterator = mindspore::dataset::Iterator;
using mindspore::dataset::Cifar10;
using mindspore::dataset::Path;
using mindspore::dataset::RandomSampler;
using mindspore::dataset::Tensor;

extern "C" JNIEXPORT void JNICALL Java_com_example_mindsporepredict_MainActivity_pathTest(JNIEnv *env,
                                                                                          jobject /* this */,
                                                                                          jstring path) {
  MS_LOG(WARNING) << env->GetStringUTFChars(path, 0);
  Path f(env->GetStringUTFChars(path, 0));
  MS_LOG(WARNING) << f.Exists() << f.IsDirectory() << f.ParentPath();
  // Print out the first few items in the directory
  auto dir_it = Path::DirIterator::OpenDirectory(&f);
  MS_LOG(WARNING) << dir_it.get();
  int i = 0;
  while (dir_it->hasNext()) {
    Path v = dir_it->next();
    MS_LOG(WARNING) << v.toString();
    i++;
    if (i > 5) break;
  }
}

extern "C" JNIEXPORT void JNICALL Java_com_example_mindsporepredict_MainActivity_TestCifar10Dataset(JNIEnv *env,
                                                                                                    jobject /* this */,
                                                                                                    jstring path) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10Dataset.";

  // Create a Cifar10 Dataset
  std::string folder_path = env->GetStringUTFChars(path, 0);
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, std::string(), RandomSampler(false, 10));

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  // Manually terminate the pipeline
  iter->Stop();
}
