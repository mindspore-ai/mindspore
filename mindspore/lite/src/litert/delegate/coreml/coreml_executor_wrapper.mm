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

#include "src/litert/delegate/coreml/coreml_executor_wrapper.h"
#import "src/litert/delegate/coreml/coreml_executor.h"

namespace mindspore::lite {
CoreMLExecutorWrapper::CoreMLExecutorWrapper() {
  if (coreml_executor_ == nullptr) {
    //cast object-c ptr to c ptr, and transfer its ownership to a c object to avoid auto release
    coreml_executor_ = (__bridge_retained void*)[CoreMLExecutor new];
  }
}

CoreMLExecutorWrapper::~CoreMLExecutorWrapper() {
  //cast c ptr to object-c ptr, and transfer its ownership to an ARC object which is able to auto release
  auto arc_executor = (__bridge_transfer CoreMLExecutor*)coreml_executor_;
  (void)arc_executor;
  coreml_executor_ = nullptr;
}

int CoreMLExecutorWrapper::CompileMLModel(const std::string &modelPath) {
  mlmodel_path_ = modelPath;
  NSString *MLModelSrcPath = [NSString stringWithCString:modelPath.c_str() encoding:[NSString defaultCStringEncoding]];
  NSError *error = nil;
  NSURL *MLModelCURL = [MLModel compileModelAtURL:[NSURL fileURLWithPath:MLModelSrcPath] error:nil];
  if (error) {
    NSLog(@"Compile MLModel to MLModelC Error: %@", error);
    (void)CleanTmpFile();
    return RET_ERROR;
  }
  mlmodelc_path_ = [[MLModelCURL path] UTF8String];
  bool success = [(__bridge id)coreml_executor_ loadModelC:MLModelCURL];
  if (!success) {
    NSLog(@"Load MLModelC failed!");
    (void)CleanTmpFile();
    return RET_ERROR;
  }
  auto ret = CleanTmpFile();
  if (ret != RET_OK) {
    NSLog(@"Clean temp model file failed!");
  }
  return RET_OK;
}

int CoreMLExecutorWrapper::Run(const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors){
  auto success = [(__bridge id)coreml_executor_ ExecuteWithInputs:in_tensors outputs:out_tensors];
  if (!success) {
    NSLog(@"coreML model execute failed!");
    return RET_ERROR;
  }
  NSLog(@"coreML model execute success!");
  return RET_OK;
}

int CoreMLExecutorWrapper::CleanTmpFile() {
  NSError* error = nil;
  NSString *mlModelPath = [NSString stringWithCString:mlmodel_path_.c_str() encoding:[NSString defaultCStringEncoding]];
  NSString *mlModelCPath = [NSString stringWithCString:mlmodelc_path_.c_str() encoding:[NSString defaultCStringEncoding]];
  NSFileManager *fileManager = [NSFileManager defaultManager];
  bool isDir = NO;
  if ([fileManager fileExistsAtPath:mlModelPath isDirectory:&isDir] && isDir) {
    [fileManager removeItemAtPath:mlModelPath error:&error];
    if (error != nil) {
      NSLog(@"Failed cleaning up model: %@", [error localizedDescription]);
      return RET_ERROR;
    }
  }
  isDir = NO;
  if ([fileManager fileExistsAtPath:mlModelCPath isDirectory:&isDir] && isDir) {
    [fileManager removeItemAtPath:mlModelCPath error:&error];
    if (error != nil) {
      NSLog(@"Failed cleaning up compiled model: %@", [error localizedDescription]);
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
