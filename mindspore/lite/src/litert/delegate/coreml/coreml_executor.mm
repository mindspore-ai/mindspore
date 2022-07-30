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

#import "src/litert/delegate/coreml/coreml_executor.h"
#include <fstream>
#include <iostream>

namespace {
// The subgraph split can cause the change of tensor name. This function is used to get the original name.
std::string GetOrgFeatureName(const std::string &input_name) {
  auto org_name = input_name;
  std::string pattern_1 = "_duplicate_";
  auto pos_1 = input_name.find(pattern_1);
  if (pos_1 != std::string::npos) {
    org_name = input_name.substr(pos_1 + pattern_1.length());
    return org_name;
  }
  std::string pattern_2 = "_duplicate";
  auto pos_2 = input_name.find(pattern_2);
  if (pos_2 != std::string::npos) {
    org_name = input_name.substr(0, pos_2);
    return org_name;
  }
  return org_name;
}
}  // namespace

@implementation InputFeatureProvider

- (instancetype)initWithInputs:(const std::vector<mindspore::MSTensor>*)inputs
                 coreMLVersion:(int)coreMLVersion {
  self = [super init];
  _inputs = inputs;
  _coreMLVersion = coreMLVersion;
  NSMutableArray* names = [[NSMutableArray alloc] init];
  for (auto& input : *_inputs) {
    auto input_name = GetOrgFeatureName(input.Name());
    [names addObject:[NSString stringWithCString:input_name.c_str()
                                        encoding:[NSString defaultCStringEncoding]]];
  }
  _featureNames = [NSSet setWithArray:names];
  return self;
}

- (NSSet<NSString*>*)featureNames{ return _featureNames; }

- (MLFeatureValue*)featureValueForName:(NSString*)featureName {
  for (auto input : *_inputs) {
    auto input_name = GetOrgFeatureName(input.Name());
    if ([featureName cStringUsingEncoding:NSUTF8StringEncoding] == input_name) {
      NSArray* shape;
      NSArray* strides;
      int tensorRank = input.Shape().size();
      switch(tensorRank) {
        case 1:
          shape = @[
            @(input.Shape()[0])
          ];
          strides = @[
            @1
          ];
          break;
        case 2:
          shape = @[
            @(input.Shape()[0]),
            @(input.Shape()[1])
          ];
          strides = @[
            @(input.Shape()[1]),
            @1
          ];
          break;
        case 3:
          shape = @[
            @(input.Shape()[0]),
            @(input.Shape()[1]),
            @(input.Shape()[2])
          ];
          strides = @[
            @(input.Shape()[2] * input.Shape()[1]),
            @(input.Shape()[2]),
            @1
          ];
          break;
        case 4:
          shape = @[
            @(input.Shape()[0]),
            @(input.Shape()[1]),
            @(input.Shape()[2]),
            @(input.Shape()[3])
          ];
          strides = @[
            @(input.Shape()[3] * input.Shape()[2] * input.Shape()[1]),
            @(input.Shape()[3] * input.Shape()[2]),
            @(input.Shape()[3]),
            @1
          ];
          break;
        default:
          NSLog(@"The rank of input tensor:%@ is unsupported!", featureName);
      }

      NSError* error = nil;
      MLMultiArray* mlArray = [[MLMultiArray alloc] initWithDataPointer:(float*)input.MutableData()
                                                                  shape:shape
                                                               dataType:MLMultiArrayDataTypeFloat32
                                                                strides:strides
                                                            deallocator:(^(void* bytes){
                                                                        })error:&error];
      if (error != nil) {
        NSLog(@"Failed to create MLMultiArray for input tensor %@ error: %@!", featureName,
              [error localizedDescription]);
        return nil;
      }
      auto* mlFeatureValue = [MLFeatureValue featureValueWithMultiArray:mlArray];
      return mlFeatureValue;
    }
  }

  NSLog(@"Input tensor %@ not found!", featureName);
  return nil;
}
@end

@implementation CoreMLExecutor

- (bool)ExecuteWithInputs:(const std::vector<mindspore::MSTensor>&)inputs
                 outputs:(const std::vector<mindspore::MSTensor>&)outputs {
  if (_model == nil) {
    return NO;
  }
  _coreMLVersion = 3;
  NSError* error = nil;
  //Initialize the CoreML feature provider with input MSTensor
  InputFeatureProvider* inputFeature =
      [[InputFeatureProvider alloc] initWithInputs:&inputs coreMLVersion:[self coreMLVersion]];
  if (inputFeature == nil) {
    NSLog(@"inputFeature initialization failed.");
    return NO;
  }
  //Inference configuration, auto use GPU by default
  MLPredictionOptions* options = [[MLPredictionOptions alloc] init];

  //inference with specific input
  id<MLFeatureProvider> outputFeature = [_model predictionFromFeatures:inputFeature
                                                               options:options
                                                                 error:&error];
  if (error != nil) {
    NSLog(@"Execute model failed, error code: %@", [error localizedDescription]);
    return NO;
  }
  NSSet<NSString*>* outputFeatureNames = [outputFeature featureNames];
  for (auto output : outputs) {
    auto orgOutputName = GetOrgFeatureName(output.Name());
    NSString* outputName = [NSString stringWithCString:orgOutputName.c_str()
                                              encoding:[NSString defaultCStringEncoding]];
    MLFeatureValue* outputValue =
        [outputFeature featureValueForName:[outputFeatureNames member:outputName]];
    auto* data = [outputValue multiArrayValue];
    float* outputData = (float*)data.dataPointer;
    if (outputData == nullptr) {
      NSLog(@"Output data is null!");
      return NO;
    }
    memcpy(output.MutableData(), outputData, output.DataSize());
  }
  return YES;
}

- (bool)loadModelC:(NSURL*)compileUrl {
  NSError* error = nil;
  if (@available(iOS 12.0, *)) {
    MLModelConfiguration* config = [MLModelConfiguration alloc];
    config.computeUnits = MLComputeUnitsAll;
    _model = [MLModel modelWithContentsOfURL:compileUrl configuration:config error:&error];
  } else {
    _model = [MLModel modelWithContentsOfURL:compileUrl error:&error];
  }
  if (error != NULL) {
    NSLog(@"Create MLModel failed, error code: %@", [error localizedDescription]);
    return NO;
  }
  return YES;
}
@end
