// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
// BCJ modified on Oct 25, 2021
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#import "InferenceModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>

// 640x640 is the default image size used in the export.py in the yolov5 repo to export the TorchScript model, 25200*85 is the model output size
const int input_width = 640;
const int input_height = 640;
const int pattern_width = 320;
const int pattern_height = 320;
//const int output_size = 25200*85;
//const int output_size = 25200*9; // BCJ - determined by inspecting the shape of the prediction from detect.py for df2.pt for 4-class small model
const int output_size = 25200*16; // 25200 for 640x640 BCJ - determined by inspecting the shape of the prediction from detect.py for df2_all.pt for 11-class small model
const int pattern_size = 6300*12;//6300*12 for 320x320; // BCJ - 7 classes of patterns

@implementation InferenceModule {
    @protected torch::jit::mobile::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (NSArray<NSNumber*>*)detectImage:(void*)imageBuffer {
    try {
        at::Tensor tensor = torch::from_blob(imageBuffer, { 1, 3, input_width, input_height }, at::kFloat);

        c10::InferenceMode guard;
//        for (int i = 0; i < tensor.sizes().size(); i++) {
//            NSLog(@"tensor dim %i: %.2lld", i, tensor.sizes()[i]);
//        }
        CFTimeInterval startTime = CACurrentMediaTime();
        auto outputTuple = _impl.forward({ tensor }).toTuple();
        CFTimeInterval elapsedTime = CACurrentMediaTime() - startTime;
        NSLog(@"garment inf time:%.2f", elapsedTime);

        auto outputTensor = outputTuple->elements()[0].toTensor();

        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        
        NSMutableArray* results = [[NSMutableArray alloc] init];
        
        for (int i = 0; i < output_size; i++) {
          //NSLog(@"%i", i);
          [results addObject:@(floatBuffer[i])];
        }

        return [results copy];
        
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

- (NSArray<NSNumber*>*)detectPattern:(void*)imageBuffer {
    try {
        at::Tensor tensor = torch::from_blob(imageBuffer, { 1, 3, pattern_width, pattern_height }, at::kFloat);
        c10::InferenceMode guard;
//        for (int i = 0; i < tensor.sizes().size(); i++) {
//            NSLog(@"tensor dim %i: %.2lld", i, tensor.sizes()[i]);
//        }
        CFTimeInterval startTime = CACurrentMediaTime();
        auto outputTuple = _impl.forward({ tensor }).toTuple();
        CFTimeInterval elapsedTime = CACurrentMediaTime() - startTime;
        NSLog(@"pattern inf time:%.2f", elapsedTime);
        auto outputTensor = outputTuple->elements()[0].toTensor();

        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        
        NSMutableArray* results = [[NSMutableArray alloc] init];
        
        for (int i = 0; i < pattern_size; i++) {
          //NSLog(@"%i", i);
          [results addObject:@(floatBuffer[i])];
        }

        return [results copy];
        
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}
@end
