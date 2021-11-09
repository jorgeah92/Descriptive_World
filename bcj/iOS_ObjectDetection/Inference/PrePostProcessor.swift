//// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//import UIKit
import ColorKit // BCJ added package to project via project settings/package dependencies
import AVFoundation // BCJ

struct Prediction {
  let classIndex: Int
  let score: Float
  let rect: CGRect
}

extension CGSize {
    static func * (lhs: CGSize, rhs: CGFloat) -> CGSize {
        return CGSize(width: lhs.width * rhs, height: lhs.height * rhs)
    }
}

let synthesizer = AVSpeechSynthesizer() // BCJ
let patternInferencer = PatternDetector() // BCJ

class PrePostProcessor : NSObject {
    // model input image size
    static let inputWidth = 640
    static let inputHeight = 640
    static let patternWidth = 320
    static let patternHeight = 320

    // model output is of size 25200*85 for 80 classes, yolo-small
    //static let outputRow = 25200 // as decided by the YOLOv5 model for input image of size 640*640
    //static let outputColumn = 85 // left, top, right, bottom, score and 4 class probability
    // BCJ:  by examining output from detect.py prediction for df2.pt
    static let outputRow = 25200 // YOLOv5-small model output for input image of size 640*640
    static let outputColumn = 16 // 11-class probability + left, top, right, bottom, score
    static let outputPattern = 12 // 7-class probability + left, top, right, bottom, score
    static let patternRow = 6485
    static let threshold : Float = 0.1 //0.35 // score above which a detection is generated
    static let nmsLimit = 15 // max number of detections
    
    // The two methods nonMaxSuppression and IOU below are from  https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift
    /**
      Removes bounding boxes that overlap too much with other boxes that have
      a higher score.
      - Parameters:
        - boxes: an array of bounding boxes and their scores
        - limit: the maximum number of boxes that will be selected
        - threshold: used to decide whether boxes overlap too much
    */
    static func nonMaxSuppression(boxes: [Prediction], limit: Int, threshold: Float) -> [Prediction] {

      // Do an argsort on the confidence scores, from high to low.
      let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }

      var selected: [Prediction] = []
      var active = [Bool](repeating: true, count: boxes.count)
      var numActive = active.count

      // The algorithm is simple: Start with the box that has the highest score.
      // Remove any remaining boxes that overlap it more than the given threshold
      // amount. If there are any boxes left (i.e. these did not overlap with any
      // previous boxes), then repeat this procedure, until no more boxes remain
      // or the limit has been reached.
      outer: for i in 0..<boxes.count {
        if active[i] {
          let boxA = boxes[sortedIndices[i]]
          selected.append(boxA)
          if selected.count >= limit { break }

          for j in i+1..<boxes.count {
            if active[j] {
              let boxB = boxes[sortedIndices[j]]
              if IOU(a: boxA.rect, b: boxB.rect) > threshold {
                active[j] = false
                numActive -= 1
                if numActive <= 0 { break outer }
              }
            }
          }
        }
      }
      return selected
    }

    /**
      Computes intersection-over-union overlap between two bounding boxes.
    */
    static func IOU(a: CGRect, b: CGRect) -> Float {
      let areaA = a.width * a.height
      if areaA <= 0 { return 0 }

      let areaB = b.width * b.height
      if areaB <= 0 { return 0 }

      let intersectionMinX = max(a.minX, b.minX)
      let intersectionMinY = max(a.minY, b.minY)
      let intersectionMaxX = min(a.maxX, b.maxX)
      let intersectionMaxY = min(a.maxY, b.maxY)
      let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                             max(intersectionMaxX - intersectionMinX, 0)
      return Float(intersectionArea / (areaA + areaB - intersectionArea))
    }

    static func outputsToNMSPredictions(outputs: [NSNumber], imgScaleX: Double, imgScaleY: Double, ivScaleX: Double, ivScaleY: Double, startX: Double, startY: Double, patternType: Bool) -> [Prediction] {
        var classCount = outputColumn
        var rowCount = outputRow
        if (patternType) {
            classCount = outputPattern
            rowCount = patternRow
        }
        var predictions = [Prediction]()
        for i in 0..<rowCount {
            if Float(truncating: outputs[i*classCount+4]) > threshold {
                let x = Double(truncating: outputs[i*classCount])
                let y = Double(truncating: outputs[i*classCount+1])
                let w = Double(truncating: outputs[i*classCount+2])
                let h = Double(truncating: outputs[i*classCount+3])
                
                let left = imgScaleX * (x - w/2)
                let top = imgScaleY * (y - h/2)
                let right = imgScaleX * (x + w/2)
                let bottom = imgScaleY * (y + h/2)
                
                var max = Double(truncating: outputs[i*classCount+5])
                var cls = 0
                for j in 0 ..< classCount-5 {
                    if Double(truncating: outputs[i*classCount+5+j]) > max {
                        max = Double(truncating: outputs[i*classCount+5+j])
                        cls = j
                    }
                }

                let rect = CGRect(x: startX+ivScaleX*left, y: startY+top*ivScaleY, width: ivScaleX*(right-left), height: ivScaleY*(bottom-top))
                
                //let prediction = Prediction(classIndex: cls, score: Float(truncating: outputs[i*85+4]), rect: rect)
                let prediction = Prediction(classIndex: cls, score: Float(truncating: outputs[i*classCount+4]), rect: rect) // BCJ modified
                
                predictions.append(prediction)
            }
        }

        return nonMaxSuppression(boxes: predictions, limit: nmsLimit, threshold: threshold)
    }

    static func cleanDetection(imageView: UIImageView) {
        if let layers = imageView.layer.sublayers {
            for layer in layers {
                if layer is CATextLayer {
                    layer.removeFromSuperlayer()
                }
            }
            for view in imageView.subviews {
                view.removeFromSuperview()
            }
        }
    }
    
    static func cropImage(in imageView: UIImageView, rect: CGRect) -> UIImage {
        assert(imageView.contentMode == .scaleAspectFit)
        let image = imageView.image!

        let imageRatio = imageView.bounds.width / imageView.bounds.height
        let imageViewRatio = image.size.width / image.size.height

        let scale: CGFloat
        if imageRatio > imageViewRatio {
            scale = image.size.height / imageView.bounds.height
        } else {
            scale = image.size.width / imageView.bounds.width
        }

        // convert the `rect` into coordinates within the image, itself
        let size = rect.size * scale
        let origin = CGPoint(x: image.size.width  / 2 - (imageView.bounds.midX - rect.minX) * scale,
                             y: image.size.height / 2 - (imageView.bounds.midY - rect.minY) * scale)
        let scaledRect = CGRect(origin: origin, size: size)

        let format = UIGraphicsImageRendererFormat()
        format.scale = image.scale
        format.opaque = false

        return UIGraphicsImageRenderer(bounds: scaledRect, format: format).image { _ in
            image.draw(at: .zero)
        }
    }
    
    static func getBestPredictionIndex(predictions: [Prediction] ) -> Int {
        var min = Float.infinity
        var best_pred_idx = -1
        for (idx,el) in predictions.enumerated(){
            if el.score < min {
                min = el.score
                best_pred_idx = idx
            }
        }
        //print(best_pred_idx)
        return best_pred_idx
    }

    static func showDetection(imageView: inout UIImageView, nmsPredictions: [Prediction], classes: [String]) -> String {
        var utter_cnt = 0
        var utter_string = ""
        if (nmsPredictions.count<1) {
            let nada = "Nothing recognized."
            NSLog(nada)
            //return nada // skip speech for now
            let utterance = AVSpeechUtterance(string: nada)
            //utterance.volume = 0.5
            synthesizer.speak(utterance)
            return nada
        }
        var confidence_string = "I'm not sure, this might be a "

        let best_pred_idx = getBestPredictionIndex(predictions: nmsPredictions)
        let best_prediction = nmsPredictions[best_pred_idx]
        if best_prediction.score > 0.5 {
            confidence_string = "I think this is a "
        }
        if best_prediction.score > 0.8 {
            confidence_string = "This is a "
        }

        for pred in nmsPredictions {
            NSLog("score: " + String(format: "%.3f", pred.score))
            let bbox = UIView(frame: pred.rect)
            //print(pred.rect)
            //NSLog("bbox: " + NSCoder.string(for: bbox.frame))
            bbox.backgroundColor = UIColor.clear
            bbox.layer.borderColor = UIColor.orange.cgColor
            bbox.layer.borderWidth = 2
            imageView.addSubview(bbox)
            
            let textLayer = CATextLayer()
            let pred_garment = classes[pred.classIndex] // BCJ
            //textLayer.string = String(format: " %@ %.2f", pred_garment, pred.score).uppercased() // BCJ mod
            textLayer.string = String(format: " %@", pred_garment).uppercased() // BCJ mod
            textLayer.foregroundColor = UIColor.black.cgColor
            textLayer.backgroundColor = UIColor.orange.cgColor
            textLayer.fontSize = 14
            textLayer.frame = CGRect(x: pred.rect.origin.x, y: pred.rect.origin.y, width:100, height:20)
            imageView.layer.addSublayer(textLayer)
            
            //
            //
            //          FABRIC PATTERN DETECTION
            //
            //
            var pred_pattern = ""
            // WARNING:  the coordinate systems are different between UIImage and CGImage !!!!
            let crop_image = cropImage(in: imageView, rect: pred.rect)

            if (true) {
                let resizeDim = CGSize(width: CGFloat(PrePostProcessor.patternWidth), height: CGFloat(PrePostProcessor.patternHeight))
                let image = crop_image.resized(to: resizeDim)
                
                let imgScaleX = Double(image.size.width / CGFloat(PrePostProcessor.patternWidth));
                let imgScaleY = Double(image.size.height / CGFloat(PrePostProcessor.patternHeight));
                 
                let ivScaleX : Double = (image.size.width > image.size.height ? Double(imageView.frame.size.width / image.size.width) : Double(imageView.frame.size.height / image.size.height))
                let ivScaleY : Double = (image.size.height > image.size.width ? Double(imageView.frame.size.height / image.size.height) : Double(imageView.frame.size.width / image.size.width))

                let startX = Double((imageView.frame.size.width - CGFloat(ivScaleX) * image.size.width)/2)
                let startY = Double((imageView.frame.size.height -  CGFloat(ivScaleY) * image.size.height)/2)
                
                //imageView.image = image
                //NSLog(NSCoder.string(for: image.size))
                guard var pixelBuffer = image.normalized() else {
                     return ""
                 }
                
                 guard let outputs = patternInferencer.module.detectPattern(image: &pixelBuffer) else {
                    return ""
                }
                
                let nmsPredictions = PrePostProcessor.outputsToNMSPredictions(outputs: outputs, imgScaleX: imgScaleX, imgScaleY: imgScaleY, ivScaleX: ivScaleX, ivScaleY: ivScaleY, startX: startX, startY: startY, patternType: true)

                if (nmsPredictions.count<1) {
                    NSLog("No pattern detected")
                } else {
                    let best_pred_idx = getBestPredictionIndex(predictions: nmsPredictions)
                    let best_prediction = nmsPredictions[best_pred_idx]
                    pred_pattern = patternInferencer.classes[best_prediction.classIndex]
                    //NSLog("Detected " + pred_pattern)
                }
            }

            //
            //
            //          COLOR DETECTION
            //
            //
            
            var pred_color = ""
            if (true) {
                do {
                    //let dom_color = try image!.averageColor()
                    //pred_color = dom_color.name
                    let dom_color = try crop_image.dominantColors()//with: .high, algorithm: .kMeansClustering)
                    //print(dom_color.count)
                    if (dom_color.count > 0) {
                        let pred_color_long = dom_color[0].name().components(separatedBy: ".")
                        pred_color = pred_color_long[pred_color_long.count-1]
                    }
                    //NSLog(pred_color)
 
                } catch {
                    NSLog("Color detection error")
                }
            }
            
            //
            //
            //
            //
            
            
            if (utter_cnt==0) {
                utter_string += pred_color
                utter_string += " "
                utter_string += pred_pattern
                utter_string += " "
                utter_string += pred_garment
            } else {
                utter_string += ", and "
                utter_string += pred_color
                utter_string += " "
                utter_string += pred_pattern
                utter_string += " "
                utter_string += pred_garment
            }
            utter_cnt += 1
            
            //break // only process the first prediction
        }
        utter_string = confidence_string + utter_string
        let utterance = AVSpeechUtterance(string: utter_string)
        synthesizer.speak(utterance)
        NSLog(utter_string)
        
        return utter_string
        // end section BCJ
    }

}
