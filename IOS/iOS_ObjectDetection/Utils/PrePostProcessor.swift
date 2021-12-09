//// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//import UIKit
import ColorKit // BCJ added package to project via project settings/package dependencies
import AVFoundation // BCJ
import UIKit

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

extension UIImage {
    var hasAlpha: Bool {
        guard let alphaInfo = self.cgImage?.alphaInfo else {return false}
        return alphaInfo != CGImageAlphaInfo.none &&
            alphaInfo != CGImageAlphaInfo.noneSkipFirst &&
            alphaInfo != CGImageAlphaInfo.noneSkipLast
    }
    func removingAlpha() -> UIImage {
      let format = UIGraphicsImageRendererFormat()
      format.opaque = true // removes Alpha Channel
      format.scale = scale // keeps original image scale
      return UIGraphicsImageRenderer(size: size, format: format).image { _ in
        draw(in: CGRect(origin: .zero, size: size))
      }
    }
    func scalePreservingAspectRatio(targetSize: CGSize, stride: Int) -> UIImage {
        // using openCV for Swift found at:  https://github.com/vvmnnnkv/SwiftCV
        // Determine the scale factor that preserves aspect ratio
        let widthRatio = targetSize.width / size.width
        let heightRatio = targetSize.height / size.height
        let scaleFactor = min(widthRatio, heightRatio)
        let scaledImageSize = CGSize(
            width: size.width * scaleFactor,
            height: size.height * scaleFactor
        )
        var dw = Int(targetSize.width) - Int(scaledImageSize.width) // w padding
        var dh = Int(targetSize.height) - Int(scaledImageSize.height) // h padding
        dw = dw % stride
        dh = dh % stride
        let dw_d = Double(dw) / 2.0
        let dh_d = Double(dh) / 2.0
        let top = Int(round(dh_d - 0.1))
        let bottom = Int(round(dh_d + 0.1))
        let left = Int(round(dw_d - 0.1))
        let right = Int(round(dw_d + 0.1))

        var scaledImage = self.resized(to: scaledImageSize)
        scaledImage = scaledImage.withPadding(x: CGFloat(Float(top)), y: CGFloat(Float(left)))!
        return scaledImage
    }
    func withPadding(x: CGFloat, y: CGFloat) -> UIImage? {
        let newWidth = size.width + 2 * x
        let newHeight = size.height + 2 * y
        let newSize = CGSize(width: newWidth, height: newHeight)
        UIGraphicsBeginImageContextWithOptions(newSize, false, 0)
        let origin = CGPoint(x: (newWidth - size.width) / 2, y: (newHeight - size.height) / 2)
        draw(at: origin)
        let imageWithPadding = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return imageWithPadding
    }
    func scalePreservingAspectRatioOld(targetSize: CGSize) -> UIImage {
            // Determine the scale factor that preserves aspect ratio
            let widthRatio = targetSize.width / size.width
            let heightRatio = targetSize.height / size.height
            let scaleFactor = min(widthRatio, heightRatio)
            let scaledImageSize = CGSize(
                width: size.width * scaleFactor,
                height: size.height * scaleFactor
            )
            let renderer = UIGraphicsImageRenderer(
                size: scaledImageSize
            )
            let scaledImage = renderer.image { _ in
                self.draw(in: CGRect(
                    origin: .zero,
                    size: scaledImageSize
                ))
            }
            return scaledImage
        }
}

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
    static let patternRow = 6300 // 6300 for 320x320, 25200 for 640x640
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
            print(idx)
            print(el)
            if el.score < min {
                min = el.score
                best_pred_idx = idx
            }
        }
        //print(best_pred_idx)
        return best_pred_idx
    }
    
    static func whichColor(color: UIColor) -> String{
        var (h,s,b,a) : (CGFloat, CGFloat, CGFloat, CGFloat) = (0,0,0,0)
        _ = color.getHue(&h, saturation: &s, brightness: &b, alpha: &a)
        //print("HSB range- h: \(h), s: \(s), v: \(b)")
        var colorTitle = ""
        switch (h, s, b) {
            case (0...0.040, 0.05...0.50, 0.10...1.00):
                colorTitle = "brown"
            case (0...0.040, 0.51...1.00, 0.10...1.00):
                colorTitle = "red"
            case (0.041...0.111, 0.05...0.50, 0.10...1.00):
                colorTitle = "beige"
            case (0.041...0.111, 0.51...1.00, 0.10...1.00):
                colorTitle = "orange"
            case (0.112...0.222, 0.05...0.50, 0.10...1.00):
                colorTitle = "olive"
            case (0.112...0.222, 0.51...1.00, 0.10...1.00):
                colorTitle = "yellow"
            case (0.223...0.444, 0.03...0.49, 0.10...1.00):
                colorTitle = "olive"
            case (0.223...0.444, 0.05...1.00, 0.10...1.00):
                colorTitle = "green"
            case (0.445...0.542, 0.05...1.00, 0.10...1.00):
                colorTitle = "teal"
            case (0.543...0.750, 0.05...1.00, 0.10...1.00):
                colorTitle = "blue"
            case (0.751...0.778, 0.05...1.00, 0.10...1.00):
                colorTitle = "purple"
            case (0.779...0.889, 0.05...0.50, 0.10...1.00):
                colorTitle = "burgandy"
            case (0.779...0.889, 0.51...1.00, 0.10...1.00):
                colorTitle = "pink"
            case (0.890...1.00, 0.05...0.50, 0.10...1.00):
                colorTitle = "maroon"
            case (0.890...1.00, 0.51...1.00, 0.10...1.00):
                colorTitle = "red"
            case (0...1.00, 0...0.05, 0.80...1.00):
                colorTitle = "white"
            case (0...1.00, 0.1...1.00, 0.20...0.80):
                colorTitle = "grey"
            case (0...1.00, 0...1.00, 0...0.20):
                colorTitle = "black"
            default:
                NSLog("Color didn't fit defined ranges...")
        }
        return colorTitle
    }
    
    static func predictPattern(imageView: inout UIImageView, pred: Prediction) -> (pred_text: String, crop_img: UIImage) {
        //          FABRIC PATTERN DETECTION
        var pred_pattern = ""
        // WARNING:  the coordinate systems are different between UIImage and CGImage !!!!
        var crop_image = cropImage(in: imageView, rect: pred.rect)

        if (false) { // based on manual testing, this does not seem to produce better results
            // re-crop the detected garment to focus on a smaller portion of the interior
            // move the center-y to start more toward top of image for pants, etc.
            let small_rect = CGRect(x: CGFloat(crop_image.cgImage!.width)*0.3, y: CGFloat(crop_image.cgImage!.height)*0.1,
                                    width: CGFloat(crop_image.cgImage!.width) * 0.6, height: CGFloat(crop_image.cgImage!.height) * 0.6)
            let crop_ImageView = UIImageView(image: crop_image)
            crop_ImageView.frame = CGRect(x: 0, y: 0, width: CGFloat(crop_image.cgImage!.width) * 1.2, height: CGFloat(crop_image.cgImage!.height) * 1.2)
            crop_ImageView.contentMode = UIView.ContentMode.scaleAspectFit
            crop_image = cropImage(in: crop_ImageView, rect: small_rect)
        }
        print("detecting pattern")
        let resizeDim = CGSize(width: CGFloat(PrePostProcessor.patternWidth), height: CGFloat(PrePostProcessor.patternHeight))
        let image = crop_image.resized(to: resizeDim)
        //The next statement was causing the algorithm to not recognize any patterns
//        let image = crop_image.scalePreservingAspectRatio(targetSize: CGSize(width: patternWidth, height: patternHeight), stride: 32)
        //imageView.image = image // uncomment this line to see what the fabric and color detection actually sees                NSLog(NSCoder.string(for: image.size))
        let imgScaleX = Double(image.size.width / CGFloat(PrePostProcessor.patternWidth));
        let imgScaleY = Double(image.size.height / CGFloat(PrePostProcessor.patternHeight));
         
        let ivScaleX : Double = (image.size.width > image.size.height ? Double(imageView.frame.size.width / image.size.width) : Double(imageView.frame.size.height / image.size.height))
        let ivScaleY : Double = (image.size.height > image.size.width ? Double(imageView.frame.size.height / image.size.height) : Double(imageView.frame.size.width / image.size.width))

        let startX = Double((imageView.frame.size.width - CGFloat(ivScaleX) * image.size.width)/2)
        let startY = Double((imageView.frame.size.height -  CGFloat(ivScaleY) * image.size.height)/2)
        NSLog(NSCoder.string(for: image.size))
        guard var pixelBuffer = image.normalized() else {
             return ("", crop_image)
         }
        
         guard let outputs = patternInferencer.module.detectPattern(image: &pixelBuffer) else {
            return ("", crop_image)
        }
        
        let nmsPredictions = PrePostProcessor.outputsToNMSPredictions(outputs: outputs, imgScaleX: imgScaleX, imgScaleY: imgScaleY, ivScaleX: ivScaleX, ivScaleY: ivScaleY, startX: startX, startY: startY, patternType: true)

        if (nmsPredictions.count < 1) {
            NSLog("No pattern detected")
        } else {
            let best_pred_idx = getBestPredictionIndex(predictions: nmsPredictions)
            let best_prediction = nmsPredictions[best_pred_idx]
            pred_pattern = patternInferencer.classes[best_prediction.classIndex]
            //NSLog("Detected " + pred_pattern)
        }
        return (pred_pattern, crop_image)
    }

    static func predictColor(crop_image: UIImage) -> String {
        //          COLOR DETECTION
        var pred_color = ""
        do {
            //
            // VERSION 1: almost always gray
            //
            //print("avg_color")
            let dom_color = try crop_image.averageColor() // uncomment this line to use with which_color
            //print(dom_color.name())
            //let pred_color_tmp = dom_color.name()

            //
            // VERSION 2: almost always black
            //
            //print("dom_color")
            // default algorithm returns dominant color as first result
            //let dom_colors = try crop_image.dominantColors(with: .best)//, algorithm: .kMeansClustering) // uncomment this line to use with which_color
            /*for dom_col in dom_colors {
                print(dom_col)
                print(dom_col.name())
            }
            if (dom_colors.count > 0) {
                let pred_color_tmp = dom_colors[0].name()
                let pred_color_long = pred_color_tmp.components(separatedBy: ".")
                pred_color = pred_color_long[pred_color_long.count-1]
            } else {
                print("no dominant color found")
            }
             */
            //NSLog(pred_color)
            
            //
            // VERSION 3: decent results
            //
            //print("which_color")
            //use these 2 lines with averagecolor
            pred_color = whichColor(color: dom_color)
            print(pred_color)
            
            //use these 2 lines with dominantcolors
            /*if (dom_colors.count > 0) {
                pred_color = whichColor(color: dom_colors[0])
                print(pred_color)
            } else {
                print("no which_color found")
            }*/

        } catch {
            NSLog("Color detection error")
        }
        return pred_color
    }
    
    static func showDetection(imageView: inout UIImageView, nmsPredictions: [Prediction], classes: [String], detectType: String = "all", synth: AVSpeechSynthesizer) -> String {
        var utter_cnt = 0
        var utter_string = ""
        if (nmsPredictions.count < 1) {
            let nada = "Nothing recognized."
            NSLog(nada)
            //return nada // skip speech for now
            try? audioSession.setMode(AVAudioSession.Mode.spokenAudio)
            synth.speak(AVSpeechUtterance(string: nada))
            return nada
        }
        var confidence_string = "I'm not sure, this might be "

        let best_pred_idx = getBestPredictionIndex(predictions: nmsPredictions)
        let best_prediction = nmsPredictions[best_pred_idx]
        if best_prediction.score > 0.5 {
            confidence_string = "I think this is "
        }
        if best_prediction.score > 0.8 {
            confidence_string = "This is "
        }
        if ((detectType == "all") || (detectType == "garment")) {
            confidence_string += "a "
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
            
            // add garment class in top left
            let textLayer = CATextLayer()
            let pred_garment = classes[pred.classIndex] // BCJ
            textLayer.string = String(format: " %@", pred_garment).uppercased() // BCJ mod
            textLayer.foregroundColor = UIColor.black.cgColor
            textLayer.backgroundColor = UIColor.orange.cgColor
            textLayer.fontSize = 14
            textLayer.frame = CGRect(x: pred.rect.origin.x, y: pred.rect.origin.y, width:100, height:20)
            imageView.layer.addSublayer(textLayer)

            // add garment class confidence in lower left
            let confLayer = CATextLayer()
            confLayer.string = String(format: " %.2f", pred.score) // BCJ mod
            confLayer.foregroundColor = UIColor.black.cgColor
            confLayer.backgroundColor = UIColor.orange.cgColor
            confLayer.fontSize = 14
            confLayer.frame = CGRect(x: pred.rect.origin.x+pred.rect.width-37, y: pred.rect.origin.y+pred.rect.height-22, width:35, height:20)
            imageView.layer.addSublayer(confLayer)

            let detectFabric = true
            var pred_pattern = ""
            var crop_image = UIImage()
            if (detectFabric) {
                (pred_pattern, crop_image) = predictPattern(imageView: &imageView, pred: pred)
            }

            let detectColor = true
            var pred_color = ""
            if (detectColor) {
                pred_color = predictColor(crop_image: crop_image)
            }
            
            //imageView.image = crop_image // uncomment this line to see what the fabric and color detection actually sees
            
            if ((utter_cnt > 0) && ((detectType == "all") || (detectType == "garment"))) {
                utter_string += ", and "
            }
            if (detectColor && (pred_color > "") && (detectType != "pattern") && (detectType != "garment")) {
                utter_string += pred_color
                utter_string += " "
            }
            if (detectColor && (pred_color == "") && (detectType == "color")) {
                utter_string = "I don't know"
            }
            
            if (detectFabric && (pred_pattern > "") && (detectType != "color") && (detectType != "garment")) {
                utter_string += pred_pattern
                utter_string += " "
            }
            if (detectColor && (pred_pattern == "") && (detectType == "pattern")) {
                utter_string = "I don't know"
            }
            
            if ((detectType == "all") || (detectType == "garment")) {
                utter_string += pred_garment
            }
            utter_cnt += 1
            
            if ((detectType != "all") && (detectType != "garment")) {
                break // only process the first, main, prediction if only detecting pattern or color
            }
            //break // uncomment this line to only process the first prediction
        }
        utter_string = confidence_string + utter_string
        try? audioSession.setMode(AVAudioSession.Mode.spokenAudio)
        synth.speak(AVSpeechUtterance(string: utter_string))
        NSLog(utter_string)
        
        return utter_string
        // end section BCJ
    }

}
