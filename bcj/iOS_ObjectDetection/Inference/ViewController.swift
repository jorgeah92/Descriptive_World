// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import UIKit

// start BCJ
import AVFoundation
import Speech
import NaturalLanguage
// end BCJ

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate{
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var txtVoice: UITextView! // BCJ
    @IBOutlet weak var camView: UIView! // BCJ
    @IBOutlet weak var btnRun: UIButton!
    @IBOutlet weak var btnVoice: UIButton!
    @IBOutlet weak var btnCamera: UIButton!
    @IBOutlet weak var btnNext: UIButton!
    
    private let testImages = ["test1.png", "test2.jpg", "test3.png"]
    private var imgIndex = 0

    private var image : UIImage?
    private var inferencer = ObjectDetector()
    
    // start BCJ
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))!
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private let audioSession = AVAudioSession.sharedInstance()
    
    private var videoDataOutput: AVCaptureVideoDataOutput!
    private var videoDataOutputQueue: DispatchQueue!
    private var previewLayer:AVCaptureVideoPreviewLayer!
    private var captureDevice : AVCaptureDevice!
    private let session = AVCaptureSession()
    
    private var camFrame: [Float32]?
    private var camImage: UIImage!
    
    private var voice_timer: Timer!
    // end BCJ
    
    override func viewDidLoad() {
        super.viewDidLoad()
        image = UIImage(named: testImages[imgIndex])!
        if let iv = imageView {
            iv.image = image
            btnRun.setTitle("Detect", for: .normal)
        }
        // start BCJ
        let utterance = AVSpeechUtterance(string: "Descriptive World")
        //utterance.volume = 0.5
        let synthesizer = AVSpeechSynthesizer()
        synthesizer.speak(utterance)
        // end BCJ
    }
    
    @IBAction func runTapped(_ sender: Any) {
        detectObject()
    }

    @IBAction func nextTapped(_ sender: Any) {
        PrePostProcessor.cleanDetection(imageView: imageView)
        imgIndex = (imgIndex + 1) % testImages.count
        btnNext.setTitle(String(format: "Test Image %d/%d", imgIndex + 1, testImages.count), for:.normal)
        image = UIImage(named: testImages[imgIndex])!
        imageView.image = image
    }

    @IBAction func photosTapped(_ sender: Any) {
        PrePostProcessor.cleanDetection(imageView: imageView)
        let imagePickerController = UIImagePickerController()
        imagePickerController.delegate = self;
        imagePickerController.sourceType = .photoLibrary
        self.present(imagePickerController, animated: true, completion: nil)
    }
    
    @IBAction func cameraTapped(_ sender: Any) {
        PrePostProcessor.cleanDetection(imageView: imageView)
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            let imagePickerController = UIImagePickerController()
            imagePickerController.delegate = self;
            imagePickerController.sourceType = .camera
            self.present(imagePickerController, animated: true, completion: nil)
        }
    }
    
    private func voiceTapped() {
        PrePostProcessor.cleanDetection(imageView: self.imageView)
        if (self.btnVoice.title(for: .normal) == "Voice") {
            self.btnVoice.setTitle("Tap to stop", for: .normal)
            self.camView.isHidden = false
            self.imageView.isHidden = true
            self.setupAVCapture()
        } else {
            //DispatchQueue.main.async {
            self.imageView.image = self.camImage
            //}
            self.btnVoice.setTitle("Voice", for: .normal)
            self.stopCamera()
            self.camView.isHidden = true
            self.imageView.isHidden = false
        }
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        image = info[UIImagePickerController.InfoKey.originalImage] as? UIImage
        image = image!.resized(to: CGSize(width: CGFloat(PrePostProcessor.inputWidth), height: CGFloat(PrePostProcessor.inputHeight)*image!.size.height/image!.size.width))
        imageView.image = image
        self.dismiss(animated: true, completion: nil)
    }
    
    // start BCJ
    override public func viewDidAppear(_ animated: Bool) {
        checkPermissions()
    }
    
    private func checkPermissions() {
        SFSpeechRecognizer.requestAuthorization { authStatus in
            DispatchQueue.main.async {
                switch authStatus {
                case .authorized: break
                default: self.handlePermissionFailed()
                }
            }
        }
    }

    @IBAction func startVoice(_ sender: Any) {
        NSLog("begin startVoice")
        voiceTapped()
        if (self.btnVoice.title(for: .normal)=="Voice") {
            return
        }
        do {
            self.txtVoice.text = ""
            
            try audioSession.setCategory(.playAndRecord, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
            let inputNode = audioEngine.inputNode

            // Configure the microphone input.
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            inputNode.removeTap(onBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { (buffer: AVAudioPCMBuffer, when: AVAudioTime) in
                self.recognitionRequest?.append(buffer)
            }

            audioEngine.prepare()
            try audioEngine.start()
            
            // Create and configure the speech recognition request.
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            guard let recognitionRequest = recognitionRequest else { fatalError("Unable to create a SFSpeechAudioBufferRecognitionRequest object") }
            recognitionRequest.shouldReportPartialResults = true
            
            self.voice_timer = Timer.scheduledTimer(timeInterval: 2, target: self, selector: #selector(self.processVoice), userInfo: nil, repeats: false)
            
            // Setup a recognition task for the speech recognition session.
            // Keep a reference to the task so that it can be canceled.
            self.recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { result, error in
                var isFinal = false
                //NSLog("recognizing...")
                if let result = result {
                    self.txtVoice.text = result.bestTranscription.formattedString
                    //NSLog(result.bestTranscription.formattedString)
                    isFinal = result.isFinal
                }
                if isFinal {
                    NSLog("is final")
                    self.audioEngine.stop()
                    inputNode.removeTap(onBus: 0)
                    self.recognitionTask!.cancel()
                    self.recognitionRequest = nil
                } else if (error == nil) {
                    NSLog("reset timer")
                    self.voice_timer?.invalidate()
                    self.voice_timer = Timer.scheduledTimer(timeInterval: 2, target: self, selector: #selector(self.processVoice), userInfo: nil, repeats: false)
                }
                
            }
            
        } catch {
            NSLog("error in startVoice")
        }
        NSLog("end startVoice")
    }

    @IBAction func endVoice(_ sender: Any) {
    }
    
    @objc private func processVoice() {
        NSLog("begin processVoice")

        if let sentenceEmbedding = NLEmbedding.sentenceEmbedding(for: .english) {
            let sentence = self.txtVoice.text.lowercased()+""
            //NSLog(sentence)

            //if let vector = sentenceEmbedding.vector(for: sentence) {
                //print(vector)
            //}
            var commandScores = [Double]()
            let knownCommands = ["what is this",
                                 "tell me what this is",
                                 "color",
                                 "what color is this",
                                 "tell me which color this is",
                                 "pattern",
                                 "what pattern is this",
                                 "tell me which pattern this is",
                                 //"is this a [item]",
                                 //"is this [color]",
                                 //"is this [pattern]",
                                ]
            for commandOption in knownCommands {
                let distance = sentenceEmbedding.distance(between: sentence, and: commandOption)
                commandScores.append(distance)
                //let score = String(format: "%.3f", distance)
                //NSLog("score:"+commandOption+", "+score)
            }
            let bestScore = commandScores.min()
            if (bestScore! < 0.5) {
                self.voiceTapped()
                self.audioEngine.stop()
                self.audioEngine.inputNode.removeTap(onBus: 0)
                self.recognitionTask!.cancel()
                self.recognitionRequest = nil
                //try? self.audioSession.setActive(false)

                let bestCommand = commandScores.firstIndex(of: bestScore!)
                switch bestCommand! {
                case 0 ... 1:
                    self.detectObject()
                case 2 ... 4:
                    self.txtVoice.text = self.txtVoice.text + "? I don't know how to do that yet."
                    NSLog("detect color to be implemented")
                    break
                    //self.detectColor()
                case 5 ... 7:
                    self.txtVoice.text = self.txtVoice.text + "? I don't know how to do that yet."
                    NSLog("detect pattern to be implemented")
                    break
                    //self.detectPattern()
                default:
                    break
                }
            } else {
                // try again
                NSLog("try again")
                self.voice_timer?.invalidate()
                self.voice_timer = Timer.scheduledTimer(timeInterval: 2, target: self, selector: #selector(self.processVoice), userInfo: nil, repeats: false)
            }
        }
        
        NSLog("end processVoice")
    }
    
    private func detectObject() {
        btnRun.isEnabled = false
        btnRun.setTitle("Detecting...", for: .normal)
        self.txtVoice.text = ""

        image = self.imageView.image
        
        let resizedImage = image!.resized(to: CGSize(width: CGFloat(PrePostProcessor.inputWidth), height: CGFloat(PrePostProcessor.inputHeight)))
        
        let imgScaleX = Double(image!.size.width / CGFloat(PrePostProcessor.inputWidth));
        let imgScaleY = Double(image!.size.height / CGFloat(PrePostProcessor.inputHeight));
        
        let ivScaleX : Double = (image!.size.width > image!.size.height ? Double(imageView.frame.size.width / image!.size.width) : Double(imageView.frame.size.height / image!.size.height))
        let ivScaleY : Double = (image!.size.height > image!.size.width ? Double(imageView.frame.size.height / image!.size.height) : Double(imageView.frame.size.width / image!.size.width))

        let startX = Double((imageView.frame.size.width - CGFloat(ivScaleX) * image!.size.width)/2)
        let startY = Double((imageView.frame.size.height -  CGFloat(ivScaleY) * image!.size.height)/2)

        guard var pixelBuffer = resizedImage.normalized() else {
            return
        }
        
        DispatchQueue.global().async {
            guard let outputs = self.inferencer.module.detect(image: &pixelBuffer) else {
                return
            }
            
            let nmsPredictions = PrePostProcessor.outputsToNMSPredictions(outputs: outputs, imgScaleX: imgScaleX, imgScaleY: imgScaleY, ivScaleX: ivScaleX, ivScaleY: ivScaleY, startX: startX, startY: startY, patternType: false)
            
            DispatchQueue.main.async {
                PrePostProcessor.cleanDetection(imageView: self.imageView)
                let pred_text = PrePostProcessor.showDetection(imageView: &self.imageView, nmsPredictions: nmsPredictions, classes: self.inferencer.classes)
                if (self.txtVoice.text != "") {
                    self.txtVoice.text = self.txtVoice.text + "? "
                }
                self.txtVoice.text = self.txtVoice.text + pred_text
                self.btnRun.setTitle("Detect", for: .normal)
                self.btnRun.isEnabled = true
            }
        }
    }
    
    private func handlePermissionFailed() {
        // Present an alert asking the user to change their settings.
        let ac = UIAlertController(title: "This app must have access to speech recognition for this feature to work.", message: "Please update your settings.", preferredStyle: .alert)
        ac.addAction(UIAlertAction(title: "Open settings", style: .default) { _ in
            let url = URL(string: UIApplication.openSettingsURLString)!
            UIApplication.shared.open(url)
        })
        ac.addAction(UIAlertAction(title: "Close", style: .cancel))
        present(ac, animated: true)
        
        // Disable the record button.
        btnVoice.isEnabled = false
        btnVoice.setTitle("Enable Voice", for: .normal)
    }
    
    private func handleError(withMessage message: String) {
        // Present an alert.
        let ac = UIAlertController(title: "An error occured", message: message, preferredStyle: .alert)
        ac.addAction(UIAlertAction(title: "OK", style: .default))
        present(ac, animated: true)

        // Disable record button.
        //btnVoice.setTitle("Not available.", for: .normal)
        //btnVoice.isEnabled = false
    }
    // end BCJ
}

// AVCaptureVideoDataOutputSampleBufferDelegate protocol and related methods
extension ViewController:  AVCaptureVideoDataOutputSampleBufferDelegate{
     func setupAVCapture(){
        session.sessionPreset = AVCaptureSession.Preset.vga640x480
        guard let device = AVCaptureDevice
        .default(AVCaptureDevice.DeviceType.builtInWideAngleCamera,
                 for: .video,
                 position: AVCaptureDevice.Position.back) else {
                            return
        }
        captureDevice = device
        beginSession()
    }

    func beginSession(){
        var deviceInput: AVCaptureDeviceInput!

        do {
            deviceInput = try AVCaptureDeviceInput(device: captureDevice)
            guard deviceInput != nil else {
                print("error: cant get deviceInput")
                return
            }

            if self.session.canAddInput(deviceInput){
                self.session.addInput(deviceInput)
            }

            videoDataOutput = AVCaptureVideoDataOutput()
            videoDataOutput.alwaysDiscardsLateVideoFrames=true
            videoDataOutput.videoSettings = [String(kCVPixelBufferPixelFormatTypeKey): kCMPixelFormat_32BGRA]
            videoDataOutputQueue = DispatchQueue(label: "VideoDataOutputQueue")
            videoDataOutput.setSampleBufferDelegate(self, queue:self.videoDataOutputQueue)

            if session.canAddOutput(self.videoDataOutput){
                session.addOutput(self.videoDataOutput)
            }

            videoDataOutput.connection(with: .video)?.isEnabled = true
            videoDataOutput.connection(with: .video)?.videoOrientation = .portrait
            

            previewLayer = AVCaptureVideoPreviewLayer(session: self.session)
            previewLayer.videoGravity = AVLayerVideoGravity.resizeAspect

            let rootLayer :CALayer = self.camView.layer
            rootLayer.masksToBounds=true
            previewLayer.frame = rootLayer.bounds
            rootLayer.addSublayer(self.previewLayer)
            session.startRunning()
        } catch let error as NSError {
            deviceInput = nil
            print("error: \(error.localizedDescription)")
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        //NSLog("captured")
        connection.videoOrientation = .portrait
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        guard let normalizedBuffer = pixelBuffer.normalized(PrePostProcessor.inputWidth, PrePostProcessor.inputHeight) else {
            return
        }
        self.camFrame = normalizedBuffer

        if let cvImageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            let ciimage = CIImage(cvImageBuffer: cvImageBuffer)
            let context = CIContext()

            if let cgImage = context.createCGImage(ciimage, from: ciimage.extent) {
                let uiImage = UIImage(cgImage: cgImage)
                self.camImage = uiImage
            }
        }
    }

    // clean up AVCapture
    func stopCamera(){
        session.stopRunning()
    }

}
