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

public let audioSession = AVAudioSession.sharedInstance()
public let synthesizer = AVSpeechSynthesizer()
// end BCJ


class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate{
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var txtVoice: UITextView! // BCJ
    @IBOutlet weak var camView: UIView! // BCJ
    @IBOutlet weak var imgMic: UIImageView!
    @IBOutlet weak var btnRun: UIButton!
    @IBOutlet weak var btnVoice: UIButton!
    @IBOutlet weak var btnCamera: UIButton!
    @IBOutlet weak var btnNext: UIButton!
    
    private let testImages = ["test1.png", "test2.jpg", "test3.png"]
    private var imgIndex = 0

    private var inferencer = ObjectDetector()
    
    // start BCJ
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))!
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()

    private var videoDataOutput: AVCaptureVideoDataOutput!
    private var videoDataOutputQueue: DispatchQueue!
    private var previewLayer:AVCaptureVideoPreviewLayer!
    private var captureDevice : AVCaptureDevice!
    private let session = AVCaptureSession()
    
    private var camFrame: [Float32]?
    private var camImage: UIImage!
    
    private var voice_timer: Timer!
    private var continuous_timer: Timer!
    private var restart_timer: Timer!
    private var relaunch_timer: Timer!

    private var fullsTring: String!

    private var detectingState = false
    private var voiceMode = false
    
    private var long_synthesizer = AVSpeechSynthesizer()
    
    func toggleTorch(on: Bool) {
        guard let device = AVCaptureDevice.default(for: AVMediaType.video) else { return }
        guard device.hasTorch else { print("Torch isn't available"); return }

        do {
            try device.lockForConfiguration()
            device.torchMode = on ? .on : .off
            if on { try device.setTorchModeOn(level: AVCaptureDevice.maxAvailableTorchLevel) }
            device.unlockForConfiguration()
        } catch {
            print("Torch can't be used")
        }
    }
    
    @objc func activateTorch() {
        toggleTorch(on: true)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        if let iv = imageView {
            iv.image = UIImage(named: testImages[imgIndex])!
            btnRun.setTitle("Detect", for: .normal)
        }
        // start BCJ
        btnVoice.backgroundColor = UIColor.systemGreen
        
        try? audioSession.setCategory(AVAudioSession.Category.playAndRecord, mode: AVAudioSession.Mode.default, options: AVAudioSession.CategoryOptions.defaultToSpeaker)
        try? audioSession.setMode(AVAudioSession.Mode.spokenAudio)
        try? audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        synthesizer.speak(AVSpeechUtterance(string: "Descriptive World"))
        long_synthesizer.delegate = self

        NotificationCenter.default.addObserver(self,
                                               selector: #selector(activateTorch),
                                               name: UIApplication.didBecomeActiveNotification,
                                               object: nil)
        startVoice(self)
        // end BCJ
    }
        
    @IBAction func runTapped(_ sender: Any) {
        detectObject()
    }

    @IBAction func nextTapped(_ sender: Any) {
        print("nextTapped")
        PrePostProcessor.cleanDetection(imageView: imageView)
        self.txtVoice.text = ""
        imgIndex = (imgIndex + 1) % testImages.count
        btnNext.setTitle(String(format: "Test Image %d/%d", imgIndex + 1, testImages.count), for:.normal)
        imageView.image = UIImage(named: testImages[imgIndex])!
    }

    @IBAction func photosTapped(_ sender: Any) {
        PrePostProcessor.cleanDetection(imageView: imageView)
        self.txtVoice.text = ""
        let imagePickerController = UIImagePickerController()
        imagePickerController.delegate = self;
        imagePickerController.sourceType = .photoLibrary
        self.present(imagePickerController, animated: true, completion: nil)
    }
    
    @IBAction func cameraTapped(_ sender: Any) {
        PrePostProcessor.cleanDetection(imageView: imageView)
        self.txtVoice.text = ""
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            let imagePickerController = UIImagePickerController()
            imagePickerController.delegate = self;
            imagePickerController.sourceType = .camera
            self.present(imagePickerController, animated: true, completion: nil)
        }
    }
    
    private func voiceTapped() {
        PrePostProcessor.cleanDetection(imageView: self.imageView)
        self.txtVoice.text = ""
        if (!self.voiceMode) { // start voice processing
            self.voiceMode = true
            //self.btnVoice.setImage(UIImage(systemName: "mic.circle"), for: .normal)
            self.btnVoice.setTitle("Stop", for: .normal)
            self.btnVoice.backgroundColor = UIColor.systemPurple
            sayReady()
            self.camView.isHidden = false
            self.imageView.isHidden = true
            self.setupAVCapture()
        } else { // shutdown voice processing
            self.voiceMode = false
            self.restart_timer?.invalidate()
            self.recognitionTask?.cancel()
            self.recognitionRequest = nil
            self.voice_timer?.invalidate()
            self.audioEngine.stop()
            //self.btnVoice.setImage(UIImage(systemName: "mic.circle.fill"), for: .normal)
            self.btnVoice.setTitle("Start Mic", for: .normal)
            self.btnVoice.backgroundColor = UIColor.systemGreen
            sayReady(text: "finished")
            self.stopCamera()
            self.camView.isHidden = true
            self.imageView.isHidden = false
        }
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        print("imagePickerController")
        self.txtVoice.text = ""
        var image = info[UIImagePickerController.InfoKey.originalImage] as? UIImage
        image = image!.resized(to: CGSize(width: CGFloat(PrePostProcessor.inputWidth), height: CGFloat(PrePostProcessor.inputHeight)*image!.size.height/image!.size.width))
        imageView.image = image
        self.dismiss(animated: true, completion: nil)
        self.detectObject()
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
    
    @objc func sayReady(text: String = "ready") {
        try? audioSession.setMode(AVAudioSession.Mode.spokenAudio)
        synthesizer.speak(AVSpeechUtterance(string: text))
    }
    
    @IBAction func startVoice(_ sender: Any) {
        NSLog("begin startVoice")
        voiceTapped()
        _ = Timer.scheduledTimer(withTimeInterval: 2, repeats: false) { timer in
            try? audioSession.setMode(.measurement)
            self.startListen(self)
        }
        NSLog("end startVoice")
    }
    
    @IBAction func startListen(_ sender: Any) {
        NSLog("begin startListen")
        if (self.btnVoice.title(for: .normal)=="Voice") {
            return
        }
        self.txtVoice.text = ""
        
        audioEngine.reset()
        if recognitionTask != nil {
                recognitionTask?.cancel()
                recognitionTask = nil
        }

        // Create and configure the speech recognition request.
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { fatalError("Unable to create a SFSpeechAudioBufferRecognitionRequest object") }
        recognitionRequest.shouldReportPartialResults = true

        // Configure the microphone input.
        let inputNode = audioEngine.inputNode

        // Setup a recognition task for the speech recognition session.
        // Keep a reference to the task so that it can be canceled.
        self.continuous_timer = Timer.scheduledTimer(timeInterval: 10, target: self, selector: #selector(self.restartListen), userInfo: nil, repeats: false)
        
        self.recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { result, error in
            var isFinal = false
            //NSLog("recognizing...")
            if result != nil {
                print("result != nil")
                self.restart_timer?.invalidate()
                self.restart_timer = Timer.scheduledTimer(timeInterval: 0.5, target: self, selector: #selector(self.didFinishTalk), userInfo: nil, repeats: false)
                
                let bestString = result?.bestTranscription.formattedString
                self.fullsTring = bestString!
                self.txtVoice.text = bestString
                isFinal = result!.isFinal
                
            }
            if isFinal {
                print("final")
                self.audioEngine.stop()
                inputNode.removeTap(onBus: 0)
                self.recognitionRequest = nil
                self.recognitionTask = nil
                isFinal = false
            }
            if error != nil{
                print("error != nil")
                URLCache.shared.removeAllCachedResponses()
                guard let task = self.recognitionTask else {
                    return
                }
                task.cancel()
                task.finish()
            }
        }
        audioEngine.reset()
        inputNode.removeTap(onBus: 0)
        
        //let recordingFormat = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { (buffer, when) in
            self.recognitionRequest?.append(buffer)
        }
        
        audioEngine.prepare()
        
        do {
            try audioEngine.start()
        } catch {
            print("audioEngine couldn't start because of an error.")
        }
        self.imgMic.image = UIImage(systemName: "mic.fill")

        NSLog("end startListen")
    }
    
    @objc func restartListen(_ forceReset: Bool=false){
        print("restartListen")
        //print(self.txtVoice.text)
        if (forceReset ||
            self.txtVoice.text.contains("restart") ||
            self.txtVoice.text.contains("reset") ||
            self.txtVoice.text.contains("clear")) {
            
            self.continuous_timer?.invalidate()
            self.restart_timer?.invalidate()
            if ((self.audioEngine.isRunning)){
                self.audioEngine.stop()
                self.recognitionRequest?.endAudio()
                self.recognitionTask?.finish()
            }
            if self.voiceMode {
                self.camView.isHidden = false
                self.imageView.isHidden = true
                self.relaunch_timer = Timer.scheduledTimer(timeInterval: 0.2, target: self, selector: #selector(startListen), userInfo: nil, repeats: false)
            }
        }
    }
    
    @objc func didFinishTalk(){
        print("didFinishTalk")
        if processVoice() {
            self.continuous_timer?.invalidate()
            self.restart_timer?.invalidate()
            self.relaunch_timer?.invalidate()
            
            if ((self.audioEngine.isRunning)){
                self.imgMic.image = UIImage(systemName: "mic.slash")
                self.audioEngine.stop()
                guard let task = self.recognitionTask else {
                    return
                }
                task.cancel()
                task.finish()
            }
        }
    }

    @IBAction func endVoice(_ sender: Any) {
    }
    
    @objc private func processVoice() -> Bool {
        NSLog("begin processVoice")
        if (detectingState) {
            return true
        }
        if (self.txtVoice.text.contains("restart") ||
            self.txtVoice.text.contains("reset") ||
            self.txtVoice.text.contains("clear")) {
            restartListen()
        }
        var recognizedCommand = false
        if let sentenceEmbedding = NLEmbedding.sentenceEmbedding(for: .english) {
            let sentence = self.txtVoice.text.lowercased()+""
            var commandScores = [Double]()
            let knownCommands = ["what is this",
                                 "tell me what this is",
                                 "color",
                                 "what color is this",
                                 "tell me which color this is",
                                 "pattern",
                                 "what pattern is this",
                                 "tell me which pattern this is",
                                 "fabric",
                                 "what fabric is this",
                                 "tell me which fabric this is",
                                 "garment",
                                 "what garment is this",
                                 "tell me which garment this is",
                                 "kind of garment",
                                 "clothing",
                                 "what clothing is this",
                                 "tell me which clothing this is",
                                 "kind of clothing",
                                 "item",
                                 "what item is this",
                                 "tell me which item this is",
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
                print("set image after NLP")
                self.imageView.image = self.camImage
                self.camView.isHidden = true
                self.imageView.isHidden = false
                let bestCommand = commandScores.firstIndex(of: bestScore!)
                switch bestCommand! {
                    case 0 ... 1:
                        self.detectObject(detect_type: "all")
                    case 2 ... 4:
                        self.detectObject(detect_type: "color")
                    case 5 ... 10:
                        self.detectObject(detect_type: "pattern")
                    case 11 ... 21:
                        self.detectObject(detect_type: "garment")
                    default:
                        break
                }
                recognizedCommand = true
            }
        }
        NSLog("end processVoice")
        return recognizedCommand
    }

    private func detectObject(detect_type: String = "all") {
        if (detectingState) {
            return
        }
        detectingState = true
        print("begin detectObject:  detecting "+detect_type)
        PrePostProcessor.cleanDetection(imageView: self.imageView)
        
        try? audioSession.setMode(AVAudioSession.Mode.spokenAudio)
        synthesizer.speak(AVSpeechUtterance(string: "detecting"))
        
        btnRun.isEnabled = false
        btnRun.setTitle("Detecting...", for: .normal)
        self.txtVoice.text = ""

        let image = self.imageView.image

        let resizedImage = image!.resized(to: CGSize(width: CGFloat(PrePostProcessor.inputWidth), height: CGFloat(PrePostProcessor.inputHeight)))
        //print(resizedImage.hasAlpha)
        //self.imageView.image = resizedImage // uncomment to see what the algo sees
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
                let pred_text = PrePostProcessor.showDetection(imageView: &self.imageView, nmsPredictions: nmsPredictions, classes: self.inferencer.classes, detectType: detect_type, synth: self.long_synthesizer)
                if (self.txtVoice.text != "") {
                    self.txtVoice.text = self.txtVoice.text + "? "
                }
                self.txtVoice.text = self.txtVoice.text + pred_text
                self.btnRun.setTitle("Detect", for: .normal)
                self.btnRun.isEnabled = true
                
                self.detectingState = false
                
                /*if self.voiceMode {  // this became annoying after each detection
                    try? audioSession.setMode(AVAudioSession.Mode.spokenAudio)
                    self.long_synthesizer.speak(AVSpeechUtterance(string: "ready"))
                }*/
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
         toggleTorch(on: true)
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
extension ViewController: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        self.restartListen(true)
/*        if (utterance.speechString=="ready") {
            print("finished reading out detections")
            self.restartListen(true)
        }*/
    }
}
