//
//  ViewController.swift
//  second_sight2
//
//  Created by Camille Church on 9/18/23.
//
import Vision
import UIKit
import AVFoundation

let synthesizer = AVSpeechSynthesizer()
private let label: UILabel = {
    let label = UILabel()
    label.numberOfLines = 0
    label.backgroundColor = UIColor.darkGray
    label.textAlignment = .center
    label.text = "Starting..."
    return label
}()

private let imageView: UIImageView = {
    let imageView = UIImageView()
    imageView.image = UIImage(named: "example1")
    imageView.contentMode = .scaleAspectFit
    return imageView
}()

private let button: UIButton = {
    let button = UIButton()
    button.setTitle("Take Picture", for: .normal)
    button.backgroundColor = UIColor.blue
    return button
    
}()

class ViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let utterance = AVSpeechUtterance(string: "Application is ready for use")
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        synthesizer.speak(utterance)
            
        view.addSubview(label)
        view.addSubview(imageView)
        view.addSubview(button)
        button.addTarget(self, action: #selector(pressed), for: .touchUpInside)
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        imageView.frame = CGRect(x:20, y: view.safeAreaInsets.top,
                                 width: view.frame.size.width - 40,
                                 height: view.frame.size.width - 40
        )
        label.frame = CGRect(x: 20,
                             y: view.frame.size.width + view.safeAreaInsets.top,
                             width: view.frame.size.width - 40,
                             height: 200)
        button.frame = CGRect(x: 20,
                              y: view.frame.size.width + view.safeAreaInsets.top + 220,
                              width: view.frame.size.width - 40,
                              height: 80)
    }
    
    @objc func pressed(image: UIImage) {
        print("button clicked")
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = self
        present(picker, animated: true)
        recognizeText(image: imageView.image)
    }


}

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        picker.dismiss(animated: true, completion: nil)
        
        guard let image = info[UIImagePickerController.InfoKey.originalImage] as? UIImage else {
            return
        }
        imageView.image = image
        recognizeText(image: imageView.image)
    }
    
    
}

func recognizeText(image: UIImage?){
    guard let cgImage = image?.cgImage else {
        fatalError("Could not get image")
    }
    // Handler
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    
    // Request
    let request = VNRecognizeTextRequest {request, error in
        guard let observations = request.results as? [VNRecognizedTextObservation],
              error == nil else {
            return
        }
        let text = observations.compactMap({
            $0.topCandidates(1).first?.string
        }).joined(separator: ", ")
        
        DispatchQueue.main.async {
            label.text = text
            let utterance = AVSpeechUtterance(string: text)
            utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
            utterance.rate = 0.5
            synthesizer.speak(utterance)
        }
    }

    // Process Request
    do {
        try handler.perform([request])
        
    }
    catch{
        print(error)
    }
}
