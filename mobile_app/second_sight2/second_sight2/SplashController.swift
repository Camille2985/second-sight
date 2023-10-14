//
//  SplashController.swift
//  second_sight2
//
//  Created by Camille Church on 10/9/23.
//

import UIKit
import AVFoundation

class SplashController: UIViewController {

    let synthesizer = AVSpeechSynthesizer()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        let utterance = AVSpeechUtterance(string: "Second Sight is Loading")
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        self.synthesizer.speak(utterance)
        
        // Wait 3 Seconds and go to next screen
        DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + 3) {
            self.performSegue(withIdentifier: "OpenApp", sender: nil)
            
        }
    }
    
}
