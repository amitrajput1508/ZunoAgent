import webview
import json
from backend import handle_intent
import speech_recognition as sr
import threading
import time
import sys

from gtts import gTTS
import tempfile
import os
from datetime import datetime
from pathlib import Path
import base64
import requests
from PIL import Image, ImageFilter
import mss
import subprocess
import pvporcupine
import pyaudio
import struct

# Vision configuration
VISION_INTERVAL = 15  # seconds between screenshots
VISION_MODEL = "meta-llama/llama-3.2-11b-vision-instruct"
VISION_API_URL = "https://openrouter.ai/api/v1/chat/completions"
VISION_API_KEY = ""
VISION_OUTPUT_DIR = Path("screen_logs")
VISION_SCREENSHOT_DIR = VISION_OUTPUT_DIR / "screenshots"
VISION_LOG_FILE = VISION_OUTPUT_DIR / "descriptions.log"
DELETE_SCREENSHOT_AFTER_SEND = True
BLUR_SENSITIVE_REGION = False
SENSITIVE_BOX = None

# Create directories if they don't exist
VISION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISION_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

class WakeWordDetector:
    def __init__(self, access_key=None):
        self.access_key = access_key
        self.is_listening = False
        self.callback = None
        self.porcupine = None
        self.audio_stream = None
        self.pa = None
        
    def initialize(self):
        """Initialize the wake word detector"""
        try:
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=["picovoice", "hey juno"]
            )
            
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            return True
        except Exception as e:
            print(f"Wake word initialization failed: {e}")
            return False
    
    def start_listening(self, callback):
        if not self.initialize():
            return False
            
        self.callback = callback
        self.is_listening = True
        
        def listen_loop():
            print("🔊 Wake word detector started...")
            while self.is_listening:
                try:
                    pcm = self.audio_stream.read(self.porcupine.frame_length)
                    pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                    
                    result = self.porcupine.process(pcm)
                    if result >= 0:
                        print(f"🎯 Wake word detected at {datetime.now()}")
                        if self.callback:
                            self.callback()
                        time.sleep(2)
                        
                except Exception as e:
                    print(f"Wake word error: {e}")
                    time.sleep(0.1)
            
            self.cleanup()
        
        self.listener_thread = threading.Thread(target=listen_loop, daemon=True)
        self.listener_thread.start()
        return True
    
    def stop_listening(self):
        self.is_listening = False
        self.cleanup()
    
    def cleanup(self):
        try:
            if self.audio_stream:
                self.audio_stream.close()
            if self.pa:
                self.pa.terminate()
            if self.porcupine:
                self.porcupine.delete()
        except:
            pass

class SimpleVAD:
    def __init__(self):
        self.is_listening = False
        self.callback = None
        
    def start_listening(self, callback):
        self.callback = callback
        self.is_listening = True
        
        def vad_loop():
            import speech_recognition as sr
            r = sr.Recognizer()
            
            print("🔊 Simple VAD started - Say 'Zuno' to activate")
            
            while self.is_listening:
                try:
                    with sr.Microphone() as source:
                        r.adjust_for_ambient_noise(source, duration=0.5)
                        audio = r.listen(source, timeout=1, phrase_time_limit=3)
                    
                    text = r.recognize_google(audio).lower()
                    if 'zuno' in text or 'hello' in text or 'hey' in text:
                        print(f"🎯 Activation word detected: {text}")
                        if self.callback:
                            self.callback()
                        time.sleep(2)
                        
                except sr.UnknownValueError:
                    pass
                except sr.WaitTimeoutError:
                    pass
                except Exception as e:
                    print(f"VAD error: {e}")
                    time.sleep(0.1)
        
        self.vad_thread = threading.Thread(target=vad_loop, daemon=True)
        self.vad_thread.start()
    
    def stop_listening(self):
        self.is_listening = False

# UNIVERSE LANDING PAGE HTML with planet images
UNIVERSE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Zuno AI — Cosmic Universe</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      background: #000;
      font-family: "Poppins", sans-serif;
      height: 100vh;
      overflow: hidden;
      color: white;
      position: relative;
    }

    /* Window Controls */
    .window-controls {
        position: fixed;
        top: 0;
        right: 0;
        display: flex;
        z-index: 9999;
        -webkit-app-region: no-drag;
    }
    
    .window-btn {
        width: 46px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #97a7ca;
        cursor: pointer;
        transition: all 0.2s ease;
        background: transparent;
        border: none;
        font-size: 12px;
    }

    .window-btn:hover {
        background: rgba(255,255,255,0.1);
        color: white;
    }
    
    .window-btn.close:hover {
        background: #e81123;
        color: white;
    }

    #universe {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      z-index: -1;
      background: radial-gradient(ellipse at center, #050520 0%, #000000 100%);
    }

    .star {
      position: absolute;
      border-radius: 50%;
      background: white;
      opacity: 0.8;
      animation: twinkle 6s infinite ease-in-out;
    }

    .star.small {
      width: 1px;
      height: 1px;
    }
    .star.medium {
      width: 2px;
      height: 2px;
    }
    .star.large {
      width: 3px;
      height: 3px;
      box-shadow: 0 0 8px white;
    }

    @keyframes twinkle {
      0%, 100% { opacity: 0.2; }
      50% { opacity: 1; }
    }

    .shooting-star {
      position: absolute;
      width: 2px;
      height: 2px;
      background: white;
      box-shadow: 0 0 10px 3px white;
      border-radius: 50%;
      opacity: 0;
    }

    @keyframes shoot {
      0% {
        opacity: 1;
        transform: translateX(0) translateY(0);
      }
      100% {
        opacity: 0;
        transform: translateX(-500px) translateY(250px);
      }
    }

    .nebula {
      position: absolute;
      border-radius: 50%;
      filter: blur(90px);
      opacity: 0.25;
      mix-blend-mode: screen;
      animation: drift 60s infinite linear;
    }

    .nebula.purple {
      background: radial-gradient(circle, #8a2be2, transparent 70%);
    }
    .nebula.blue {
      background: radial-gradient(circle, #00bfff, transparent 70%);
    }
    .nebula.pink {
      background: radial-gradient(circle, #ff1493, transparent 70%);
    }

    @keyframes drift {
      0% { transform: translate(0, 0); }
      50% { transform: translate(100px, -50px); }
      100% { transform: translate(0, 0); }
    }

    .zuno-text {
      position: absolute;
      top: 25%;
      left: 50%;
      transform: translate(-50%, -50%);
      text-align: center;
      z-index: 100;
    }

    .zuno-main {
      font-size: 5rem;
      font-weight: 800;
      background: linear-gradient(45deg, #8a2be2, #00bfff, #ff1493, #7fff00);
      -webkit-background-clip: text;
      color: transparent;
      animation: textGlow 3s ease-in-out infinite alternate;
      letter-spacing: 4px;
      margin-bottom: 10px;
      position: relative;
      text-transform: uppercase;
    }

    .zuno-main::after {
      content: "";
      position: absolute;
      bottom: -10px;
      left: 50%;
      transform: translateX(-50%);
      width: 200px;
      height: 3px;
      background: linear-gradient(90deg, transparent, #00bfff, #ff1493, transparent);
      border-radius: 10px;
    }

    @keyframes textGlow {
      0% { text-shadow: 0 0 20px #8a2be2, 0 0 40px #00bfff; }
      100% { text-shadow: 0 0 40px #ff1493, 0 0 80px #7fff00; }
    }

    .zuno-sub {
      font-size: 1.4rem;
      color: #b5dfff;
      text-shadow: 0 0 10px #00bfff;
      animation: subtitleFloat 4s infinite ease-in-out;
      font-weight: 300;
      letter-spacing: 2px;
      margin-top: 20px;
    }

    @keyframes subtitleFloat {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-10px); }
    }

    .center-container {
      position: absolute;
      top: 55%;
      left: 50%;
      transform: translate(-50%, -50%);
      z-index: 200;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .start-button {
      width: 160px;
      height: 160px;
      border-radius: 50%;
      border: none;
      background: radial-gradient(circle at center, #8a2be2 0%, #4b0082 100%);
      box-shadow: 0 0 100px #8a2be2, inset 0 0 20px rgba(255, 255, 255, 0.3);
      cursor: pointer;
      position: relative;
      transition: all 0.4s ease;
      display: flex;
      justify-content: center;
      align-items: center;
    }

    .start-button::before {
      content: "";
      position: absolute;
      top: -4px;
      left: -4px;
      right: -4px;
      bottom: -4px;
      background: conic-gradient(from 0deg, #8a2be2, #00bfff, #ff1493, #7fff00, #8a2be2);
      border-radius: 50%;
      filter: blur(10px);
      animation: rotate 4s linear infinite;
      z-index: -1;
    }

    @keyframes rotate {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    .start-label {
      font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
      font-size: 20px;
      color: white;
      font-weight: 600;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      padding: 12px 24px;
      background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
      border: 1px solid rgba(255,255,255,0.2);
      border-radius: 12px;
      backdrop-filter: blur(10px);
      display: inline-block;
      text-shadow: 0 2px 4px rgba(0,0,0,0.3);
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .start-label:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(0,0,0,0.3);
      background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.08) 100%);
    }

    .start-text {
      margin-top: 20px;
      color: #b5dfff;
      font-size: 1rem;
      text-shadow: 0 0 10px #00bfff;
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 0.7; }
      50% { opacity: 1; }
    }

    .solar-system {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 1000px;
      height: 1000px;
      z-index: 10;
    }

    .orbit {
      position: absolute;
      top: 50%;
      left: 50%;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 50%;
      transform: translate(-50%, -50%);
    }

    .planet {
      position: absolute;
      border-radius: 50%;
      top: 50%;
      left: 50%;
      transform-style: preserve-3d;
      cursor: pointer;
      transition: transform 0.3s ease;
      background-size: cover;
      background-position: center;
      box-shadow: 0 0 20px var(--planet-glow);
    }

    .planet:hover {
      transform: scale(1.2);
      z-index: 20;
    }

    .planet-info {
      position: absolute;
      top: -40px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(0, 0, 0, 0.7);
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 0.8rem;
      opacity: 0;
      transition: all 0.3s ease;
      white-space: nowrap;
      border: 1px solid rgba(255, 255, 255, 0.2);
      box-shadow: 0 0 10px rgba(0, 191, 255, 0.3);
    }

    .planet:hover .planet-info {
      opacity: 1;
      transform: translateX(-50%) translateY(-10px);
    }

    @keyframes orbit {
      0% { transform: rotate(0deg) translateX(var(--distance)) rotate(0deg); }
      100% { transform: rotate(360deg) translateX(var(--distance)) rotate(-360deg); }
    }

    .particle {
      position: absolute;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.6);
      pointer-events: none;
      z-index: 5;
    }

    .transition-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: black;
      opacity: 0;
      pointer-events: none;
      z-index: 1000;
      transition: opacity 0.6s ease-in-out;
    }

    .footer {
      position: absolute;
      bottom: 20px;
      left: 0;
      width: 100%;
      text-align: center;
      color: rgba(255, 255, 255, 0.5);
      font-size: 0.8rem;
      z-index: 100;
    }

    @media (max-width: 1024px) {
      .solar-system { width: 800px; height: 800px; }
    }

    @media (max-width: 768px) {
      .zuno-main { font-size: 3rem; }
      .zuno-sub { font-size: 1.1rem; }
      .start-button { width: 120px; height: 120px; }
      .solar-system { width: 600px; height: 600px; }
    }

    @media (max-width: 480px) {
      .zuno-main { font-size: 2.2rem; }
      .zuno-sub { font-size: 0.9rem; }
      .start-button { width: 100px; height: 100px; }
      .solar-system { width: 400px; height: 400px; }
    }
  </style>
</head>

<body>
  <!-- Window Controls -->
  <div class="window-controls">
      <div class="window-btn" id="minimize-btn">
          <i class="fas fa-minus"></i>
      </div>
      <div class="window-btn" id="maximize-btn">
          <i class="far fa-square"></i>
      </div>
      <div class="window-btn close" id="close-btn">
          <i class="fas fa-times"></i>
      </div>
  </div>

  <div id="universe"></div>
  
  <div class="solar-system" id="solarSystem"></div>

  <div class="zuno-text">
    <div class="zuno-main">ZUNO AI</div>
    <div class="zuno-sub">Your Cosmic Assistant</div>
  </div>

  <div class="center-container">
    <button class="start-button" id="startButton">
      <div class="start-label">Start →</div>
    </button>
    <div class="start-text">Click to begin your journey</div>
  </div>

  <div class="footer">
    Powered by Zuno AI • Explore the Universe of Possibilities
  </div>

  <div class="transition-overlay" id="transitionOverlay"></div>

  <script>
    // ========== STAR CREATION ==========
    function createStars() {
      const universe = document.getElementById("universe");
      // Clear any existing stars
      universe.innerHTML = '';
      
      for (let i = 0; i < 800; i++) {
        const star = document.createElement("div");
        const sizes = ["small", "medium", "large"];
        const sizeClass = sizes[Math.floor(Math.random() * sizes.length)];
        star.classList.add("star", sizeClass);
        
        star.style.left = Math.random() * 100 + "%";
        star.style.top = Math.random() * 100 + "%";
        star.style.animationDelay = Math.random() * 6 + "s";
        star.style.animationDuration = (3 + Math.random() * 4) + "s";
        
        universe.appendChild(star);
      }
    }

    // ========== NEBULA CREATION ==========
    function createNebulas() {
      const universe = document.getElementById("universe");
      const colors = ["purple", "blue", "pink"];
      for (let i = 0; i < 8; i++) {
        const nebula = document.createElement("div");
        nebula.classList.add("nebula", colors[Math.floor(Math.random() * colors.length)]);
        nebula.style.width = 300 + Math.random() * 400 + "px";
        nebula.style.height = 300 + Math.random() * 400 + "px";
        nebula.style.top = Math.random() * 100 + "%";
        nebula.style.left = Math.random() * 100 + "%";
        nebula.style.animationDuration = (40 + Math.random() * 40) + "s";
        universe.appendChild(nebula);
      }
    }

    // ========== PLANETS ==========
    const planets = [
      { name: "Mercury", size: 25, distance: "250px", color: "#8C7853", glow: "#8C7853", image: "images/planets/mercury.png" },
      { name: "Venus", size: 35, distance: "320px", color: "#E39E54", glow: "#E39E54", image: "images/planets/venus.png" },
      { name: "Earth", size: 40, distance: "400px", color: "#6B93D6", glow: "#6B93D6", image: "images/planets/earth.png" },
      { name: "Mars", size: 30, distance: "460px", color: "#C1440E", glow: "#C1440E", image: "images/planets/mars.png" },
      { name: "Jupiter", size: 70, distance: "520px", color: "#C19A6B", glow: "#C19A6B", image: "images/planets/jupiter.png" },
      { name: "Saturn", size: 65, distance: "600px", color: "#E4CD9E", glow: "#E4CD9E", image: "images/planets/saturn.png" },
      { name: "Uranus", size: 50, distance: "680px", color: "#D1E7E7", glow: "#D1E7E7", image: "images/planets/uranus.png" },
      { name: "Neptune", size: 50, distance: "760px", color: "#5B5DDF", glow: "#5B5DDF", image: "images/planets/neptune.png" },
    ];

    function createPlanets() {
      const solarSystem = document.getElementById("solarSystem");
      
      // Create orbits
      planets.forEach((p, index) => {
        const orbit = document.createElement("div");
        orbit.classList.add("orbit");
        orbit.style.width = `calc(${p.distance} * 2)`;
        orbit.style.height = `calc(${p.distance} * 2)`;
        solarSystem.appendChild(orbit);
      });
      
      // Create planets
      planets.forEach((p, index) => {
        const planet = document.createElement("div");
        planet.classList.add("planet");
        planet.style.width = p.size + "px";
        planet.style.height = p.size + "px";
        planet.style.setProperty("--distance", p.distance);
        planet.style.setProperty("--planet-glow", p.glow);
        planet.style.animation = `orbit ${30 + index * 10}s linear infinite`;
        planet.style.animationDelay = `${index * 2}s`;
        
        // Set planet image as background
        planet.style.backgroundImage = `url('${p.image}')`;

        const info = document.createElement("div");
        info.classList.add("planet-info");
        info.textContent = p.name;
        info.style.color = p.color;
        planet.appendChild(info);

        solarSystem.appendChild(planet);
      });
    }

    // ========== SHOOTING STARS ==========
    function createShootingStar() {
      const universe = document.getElementById("universe");
      const s = document.createElement("div");
      s.classList.add("shooting-star");
      s.style.left = 80 + Math.random() * 20 + "%";
      s.style.top = Math.random() * 30 + "%";
      s.style.animation = `shoot ${1 + Math.random() * 2}s ease-out forwards`;
      universe.appendChild(s);
      setTimeout(() => s.remove(), 3000);
    }

    // ========== FLOATING PARTICLES ==========
    function createParticles() {
      const universe = document.getElementById("universe");
      for (let i = 0; i < 50; i++) {
        const particle = document.createElement("div");
        particle.classList.add("particle");
        const size = Math.random() * 3 + 1;
        particle.style.width = size + "px";
        particle.style.height = size + "px";
        particle.style.left = Math.random() * 100 + "%";
        particle.style.top = Math.random() * 100 + "%";
        particle.style.animation = `float ${15 + Math.random() * 20}s linear infinite`;
        particle.style.animationDelay = Math.random() * 10 + "s";
        universe.appendChild(particle);
      }
    }

    // ========== TRANSITION ==========
    function transitionToChatbot() {
      const overlay = document.getElementById("transitionOverlay");
      overlay.style.opacity = "1";
      setTimeout(() => {
        window.pywebview.api.load_chatbot();
      }, 600);
    }

    // ========== INIT ==========
    document.addEventListener("DOMContentLoaded", () => {
      createStars();
      createNebulas();
      createPlanets();
      createParticles();

      // Create shooting stars periodically
      setInterval(() => {
        if (Math.random() > 0.7) createShootingStar();
      }, 2000);

      document.getElementById("startButton").addEventListener("click", () => {
        transitionToChatbot();
      });

      // Window controls
      document.getElementById("minimize-btn").addEventListener("click", () => {
        window.pywebview.api.minimize_window();
      });
      
      document.getElementById("maximize-btn").addEventListener("click", () => {
        window.pywebview.api.maximize_window();
      });
      
      document.getElementById("close-btn").addEventListener("click", () => {
        window.pywebview.api.close_window();
      });

      // Add keyboard support
      document.addEventListener("keydown", (e) => {
        if (e.code === "Enter" || e.code === "Space") {
          transitionToChatbot();
        }
      });
    });
  </script>
</body>
</html>
"""

# CHATBOT HTML with Labs feature
CHATBOT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Zuno AI</title>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                       "Roboto", "Helvetica Neue", Arial, sans-serif;
        }

        :root {
            --text-color: #edf3ff;
            --subheading-color: #97a7ca;
            --placeholder-color: #c3cdde;
            --primary-color: #101623;
            --secondary-color: #283045;
            --secondary-hover-color: #333e58;
            --scrollbar-color: #626a7f;
            --accent-blue: #1d7efd;
            --accent-purple: #8f6fff;
            --gradient: linear-gradient(to right, #1d7efd, #8f6fff);
        }

        body {
            color: var(--text-color);
            background: var(--primary-color);
            overflow: hidden;
            height: 100vh;
            -webkit-app-region: drag;
        }

        /* Window Controls */
        .window-controls {
            position: fixed;
            top: 0;
            right: 0;
            display: flex;
            z-index: 9999;
            -webkit-app-region: no-drag;
        }
        
        .window-btn {
            width: 46px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--subheading-color);
            cursor: pointer;
            transition: all 0.2s ease;
            background: transparent;
            border: none;
            font-size: 12px;
        }

        .window-btn:hover {
            background: rgba(255,255,255,0.1);
            color: var(--text-color);
        }
        
        .window-btn.close:hover {
            background: #e81123;
            color: white;
        }

        .container {
            padding: 32px 0 120px;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
        }

        .container :where(.app-header, .suggestions, .message, .prompt-wrapper, .dislaimer-text) {
            margin: 0 auto;
            width: 100%;
            padding: 0 20px;
            max-width: 980px;
        }

        .container .app-header {
            margin-top: 20vh;
            margin-left: 600px;
            width: calc(100% - 370px);
            transition: all 0.5s ease;
        }

        .app-header .heading {
            font-size: 3rem;
            width: fit-content;
            background: var(--gradient);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 600;
        }

        .app-header .sub-heading {
            font-size: 2.6rem;
            margin-top: -5px;
            color: var(--subheading-color);
            font-weight: 500;
        }

        .container .suggestions {
            display: flex;
            gap: 15px;
            margin-top: 9.5vh;
            list-style: none;
            scrollbar-width: none;
            margin-left: 350px;
            width: calc(100% - 370px);
            padding-right: 20px;
            transition: all 0.5s ease;
            margin-left: 600px;
        }

        .suggestions .suggestions-item {
            width: 240px;
            padding: 18px;
            flex-shrink: 0;
            display: flex;
            cursor: pointer;
            flex-direction: column;
            align-items: flex-end;
            justify-content: space-between;
            border-radius: 12px;
            background: var(--secondary-color);
            transition: 0.3s ease;
            border: 1px solid rgba(255,255,255,0.05);
        }

        .suggestions .suggestions-item:hover {
            background: var(--secondary-hover-color);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }

        .suggestions .suggestions-item .text {
            font-size: 1.1rem;
            line-height: 1.4;
            color: var(--text-color);
        }

        .suggestions .suggestions-item span {
            height: 45px;
            width: 45px;
            margin-top: 35px;
            display: flex;
            align-self: flex-end;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            color: var(--accent-blue);
            background: var(--primary-color);
            transition: 0.3s ease;
        }

        .suggestions .suggestions-item:hover span {
            transform: scale(1.1);
        }

        .suggestions .suggestions-item:nth-child(2) span {
            color: #28a745;
        }

        .suggestions .suggestions-item:nth-child(3) span {
            color: #ffc107;
        }

        .suggestions .suggestions-item:nth-child(4) span {
            color: #6f42c1;
        }
        
        .chats-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
            padding: 20px 0;
            margin-bottom: 10px;
            max-height: calc(100vh - 300px);
            margin-left: 350px;
            padding-right: 20px;
            width: calc(100% - 370px);
            position: relative;
            z-index: 1;
        }

        .chats-container::-webkit-scrollbar {
            display: none;
        }
        .chats-container {
            -ms-overflow-style: none;
            scrollbar-width: none;
        }

        .chats-container .message {
            display: flex;
            gap: 11px;
            align-items: flex-start;
            width: 100%;
            max-width: 100%;
        }

        .chats-container .bot-message .avatar {
            height: 50px;
            width: 50px;
            flex-shrink: 0;
            padding: 6px;
            align-self: flex-start;
            background: var(--secondary-color);
            border-radius: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-left: -1px;
            margin-top: -6px;
        }

        .chats-container .bot-message.loading .avatar {
            animation: rotate 4s linear infinite;
        }

        @keyframes rotate {
            100% { transform: rotate(360deg); }
        }

        .chats-container .message .message-text {
            word-wrap: break-word;
            white-space: pre-line;
            line-height: 1.4;
            margin-right: 350px;
        }

        .chats-container .bot-message {
            margin: 10px 0;
            margin-left: 280px;
            max-width: 80%;           
            padding: 10px 15px;
            word-wrap: break-word;
            white-space: normal;
        }

        .chats-container .user-message {
            flex-direction: column;
            align-items: flex-end;
            width: 100%;
        }

        .chats-container .user-message .message-text {
            padding: 12px 18px;
            max-width: 70%;
            border-radius: 18px 18px 4px 18px;
            background: var(--gradient);
            color: #fff;
            font-size: 15px;
            line-height: 1.4;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
            word-wrap: break-word;
            word-break: break-word;
        }

        .chats-container .bot-message .message-text {
            padding: 12px 0;
            word-wrap: break-word;
            white-space: pre-line;
            line-height: 1.4;
            width: 100%;
            max-width: 100%;
            word-break: break-word;
            margin-left: 30px;
        }

        .chats-container .user-message .message-text:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
        }

        .prompt-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: calc(100% - 60px);
            padding: 16px 0;
            background: var(--primary-color);
            margin-left: 300px;
            z-index: 10;
            border-top: none;
            -webkit-app-region: no-drag;
        }

        .prompt-wrapper {
            max-width: 980px;
            margin: 0 auto;
            padding: 0 20px;
            margin-left: 300px;
        }

        .prompt-container :where(.prompt-wrapper, .prompt-form, .prompt-actions) {
            display: flex;
            gap: 12px;
            height: 56px;
            align-items: center;
        }

        .prompt-wrapper .prompt-form {
            width: 100%;
            height: 100%;
            border-radius: 130px;
            background: var(--secondary-color);
            border: 1px solid rgba(255,255,255,0.1);
        }

        .prompt-form .prompt-input {
            height: 100%;
            width: 100%;
            background: none;
            outline: none;
            border: none;
            font-size: 1rem;
            padding-left: 24px;
            color: var(--text-color);
        }

        .prompt-form .prompt-input::placeholder {
            color: var(--placeholder-color);
        }

        .prompt-wrapper button {
            width: 40px;
            height: 70%;
            border: none;
            cursor: pointer;
            border-radius: 50%;
            font-size: 1.4rem;
            flex-shrink: 0;
            color: var(--text-color);
            background: var(--secondary-color);
            transition: 0.3s ease;
        }

        .prompt-wrapper :is(button:hover, .file-icon, #cancel-file-btn) {
            background: var(--secondary-hover-color);
        }

        .prompt-form .prompt-actions {
            gap: 5px;
            margin-right: 7px;
        }

        .prompt-wrapper .prompt-form :where(.file-upload-wrapper, button, img) button {
            position: relative;
            height: 45px;
            width: 45px;
        }

        .prompt-form #send-prompt-btn {
            color: #fff;
            display: none;
            background: var(--accent-blue);
        }

        .prompt-form .prompt-input:valid ~ .prompt-actions #send-prompt-btn {
            display: block;
        }

        .prompt-form #send-prompt-btn:hover {
            background: #0264e3;
        }

        .prompt-form .file-upload-wrapper :where(button, img) {
            position: absolute;
            border-radius: 50%;
            object-fit: cover;
            display: none;
        }

        .prompt-form .file-upload-wrapper.active.img-attached img,
        .prompt-form .file-upload-wrapper.active.file-attached .file-icon,
        .prompt-form .file-upload-wrapper.active:hover #cancel-file-btn {
            display: block;
        }

        .prompt-form .file-upload-wrapper.active #add-file-btn {
            display: none;
        }

        .prompt-form #cancel-file-btn {
            color: #d62939;
        }

        .prompt-form .file-icon {
            color: var(--accent-blue);
        }

        .prompt-container .dislaimer-text {
            text-align: center;
            font-size: 0.9rem;
            padding: 16px 20px 0;
            color: var(--placeholder-color);
            margin-right:600px;
        }

        /* SIDEBAR */
        .sidebar {
            position: fixed;
            left: 0;
            top: 0;
            width: 280px;
            height: 100vh;
            background: var(--secondary-color);
            padding: 20px;
            display: flex;
            flex-direction: column;
            z-index: 100;
            border-right: 1px solid rgba(255,255,255,0.1);
            overflow-y: auto;
        }

        .sidebar::-webkit-scrollbar {
            width: 4px;
        }

        .sidebar::-webkit-scrollbar-track {
            background: transparent;
        }

        .sidebar::-webkit-scrollbar-thumb {
            background: var(--scrollbar-color);
            border-radius: 2px;
        }

        .app-logo {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px 0;
            margin-bottom: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .app-logo i {
            font-size: 34px;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 2px 6px rgba(0,0,0,0.4));
        }

        .app-logo h1 {
            font-size:40px;
            font-weight: 800;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 8px rgba(0,0,0,0.4);
            letter-spacing: 1px;
            position: relative;
            margin-left: 40px;
        }

        .quick-actions {
            margin-bottom: 20px;
        }

        .section-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--subheading-color);
            margin: 15px 0 10px 0;
        }

        .action-btn {
            width: 100%;
            padding: 12px 15px;
            margin-bottom: 8px;
            background: var(--primary-color);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            color: var(--text-color);
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .action-btn:hover {
            background: var(--secondary-hover-color);
            transform: translateY(-1px);
        }

        .action-btn.active {
            background: var(--accent-blue) !important;
            border-color: var(--accent-blue) !important;
        }

        .action-btn.active:hover {
            background: #0264e3 !important;
        }

        /* Conversation History - Simple Style */
        .conversation-history {
            flex: 1;
            display: flex;
            flex-direction: column;
            margin-bottom: 20px;
            overflow-y: auto;
        }

        .history-item {
            padding: 8px 12px;
            margin-bottom: 6px;
            background: transparent;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 13px;
            line-height: 1.3;
            color: var(--text-color);
            border-left: 2px solid transparent;
        }

        .history-item:hover {
            background: rgba(255,255,255,0.05);
            border-left: 2px solid var(--accent-blue);
        }

        .history-item.active {
            background: rgba(255,255,255,0.08);
            border-left: 2px solid var(--accent-blue);
        }

        .history-item .time {
            font-size: 10px;
            color: var(--subheading-color);
            margin-top: 2px;
            opacity: 0.7;
        }

        .clear-history {
            padding: 6px 10px;
            background: transparent;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 4px;
            color: var(--subheading-color);
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 11px;
            text-align: center;
            margin-top: 8px;
        }

        .clear-history:hover {
            background: rgba(255,255,255,0.05);
            color: var(--text-color);
        }

        .vision-controls {
            margin-top: auto;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }

        .vision-toggle {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }

        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 50px;
            height: 24px;
        }

        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255,255,255,0.1);
            transition: .4s;
            border-radius: 24px;
        }

        .slider:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 4px;
            bottom: 4px;
            background-color: var(--subheading-color);
            transition: .4s;
            border-radius: 50%;
        }

        input:checked + .slider {
            background-color: var(--accent-blue);
        }

        input:checked + .slider:before {
            transform: translateX(26px);
            background-color: white;
        }

        /* Labs Interface */
        .labs-interface {
            position: fixed;
            top: 0;
            left: 280px;
            right: 0;
            bottom: 0;
            background: var(--primary-color);
            z-index: 200;
            padding: 40px;
            overflow-y: auto;
            display: none;
        }

        .labs-interface.active {
            display: block;
        }

        .labs-header {
            margin-bottom: 40px;
        }

        .labs-title {
            font-size: 2.5rem;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .labs-subtitle {
            color: var(--subheading-color);
            font-size: 1.2rem;
        }

        .labs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }

        .lab-card {
            background: var(--secondary-color);
            border-radius: 16px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .lab-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            border-color: var(--accent-blue);
        }

        .lab-card-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }

        .lab-icon {
            width: 50px;
            height: 50px;
            border-radius: 12px;
            background: var(--gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: white;
        }

        .lab-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: var(--text-color);
        }

        .lab-description {
            color: var(--subheading-color);
            line-height: 1.5;
            margin-bottom: 20px;
        }

        .lab-features {
            list-style: none;
            margin-bottom: 20px;
        }

        .lab-features li {
            padding: 5px 0;
            color: var(--text-color);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .lab-features li:before {
            content: "•";
            color: var(--accent-blue);
            font-weight: bold;
        }

        .lab-button {
            width: 100%;
            padding: 12px;
            background: var(--accent-blue);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .lab-button:hover {
            background: #0264e3;
            transform: translateY(-2px);
        }

        .close-labs {
            position: absolute;
            top: 20px;
            right: 20px;
            background: none;
            border: none;
            color: var(--subheading-color);
            font-size: 24px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .close-labs:hover {
            color: var(--text-color);
        }

        .voice-modal {
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--secondary-color);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
            z-index: 100;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .voice-modal.active {
            opacity: 1;
            visibility: visible;
        }

        .voice-icon {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: var(--primary-color);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }

        .voice-icon i {
            font-size: 24px;
            color: var(--accent-blue);
        }

        .voice-text {
            font-size: 16px;
            font-weight: 500;
            color: var(--text-color);
        }

        .stop-button {
            position: fixed;
            bottom: 100px;
            right: 30px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: #dc3545;
            border: none;
            color: white;
            cursor: pointer;
            z-index: 100;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(220, 53, 69, 0.4);
            transition: all 0.3s ease;
            opacity: 0;
            visibility: hidden;
        }

        .stop-button.active {
            opacity: 1;
            visibility: visible;
        }

        .stop-button:hover {
            background: #c82333;
            transform: scale(1.1);
        }

        .stop-button i {
            font-size: 20px;
        }

        .settings-modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 500px;
            max-width: 90%;
            background: var(--secondary-color);
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .settings-modal.active {
            opacity: 1;
            visibility: visible;
        }

        .settings-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .settings-title {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-color);
        }

        .close-settings {
            background: none;
            border: none;
            color: var(--subheading-color);
            font-size: 20px;
            cursor: pointer;
        }

        .settings-section {
            margin-bottom: 20px;
        }

        .settings-section-title {
            font-size: 16px;
            margin-bottom: 10px;
            color: var(--accent-blue);
        }

        .settings-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .settings-label {
            font-size: 14px;
            color: var(--text-color);
        }

        .settings-input {
            width: 100px;
            padding: 8px 12px;
            background: var(--primary-color);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 6px;
            color: var(--text-color);
        }

        .settings-btn {
            padding: 10px 20px;
            background: var(--accent-blue);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .settings-btn:hover {
            background: #0264e3;
        }

        .overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            backdrop-filter: blur(5px);
            z-index: 999;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
        }

        .overlay.active {
            opacity: 1;
            visibility: visible;
        }

        .container.chat-active .app-header,
        .container.chat-active .suggestions {
            opacity: 0;
            pointer-events: none;
            transform: translateY(-20px);
            transition: all 0.5s ease;
            display: none;
        }

        .container.chat-active .chats-container {
            margin-top: 0;
            max-height: calc(100vh - 200px);
        }

        .typing-indicator {
            display: inline-flex;
            align-items: center;
            padding: 12px 18px;
            background: var(--secondary-color);
            border-radius: 18px;
            margin-right: auto;
            min-width: 80px;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .typing-dots {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-blue);
            border-radius: 50%;
            margin: 0 2px;
            animation: typingAnimation 1.4s infinite ease-in-out;
        }

        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typingAnimation {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-5px); }
        }

        .typing-text {
            color: var(--subheading-color);
            font-size: 12px;
            margin-left: 8px;
            font-style: italic;
        }

        .wake-notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--accent-blue);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            z-index: 10000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <!-- Window Controls -->
    <div class="window-controls">
        <div class="window-btn" id="minimize-btn">
            <i class="fas fa-minus"></i>
        </div>
        <div class="window-btn" id="maximize-btn">
            <i class="far fa-square"></i>
        </div>
        <div class="window-btn close" id="close-btn">
            <i class="fas fa-times"></i>
        </div>
    </div>

    <!-- Stop Button -->
    <button class="stop-button" id="stop-button">
        <i class="fas fa-stop"></i>
    </button>

    <!-- Labs Interface -->
    <div class="labs-interface" id="labs-interface">
        <button class="close-labs" id="close-labs">
            <i class="fas fa-times"></i>
        </button>
        
        <div class="labs-header">
            <h1 class="labs-title">Copilot Labs</h1>
            <p class="labs-subtitle">Discover experimental AI initiatives</p>
        </div>

        <div class="labs-grid">
            <div class="lab-card">
                <div class="lab-card-header">
                    <div class="lab-icon">
                        <i class="fas fa-headphones"></i>
                    </div>
                    <h3 class="lab-title">Audio Expressions</h3>
                </div>
                <p class="lab-description">
                    Experimental audio creation using Copilot's advanced voice generation models.
                </p>
                <ul class="lab-features">
                    <li>Voice synthesis</li>
                    <li>Audio processing</li>
                    <li>Real-time generation</li>
                </ul>
                <button class="lab-button">Explore Audio Expressions</button>
            </div>

            <div class="lab-card">
                <div class="lab-card-header">
                    <div class="lab-icon">
                        <i class="fas fa-robot"></i>
                    </div>
                    <h3 class="lab-title">AI Assistants</h3>
                </div>
                <p class="lab-description">
                    Next-generation AI assistants with enhanced capabilities and integration.
                </p>
                <ul class="lab-features">
                    <li>Multi-modal understanding</li>
                    <li>Context awareness</li>
                    <li>Personalized responses</li>
                </ul>
                <button class="lab-button">Try AI Assistants</button>
            </div>

            <div class="lab-card">
                <div class="lab-card-header">
                    <div class="lab-icon">
                        <i class="fas fa-code"></i>
                    </div>
                    <h3 class="lab-title">Code Generation</h3>
                </div>
                <p class="lab-description">
                    Advanced code generation and programming assistance tools.
                </p>
                <ul class="lab-features">
                    <li>Multi-language support</li>
                    <li>Code optimization</li>
                    <li>Debugging assistance</li>
                </ul>
                <button class="lab-button">Explore Code Tools</button>
            </div>

            <div class="lab-card">
                <div class="lab-card-header">
                    <div class="lab-icon">
                        <i class="fas fa-image"></i>
                    </div>
                    <h3 class="lab-title">Visual Creation</h3>
                </div>
                <p class="lab-description">
                    Creative tools for image generation and visual content creation.
                </p>
                <ul class="lab-features">
                    <li>Image synthesis</li>
                    <li>Style transfer</li>
                    <li>Visual enhancement</li>
                </ul>
                <button class="lab-button">Try Visual Tools</button>
            </div>

            <div class="lab-card">
                <div class="lab-card-header">
                    <div class="lab-icon">
                        <i class="fas fa-brain"></i>
                    </div>
                    <h3 class="lab-title">Neural Networks</h3>
                </div>
                <p class="lab-description">
                    Experimental neural network architectures and training techniques.
                </p>
                <ul class="lab-features">
                    <li>Model training</li>
                    <li>Architecture design</li>
                    <li>Performance optimization</li>
                </ul>
                <button class="lab-button">Explore Neural Networks</button>
            </div>

            <div class="lab-card">
                <div class="lab-card-header">
                    <div class="lab-icon">
                        <i class="fas fa-language"></i>
                    </div>
                    <h3 class="lab-title">Language Models</h3>
                </div>
                <p class="lab-description">
                    Advanced language understanding and generation capabilities.
                </p>
                <ul class="lab-features">
                    <li>Multi-lingual support</li>
                    <li>Context understanding</li>
                    <li>Creative writing</li>
                </ul>
                <button class="lab-button">Try Language Models</button>
            </div>
        </div>
    </div>

    <!-- Sidebar -->
    <div class="sidebar">
        <div class="app-logo">
            <h1>Zuno</h1>
        </div>
        
        <div class="quick-actions">
            <div class="section-title">Quick Actions</div>
            <button class="action-btn" id="voice-btn">
                <i class="fas fa-microphone"></i>
                <span>Voice Command</span>
            </button>
            <button class="action-btn" id="wake-word-btn">
                <i class="fas fa-robot"></i>
                <span id="wake-word-text">Enable Wake Word</span>
            </button>
            <!-- Email button removed -->
            <button class="action-btn" id="analyze-screen-btn">
                <i class="fas fa-desktop"></i>
                <span>Analyze Screen</span>
            </button>

            <button class="action-btn" id="settings-btn">
                <i class="fas fa-cog"></i>
                <span>Settings</span>
            </button>
        </div>

        <!-- Conversation History - Full height to bottom -->
        <div class="conversation-history">
            <div class="section-title">Conversation History</div>
            <div id="history-list">
                <!-- History items will be added here dynamically -->
            </div>
            <button class="clear-history" id="clear-history-btn">
                <i class="fas fa-trash"></i> Clear History
            </button>
        </div>
        
        <div class="vision-controls">
            <div class="section-title">Vision Controls</div>
            <div class="vision-toggle">
                <span>Screen Analysis</span>
                <label class="toggle-switch">
                    <input type="checkbox" id="vision-toggle">
                    <span class="slider"></span>
                </label>
            </div>
            <button class="action-btn" id="vision-settings-btn">
                <i class="fas fa-sliders-h"></i>
                <span>Vision Settings</span>
            </button>
        </div>
    </div>

    <!-- Main Container -->
    <div class="container" id="main-container">
        <header class="app-header">
            <h1 class="heading">Hello there</h1>
            <h2 class="sub-heading">How can I help you?</h2>
        </header>

        <ul class="suggestions">
            <li class="suggestions-item" data-prompt="Play Satinder Sartaaj Song">
                <p class="text">Play Satinder Sartaaj Song</p>
                <span class="material-symbols-rounded">music_note</span>
            </li>
            <li class="suggestions-item" data-prompt="Check wheather ui.py is present">
                <p class="text">Check wheather ui.py is present</p>
                <span class="material-symbols-rounded">lightbulb</span>
            </li>
            <li class="suggestions-item" data-prompt="Setup an environment for calculator gui ">
                <p class="text">Setup an environment for calculator gui </p>
                <span class="material-symbols-rounded">explore</span>
            </li>
            <li class="suggestions-item" data-prompt="Create google meet with pictocreatives@gmail.com tomorrow at 9 pm">
                <p class="text">Create google meet with pictocreatives@gmail.com tomorrow at 9 pm</p>
                <span class="material-symbols-rounded">mail</span>
            </li>
        </ul>

        <!-- Chat Container -->
        <div class="chats-container" id="chats-container">
            <!-- Messages will appear here -->
        </div>

        <!-- Prompt Container -->
        <div class="prompt-container">
            <div class="prompt-wrapper">
                <form action="#" class="prompt-form" id="prompt-form">
                    <input type="text" placeholder="Ask Zuno" class="prompt-input" id="prompt-input" required>
                    <div class="prompt-actions">
                        <div class="file-upload-wrapper">
                            <button id="add-file-btn" type="button" class="material-symbols-rounded">attach_file</button>
                        </div>
                        <button id="send-prompt-btn" class="material-symbols-rounded">arrow_upward_alt</button>
                    </div>
                </form>
                <button id="voice-input-btn" class="material-symbols-rounded">mic</button>
            </div>
            <p class="dislaimer-text">Zuno can make mistakes, so double check it.</p>
        </div>
    </div>

    <!-- Voice Modal -->
    <div class="voice-modal" id="voice-modal">
        <div class="voice-icon">
            <i class="fas fa-microphone"></i>
        </div>
        <div class="voice-text" id="voice-text">Listening...</div>
    </div>

    <!-- Settings Modal -->
    <div class="settings-modal" id="settings-modal">
        <div class="settings-header">
            <h3 class="settings-title">Settings</h3>
            <button class="close-settings" id="close-settings-btn">
                <i class="fas fa-times"></i>
            </button>
        </div>
        
        <div class="settings-section">
            <h4 class="settings-section-title">Voice</h4>
            <div class="settings-row">
                <span class="settings-label">Voice Speed</span>
                <input type="range" class="settings-input" id="voice-speed" min="0.5" max="2" step="0.1" value="1">
            </div>
        </div>
        
        <button class="settings-btn" id="save-settings-btn">Save Settings</button>
    </div>

    <!-- Overlay -->
    <div class="overlay" id="overlay"></div>

    <script>
        // DOM Elements
        const mainContainer = document.getElementById('main-container');
        const chatsContainer = document.getElementById('chats-container');
        const promptForm = document.getElementById('prompt-form');
        const promptInput = document.getElementById('prompt-input');
        const sendBtn = document.getElementById('send-prompt-btn');
        const voiceInputBtn = document.getElementById('voice-input-btn');
        const voiceModal = document.getElementById('voice-modal');
        const voiceText = document.getElementById('voice-text');
        const suggestions = document.querySelector('.suggestions');
        const appHeader = document.querySelector('.app-header');
        const minimizeBtn = document.getElementById('minimize-btn');
        const maximizeBtn = document.getElementById('maximize-btn');
        const closeBtn = document.getElementById('close-btn');
        const visionToggle = document.getElementById('vision-toggle');
        const settingsBtn = document.getElementById('settings-btn');
        const settingsModal = document.getElementById('settings-modal');
        const closeSettingsBtn = document.getElementById('close-settings-btn');
        const saveSettingsBtn = document.getElementById('save-settings-btn');
        const overlay = document.getElementById('overlay');
        const analyzeScreenBtn = document.getElementById('analyze-screen-btn');
        const voiceBtn = document.getElementById('voice-btn');
        const wakeWordBtn = document.getElementById('wake-word-btn');
        const wakeWordText = document.getElementById('wake-word-text');
        const historyList = document.getElementById('history-list');
        const clearHistoryBtn = document.getElementById('clear-history-btn');
        const labsBtn = document.getElementById('labs-btn');
        const labsInterface = document.getElementById('labs-interface');
        const closeLabs = document.getElementById('close-labs');
        const stopButton = document.getElementById('stop-button');
        
        // State
        let isListening = false;
        let isMaximized = false;
        let isVisionActive = false;
        let autoScrollEnabled = true;
        let hasMessageSent = false;
        let wakeWordEnabled = false;
        let conversationHistory = [];
        let isSpeaking = false;

        // Initialize
        function init() {
            setupEventListeners();
            loadSettings();
            loadConversationHistory();
        }

        function setupEventListeners() {
            // Text input
            promptInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = `${this.scrollHeight}px`;
            });
            
            promptInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            // Form submission
            promptForm.addEventListener('submit', (e) => {
                e.preventDefault();
                sendMessage();
            });

            // Voice input
            voiceInputBtn.addEventListener('click', toggleVoiceRecognition);
            voiceBtn.addEventListener('click', toggleVoiceRecognition);

            // Wake word
            wakeWordBtn.addEventListener('click', toggleWakeWord);

            // Labs
            labsBtn.addEventListener('click', openLabs);
            closeLabs.addEventListener('click', closeLabsInterface);

            // Stop button
            stopButton.addEventListener('click', stopSpeaking);

            // Suggestions
            document.querySelectorAll('.suggestions-item').forEach(item => {
                item.addEventListener('click', () => {
                    const prompt = item.getAttribute('data-prompt');
                    promptInput.value = prompt;
                    sendMessage();
                });
            });

            // Window controls
            minimizeBtn.addEventListener('click', () => {
                window.pywebview.api.minimize_window();
            });
            
            maximizeBtn.addEventListener('click', () => {
                if (isMaximized) {
                    window.pywebview.api.restore_window();
                    maximizeBtn.innerHTML = '<i class="far fa-square"></i>';
                } else {
                    window.pywebview.api.maximize_window();
                    maximizeBtn.innerHTML = '<i class="far fa-window-restore"></i>';
                }
                isMaximized = !isMaximized;
            });
            
            closeBtn.addEventListener('click', () => {
                window.pywebview.api.close_window();
            });

            // Vision toggle
            visionToggle.addEventListener('change', toggleVision);

            // Settings
            settingsBtn.addEventListener('click', openSettings);
            closeSettingsBtn.addEventListener('click', closeSettings);
            saveSettingsBtn.addEventListener('click', saveSettings);

            // Analyze screen
            analyzeScreenBtn.addEventListener('click', analyzeScreen);

            // Overlay
            overlay.addEventListener('click', () => {
                closeSettings();
                closeLabsInterface();
            });

            // Clear history
            clearHistoryBtn.addEventListener('click', clearConversationHistory);

            // Auto scroll detection
            chatsContainer.addEventListener('scroll', () => {
                const nearBottom = chatsContainer.scrollHeight - chatsContainer.scrollTop - chatsContainer.clientHeight < 50;
                autoScrollEnabled = nearBottom;
            });
        }

        // Labs Functions
        function openLabs() {
            labsInterface.classList.add('active');
            overlay.classList.add('active');
        }

        function closeLabsInterface() {
            labsInterface.classList.remove('active');
            overlay.classList.remove('active');
        }

        // Conversation History Functions
        function loadConversationHistory() {
            try {
                // Get history from Python backend
                window.pywebview.api.get_conversation_history().then(savedHistory => {
                    if (savedHistory) {
                        conversationHistory = JSON.parse(savedHistory);
                        renderConversationHistory();
                    }
                }).catch(error => {
                    console.log('Error loading conversation history:', error);
                    conversationHistory = [];
                    renderConversationHistory();
                });
            } catch (e) {
                console.log('Error loading conversation history:', e);
                conversationHistory = [];
                renderConversationHistory();
            }
        }

        function saveConversationHistory() {
            try {
                // Save history to Python backend
                window.pywebview.api.save_conversation_history(JSON.stringify(conversationHistory));
            } catch (e) {
                console.log('Error saving conversation history:', e);
            }
        }

        function addToConversationHistory(userMessage, botResponse) {
            const historyItem = {
                id: Date.now(),
                userMessage: userMessage,
                botResponse: botResponse,
                timestamp: new Date().toLocaleTimeString(),
                date: new Date().toLocaleDateString()
            };
            
            conversationHistory.unshift(historyItem);
            
            // Keep only last 20 conversations
            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(0, 20);
            }
            
            saveConversationHistory();
            renderConversationHistory();
        }

        function renderConversationHistory() {
            historyList.innerHTML = '';
            
            if (conversationHistory.length === 0) {
                const emptyMsg = document.createElement('div');
                emptyMsg.className = 'history-item';
                emptyMsg.textContent = 'No conversations yet';
                emptyMsg.style.opacity = '0.6';
                emptyMsg.style.fontStyle = 'italic';
                historyList.appendChild(emptyMsg);
                return;
            }
            
            conversationHistory.forEach(item => {
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                historyItem.innerHTML = `
                    ${item.userMessage.substring(0, 50)}${item.userMessage.length > 50 ? '...' : ''}
                    <div class="time">${item.timestamp}</div>
                `;
                
                historyItem.addEventListener('click', () => {
                    // Clear current chat and load this conversation
                    chatsContainer.innerHTML = '';
                    
                    // Add user message
                    const userMsgEl = createMessageElement(item.userMessage, 'user');
                    chatsContainer.appendChild(userMsgEl);
                    
                    // Add bot response
                    const botMsgEl = createMessageElement(item.botResponse, 'bot');
                    chatsContainer.appendChild(botMsgEl);
                    
                    smoothScrollToBottom();
                    hideWelcomeElements();
                });
                
                historyList.appendChild(historyItem);
            });
        }

        function clearConversationHistory() {
            if (confirm('Are you sure you want to clear all conversation history?')) {
                conversationHistory = [];
                saveConversationHistory();
                renderConversationHistory();
                // Also clear the current chat
                chatsContainer.innerHTML = '';
                mainContainer.classList.remove('chat-active');
                hasMessageSent = false;
            }
        }

        function speak(text) {
            isSpeaking = true;
            stopButton.classList.add('active');
            window.pywebview.api.speak(text).then(() => {
                isSpeaking = false;
                stopButton.classList.remove('active');
            });
        }

        function stopSpeaking() {
            if (isSpeaking) {
                window.pywebview.api.stop_speaking();
                isSpeaking = false;
                stopButton.classList.remove('active');
            }
        }

        function smoothScrollToBottom() {
            if (!autoScrollEnabled) return;
            chatsContainer.scrollTo({
                top: chatsContainer.scrollHeight,
                behavior: 'smooth'
            });
        }

        function hideWelcomeElements() {
            if (!hasMessageSent) {
                mainContainer.classList.add('chat-active');
                hasMessageSent = true;
                
                setTimeout(() => {
                    chatsContainer.style.maxHeight = 'calc(100vh - 200px)';
                    chatsContainer.style.marginTop = '20px';
                }, 500);
            }
        }

        function createMessageElement(content, sender) {
            const messageEl = document.createElement('div');
            messageEl.className = `message ${sender}-message`;
            
            if (sender === 'user') {
                messageEl.innerHTML = `<p class="message-text">${content}</p>`;
            } else {
                messageEl.innerHTML = `
                    <div class="avatar">
                        <img src="images/bot.png" class="avatar">
                    </div>
                    <p class="message-text">${content}</p>
                `;
            }
            
            return messageEl;
        }

        function showTyping() {
            const typingEl = document.createElement('div');
            typingEl.className = 'message bot-message loading';
            typingEl.id = 'typing-indicator';
            typingEl.innerHTML = `
                <div class="avatar">
                    <img src="images/bot.png" class="avatar">
                </div>
                <div class="typing-indicator">
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                    <span class="typing-text">Thinking...</span>
                </div>
            `;
            chatsContainer.appendChild(typingEl);
            smoothScrollToBottom();
        }

        function hideTyping() {
            const existingTyping = document.getElementById('typing-indicator');
            if (existingTyping) {
                existingTyping.remove();
            }
        }

        function typeText(element, text, delay = 15) {
            let index = 0;
            element.textContent = '';
            
            const interval = setInterval(() => {
                if (index < text.length) {
                    element.textContent += text.charAt(index);
                    index++;
                    smoothScrollToBottom();
                } else {
                    clearInterval(interval);
                }
            }, delay);
        }

        async function sendMessage() {
            const text = promptInput.value.trim();
            if (!text) return;

            hideWelcomeElements();
            
            const userMsgEl = createMessageElement(text, 'user');
            chatsContainer.appendChild(userMsgEl);
            promptInput.value = '';
            promptInput.style.height = 'auto';
            
            smoothScrollToBottom();

            showTyping();

            try {
                const response = await window.pywebview.api.send_message(text);
                const data = JSON.parse(response);
                
                hideTyping();
                
                const botMsgEl = createMessageElement('', 'bot');
                chatsContainer.appendChild(botMsgEl);
                const messageText = botMsgEl.querySelector('.message-text');
                
                if (data.ok) {
                    typeText(messageText, data.content);
                    speak(data.content);
                    // Add to conversation history
                    addToConversationHistory(text, data.content);
                } else {
                    typeText(messageText, `Error: ${data.error || 'Unknown error'}`);
                    speak("Sorry, there was an error.");
                }
            } catch (error) {
                hideTyping();
                const botMsgEl = createMessageElement('', 'bot');
                chatsContainer.appendChild(botMsgEl);
                const messageText = botMsgEl.querySelector('.message-text');
                typeText(messageText, `System error: ${error}`);
                speak("Sorry, there was a system error.");
            }
        }

        function toggleVoiceRecognition() {
            if (!isListening) {
                startVoiceRecognition();
            } else {
                stopVoiceRecognition();
            }
        }

        function startVoiceRecognition() {
            isListening = true;
            voiceInputBtn.classList.add('listening');
            voiceModal.classList.add('active');
            
            window.pywebview.api.start_listening().then((result) => {
                if (result && typeof result === 'string' && result.trim().length > 0) {
                    promptInput.value = result;
                    sendMessage();
                }
                stopVoiceRecognition();
            });
        }

        function stopVoiceRecognition() {
            isListening = false;
            voiceInputBtn.classList.remove('listening');
            voiceModal.classList.remove('active');
        }

        function toggleWakeWord() {
            wakeWordEnabled = !wakeWordEnabled;
            
            if (wakeWordEnabled) {
                wakeWordBtn.classList.add('active');
                wakeWordText.textContent = 'Disable Wake Word';
                window.pywebview.api.toggle_wake_word(true);
                showNotification('Wake word enabled - Say "Hey Zuno" to activate me.');
            } else {
                wakeWordBtn.classList.remove('active');
                wakeWordText.textContent = 'Enable Wake Word';
                window.pywebview.api.toggle_wake_word(false);
                showNotification('Wake word disabled');
            }
        }

        function showNotification(message) {
            const notification = document.createElement('div');
            notification.className = 'wake-notification';
            notification.textContent = message;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 3000);
        }

        function toggleVision() {
            isVisionActive = visionToggle.checked;
            if (isVisionActive) {
                startVision();
            } else {
                stopVision();
            }
            saveSettings();
        }

        function startVision() {
            console.log('Vision activated');
        }

        function stopVision() {
            console.log('Vision deactivated');
        }

        async function analyzeScreen() {
            showTyping();
            
            try {
                const result = await window.pywebview.api.analyze_screen();
                const data = JSON.parse(result);
                
                hideTyping();
                
                const botMsgEl = createMessageElement('', 'bot');
                chatsContainer.appendChild(botMsgEl);
                const messageText = botMsgEl.querySelector('.message-text');
                
                if (data.ok) {
                    typeText(messageText, data.description);
                    smoothScrollToBottom();
                } else {
                    typeText(messageText, `Vision Error: ${data.error}`);
                }
            } catch (error) {
                hideTyping();
                const botMsgEl = createMessageElement('', 'bot');
                chatsContainer.appendChild(botMsgEl);
                const messageText = botMsgEl.querySelector('.message-text');
                typeText(messageText, `Vision System Error: ${error}`);
            }
        }

        function openSettings() {
            settingsModal.classList.add('active');
            overlay.classList.add('active');
        }

        function closeSettings() {
            settingsModal.classList.remove('active');
            overlay.classList.remove('active');
        }

        function saveSettings() {
            closeSettings();
        }

        function loadSettings() {
            // Load settings implementation
        }

        // Initialize
        window.addEventListener('load', init);
    </script>
</body>
</html>
"""

class Api:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.listening = False
        self.vision_active = False
        self.vision_interval = VISION_INTERVAL
        
        # Wake word detector
        self.wake_detector = WakeWordDetector()
        self.simple_vad = SimpleVAD()
        self.wake_word_enabled = False
        
        # Speech control
        self.current_speech_process = None
        
        # Convert images
        self.f1_base64 = self.image_to_base64(r"images/f1.jpg")
        self.f2_base64 = self.image_to_base64(r"images/f2.jpg")
        self.f3_base64 = self.image_to_base64(r"images/f3.jpg")
        
        # Window reference
        self.window = None
        
        # Conversation history storage
        self.history_file = Path("conversation_history.json")
        self.conversation_history = self.load_conversation_history()
    
    def load_conversation_history(self):
        """Load conversation history from file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading conversation history: {e}")
        return []
    
    def save_conversation_history(self, history_json):
        """Save conversation history to file"""
        try:
            history = json.loads(history_json)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving conversation history: {e}")
    
    def get_conversation_history(self):
        """Get conversation history for frontend"""
        return json.dumps(self.conversation_history)
    
    def image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return ""

    def set_window(self, window):
        """Set the window reference for API calls"""
        self.window = window

    def load_chatbot(self):
        """Load the chatbot interface"""
        if self.window:
            self.window.load_html(CHATBOT_HTML)
            # Speak welcome message for chatbot
            threading.Thread(target=self.speak, args=("Welcome to Zuno AI! How can I assist you today?",)).start()

    def send_message(self, message):
        try:
            result = handle_intent(message.strip())
            return json.dumps({
                "ok": True,
                "content": result
            })
        except Exception as e:
            return json.dumps({
                "ok": False,
                "error": str(e)
            })

    def minimize_window(self):
        if self.window:
            self.window.minimize()
    
    def maximize_window(self):
        if self.window:
            self.window.maximize()
    
    def restore_window(self):
        if self.window:
            self.window.restore()
    
    def close_window(self):
        if self.window:
            self.window.destroy()
        sys.exit()
    
    def start_listening(self):
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {str(e)}"

    def speak(self, text: str):
        try:
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_path = fp.name
            tts.save(temp_path)
            
            try:
                self.current_speech_process = subprocess.Popen(
                    ["mpg123", temp_path], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
                self.current_speech_process.wait()
            except FileNotFoundError:
                try:
                    self.current_speech_process = subprocess.Popen(
                        ["mpv", temp_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    self.current_speech_process.wait()
                except FileNotFoundError:
                    self.current_speech_process = subprocess.Popen(
                        ["ffplay", "-nodisp", "-autoexit", temp_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    self.current_speech_process.wait()
            
            os.remove(temp_path)
            self.current_speech_process = None
        except Exception as e:
            print(f"TTS Error: {e}")
            self.current_speech_process = None

    def stop_speaking(self):
        """Stop current speech"""
        if self.current_speech_process:
            try:
                self.current_speech_process.terminate()
                self.current_speech_process.wait(timeout=2)
            except:
                try:
                    self.current_speech_process.kill()
                except:
                    pass
            self.current_speech_process = None

    def toggle_wake_word(self, enabled):
        self.wake_word_enabled = enabled
        if enabled:
            success = self.wake_detector.start_listening(self.on_wake_word_detected)
            if not success:
                print("Using simple voice activation")
                self.simple_vad.start_listening(self.on_wake_word_detected)
            self.speak("Wake word detection activated. Say 'Hey Zuno' to activate me.")
        else:
            self.wake_detector.stop_listening()
            self.simple_vad.stop_listening()
            self.speak("Wake word detection deactivated.")

    def on_wake_word_detected(self):
        print("Wake word detected - activating voice recognition")
        self.speak("Yes? I'm listening...")
        
        def start_voice():
            result = self.start_listening()
            if result and isinstance(result, str) and result.strip():
                response = self.send_message(result)
                data = json.loads(response)
                if data.get("ok"):
                    self.speak(data["content"])
        
        thread = threading.Thread(target=start_voice)
        thread.daemon = True
        thread.start()

    def analyze_screen(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = VISION_SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
            
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
                img.save(filename)
            
            if BLUR_SENSITIVE_REGION:
                img = Image.open(filename)
                img = self.blur_vision_region(img, box=SENSITIVE_BOX)
                img.save(filename)
            
            b64_image = self.encode_image_base64(filename)
            
            payload = {
                "model": VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe everything visible in this screenshot in detail. Mention objects, text, windows, and any notable activity."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                            }
                        ],
                    }
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {VISION_API_KEY}",
                "Content-Type": "application/json",
            }
            
            response = requests.post(VISION_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_json = response.json()
            
            description = self.extract_description_from_response(response_json)
            
            if DELETE_SCREENSHOT_AFTER_SEND and filename.exists():
                filename.unlink()
            
            return json.dumps({
                "ok": True,
                "description": description,
                "image_data": b64_image if not DELETE_SCREENSHOT_AFTER_SEND else None
            })
            
        except Exception as e:
            return json.dumps({
                "ok": False,
                "error": str(e)
            })

    def blur_vision_region(self, img: Image.Image, box=None, radius=20) -> Image.Image:
        if box is None:
            w, h = img.size
            bw, bh = int(w * 0.3), int(h * 0.2)
            left = (w - bw) // 2
            top = (h - bh) // 2
            box = (left, top, left + bw, top + bh)

        region = img.crop(box)
        blurred = region.filter(ImageFilter.GaussianBlur(radius))
        img.paste(blurred, box)
        return img

    def encode_image_base64(self, path: Path) -> str:
        with open(path, "rb") as f:
            raw = f.read()
        return base64.b64encode(raw).decode("utf-8")

    def extract_description_from_response(self, resp_json: dict) -> str:
        try:
            choices = resp_json.get("choices", [])
            if not choices:
                return json.dumps(resp_json, indent=2)
            msg = choices[0].get("message", {})
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        texts.append(block["text"])
                    elif isinstance(block, str):
                        texts.append(block)
                if texts:
                    return "\n".join(texts)
            return json.dumps(msg, indent=2)
        except Exception as e:
            return f"Failed to parse response: {e}\nFull response: {json.dumps(resp_json)[:2000]}"

def start_ui():
    api = Api()
    window = webview.create_window(
        "👑 Zuno AI - Premium Assistant",
        html=UNIVERSE_HTML,
        width=1200,
        height=800,
        min_size=(800, 600),
        text_select=True,
        js_api=api,
        frameless=True
    )
    api.set_window(window)
    # Start in full screen and hide debug console
    webview.start(debug=False)

if __name__ == "__main__":
    # Create planets directory if it doesn't exist
    planets_dir = Path("images/planets")
    planets_dir.mkdir(parents=True, exist_ok=True)
    
    # Suppress console output for audio players
    try:
        subprocess.run(["which", "mpg123"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        try:
            subprocess.run(["which", "mpv"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            try:
                subprocess.run(["which", "ffplay"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                print("Warning: No compatible audio player found (mpg123, mpv, or ffplay). TTS will not work.")
    
    start_ui()
