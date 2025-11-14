import os
import fitz
import labs
import re
import subprocess
import sys
import difflib
import mimetypes
import docx
from PIL import Image
import pytesseract
import shutil
import webbrowser
import requests
import json
import platform
import glob
import send2trash
import threading
import time
import base64
import pickle
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from pathlib import Path
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import psutil
import shlex
import queue
import tempfile
import playsound
from gtts import gTTS
import speech_recognition as sr
import json5  # For parsing lenient JSON from LLM
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from labs import (
    execute_file,
    run_code_lab,
    run_hallucination_lab
)

# ========== CONFIGURATION ==========
CHROMEDRIVER_PATH = "/usr/bin/chromedriver"
BRAVE_PATH = "/usr/bin/brave"
CHROME_PATH = "/usr/bin/google-chrome"
WHATSAPP_PROFILE_DIR = "/home/amitr/.config/whatsapp_final_session"

# ========== EMAIL CONFIGURATION ==========
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "rajputamit1508@gmail.com"  # 👈 YOUR GMAIL
EMAIL_PASSWORD = "duyu ooxk knsa wnpo"   # 👈 YOUR APP PASSWORD

# ========== GOOGLE CALENDAR CONFIGURATION ==========
CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]
CALENDAR_CREDENTIALS_FILE = 'credentials.json'
CALENDAR_TOKEN_FILE = 'calendar_token.pickle'

# ========== PERMANENT MEMORY CONFIGURATION ==========
CHAT_HISTORY_FILE = "chat_history.json"
FACTS_FILE = "permanent_facts.json"
SESSION_CONTEXT_FILE = "session_context.json"

# ========== BROWSER CONFIGURATION ==========
PROFILE_PATH = "/home/anas/.config/BraveSoftware/Brave-Browser/Default"
BRAVE_BINARY = "/usr/bin/brave"

# --- Browser Session State ---
browser_session = {
    "driver": None,
    "current_url": None,
    "last_search": None,
    "last_query": None,
}

try:
    from llm_agent import get_llm_response
except ImportError:
    # If llm_agent.py is not found, use this placeholder for testing
    print("WARNING: 'llm_agent.py' not found. Using a simple placeholder. Agent will have limited intelligence.")
    def get_llm_response(prompt, code_only=False):
        print("--- (Placeholder) PROMPT ---")
        print(prompt[-300:]) # Print last 300 chars
        print("--------------")
        
        user_msg = prompt.split("User message:")[-1].strip().lower()

        if "my name is" in user_msg:
            name = user_msg.split("is")[-1].strip()
            return f'{{"action":"remember_fact", "key":"name", "value":"{name.title()}"}}'
        if "what is my name" in user_msg:
            return '{"action":"get_fact", "key":"name"}'
        if "play" in user_msg and "song" in user_msg:
            song_name = user_msg.replace("play", "").replace("song", "").strip()
            return f'{{"action":"play_music", "song":"{song_name}"}}'
        if "hi" in user_msg or "hello" in user_msg:
            return '{"action":"chat", "message":"Hello! How can I help you?"}'
        if "create a file" in user_msg:
            return '{"action":"create_file", "folder_path":"~/Documents", "filename":"test.txt", "content":"Hello!"}'
        
        # Default fallback
        return '{"action":"chat", "message":"I am not sure how to respond to that."}'
# --- End of LLM Placeholder ---

class ZunoWakeWord:
    def __init__(self):
        self.is_listening = False
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.wake_word = "hey zuno"
        self.command_queue = queue.Queue()
        self.is_awake = False
        self.last_wake_time = 0
        self.wake_timeout = 10  # seconds

        # Adjust for ambient noise
        print("🎙️ Calibrating microphone for ambient noise...")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("✅ Microphone calibrated!")
        except Exception as e:
            print(f"❌ Microphone error: {e}. Voice input may not work.")

    def listen_for_wake_word(self):
        """Background thread that listens for 'Hey Zuno'"""
        while self.is_listening:
            try:
                # print("🔇 Sleeping... waiting for 'Hey Zuno'")
                with self.microphone as source:
                    # Listen for wake word
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)

                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"🎯 Heard: {text}")

                    if self.wake_word in text:
                        self.wake_up()
                        # Extract command if said together
                        command = text.replace(self.wake_word, "").strip()
                        if command:
                            self.command_queue.put(command)

                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"❌ Speech recognition error: {e}")
                    time.sleep(1)

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"❌ Wake word listener error: {e}")
                time.sleep(1)

    def listen_for_command(self):
        """Listen for command after wake word"""
        try:
            print("🎙️ Listening for command...")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)

            command = self.recognizer.recognize_google(audio)
            print(f"📝 Command: {command}")
            return command

        except sr.UnknownValueError:
            return None
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"❌ Command listening error: {e}")
            return None

    def wake_up(self):
        """Activate Zuno and speak confirmation"""
        self.is_awake = True
        self.last_wake_time = time.time()
        print("🔊 Zuno activated!")
        speak_response("Yes? How can I help you?")

    def should_sleep(self):
        """Check if should go back to sleep"""
        if self.is_awake and (time.time() - self.last_wake_time > self.wake_timeout):
            self.is_awake = False
            speak_response("Going to sleep. Say Hey Zuno when you need me.")
            return True
        return False

    def start_listening(self):
        """Start the wake word detection"""
        self.is_listening = True
        wake_thread = threading.Thread(target=self.listen_for_wake_word, daemon=True)
        wake_thread.start()
        print("🚀 Wake word detection started! Say 'Hey Zuno'")

    def stop_listening(self):
        """Stop the wake word detection"""
        self.is_listening = False

# Global wake word detector instance
wake_detector = ZunoWakeWord()

def speak_response(text: str):
    if not text:
        return
    try:
        # Generate speech
        tts = gTTS(text=text, lang='en')

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
        tts.save(temp_path)

        # Play the audio
        playsound.playsound(temp_path)

        # Remove the temp file
        os.remove(temp_path)

    except Exception as e:
        print(f"❌ TTS Error: {e}")

# ========== GOOGLE MEET SCHEDULING FUNCTIONS ==========

def get_calendar_service():
    """Google Calendar service setup"""
    creds = None

    # Load stored OAuth token if available
    if os.path.exists(CALENDAR_TOKEN_FILE):
        with open(CALENDAR_TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    # Refresh token OR run full authentication
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CALENDAR_CREDENTIALS_FILE):
                return None, "❌ credentials.json file missing for Google Calendar"
            
            flow = InstalledAppFlow.from_client_secrets_file(
                CALENDAR_CREDENTIALS_FILE, CALENDAR_SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save updated token
        with open(CALENDAR_TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    service = build("calendar", "v3", credentials=creds)
    return service, None

def schedule_meet(title, date, start_time, end_time, attendees_list=None, timezone="Asia/Kolkata"):
    """Schedule Google Meet meeting"""
    if attendees_list is None:
        attendees_list = []

    service, error = get_calendar_service()
    if error:
        return None, error

    try:
        # Convert input to RFC3339 format
        start = f"{date}T{start_time}:00"
        end   = f"{date}T{end_time}:00"

        attendees = [{"email": email} for email in attendees_list]

        event = {
            "summary": title,
            "start": {
                "dateTime": start,
                "timeZone": timezone
            },
            "end": {
                "dateTime": end,
                "timeZone": timezone
            },
            "attendees": attendees,

            # This block is REQUIRED for auto-creating Google Meet links
            "conferenceData": {
                "createRequest": {
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                    "requestId": "meet-" + str(datetime.now().timestamp())
                }
            },
        }

        created_event = (
            service.events()
            .insert(
                calendarId="primary",
                body=event,
                conferenceDataVersion=1,  # IMPORTANT
                sendUpdates="all",        # send email invites automatically
            )
            .execute()
        )

        # Safely extract Meet link
        meet_link = None
        try:
            meet_link = created_event["conferenceData"]["entryPoints"][0]["uri"]
        except Exception:
            meet_link = created_event.get("hangoutLink", "No Meet Link Found")

        return {
            "event_id": created_event["id"],
            "meet_link": meet_link,
            "start": start,
            "end": end,
            "title": title
        }, None
        
    except Exception as e:
        return None, f"❌ Meet scheduling failed: {str(e)}"

def parse_meeting_command(user_message):
    """Manual parsing of meeting commands"""
    print(f"🔍 Parsing meeting command: {user_message}")
    
    # Default values
    title = "Meeting"
    date = datetime.now().strftime("%Y-%m-%d")
    start_time = "14:00"
    end_time = "15:00"
    attendees = []
    
    # Extract title
    title_patterns = [
        r'called\s+([^,.]+)',
        r'for\s+([^,.]+)',
        r'meeting\s+for\s+([^,.]+)',
        r'schedule\s+([^,.]+)'
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            break
    
    # Extract date
    today = datetime.now()
    if 'tomorrow' in user_message.lower():
        date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif 'today' in user_message.lower():
        date = today.strftime("%Y-%m-%d")
    elif 'monday' in user_message.lower():
        days_ahead = (0 - today.weekday() + 7) % 7
        if days_ahead == 0: days_ahead = 7
        date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    elif 'tuesday' in user_message.lower():
        days_ahead = (1 - today.weekday() + 7) % 7
        if days_ahead == 0: days_ahead = 7
        date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    elif 'wednesday' in user_message.lower():
        days_ahead = (2 - today.weekday() + 7) % 7
        if days_ahead == 0: days_ahead = 7
        date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    elif 'thursday' in user_message.lower():
        days_ahead = (3 - today.weekday() + 7) % 7
        if days_ahead == 0: days_ahead = 7
        date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    elif 'friday' in user_message.lower():
        days_ahead = (4 - today.weekday() + 7) % 7
        if days_ahead == 0: days_ahead = 7
        date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    elif 'saturday' in user_message.lower():
        days_ahead = (5 - today.weekday() + 7) % 7
        if days_ahead == 0: days_ahead = 7
        date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    elif 'sunday' in user_message.lower():
        days_ahead = (6 - today.weekday() + 7) % 7
        if days_ahead == 0: days_ahead = 7
        date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    # Extract time
    time_pattern = r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)?'
    times = re.findall(time_pattern, user_message)
    
    if len(times) >= 1:
        hour, minute, period = times[0]
        hour = int(hour)
        minute = int(minute) if minute else 0
        
        if period and period.upper() == 'PM' and hour != 12:
            hour += 12
        elif period and period.upper() == 'AM' and hour == 12:
            hour = 0
            
        start_time = f"{hour:02d}:{minute:02d}"
        
        # Calculate end time (default 1 hour)
        end_hour = (hour + 1) % 24
        end_time = f"{end_hour:02d}:{minute:02d}"
    
    if len(times) >= 2:
        hour, minute, period = times[1]
        hour = int(hour)
        minute = int(minute) if minute else 0
        
        if period and period.upper() == 'PM' and hour != 12:
            hour += 12
        elif period and period.upper() == 'AM' and hour == 12:
            hour = 0
            
        end_time = f"{hour:02d}:{minute:02d}"
    
    # Extract duration
    duration_pattern = r'(\d+)\s*(hour|minute|hr|min)'
    duration_match = re.search(duration_pattern, user_message, re.IGNORECASE)
    if duration_match:
        duration = int(duration_match.group(1))
        unit = duration_match.group(2).lower()
        
        start_hour, start_minute = map(int, start_time.split(':'))
        
        if 'hour' in unit or 'hr' in unit:
            end_hour = (start_hour + duration) % 24
            end_time = f"{end_hour:02d}:{start_minute:02d}"
        elif 'minute' in unit or 'min' in unit:
            total_minutes = start_hour * 60 + start_minute + duration
            end_hour = (total_minutes // 60) % 24
            end_minute = total_minutes % 60
            end_time = f"{end_hour:02d}:{end_minute:02d}"
    
    # Extract attendees
    email_pattern = r'[\w\.-]+@[\w\.-]+'
    attendees = re.findall(email_pattern, user_message)
    
    print(f"✅ Parsed: Title='{title}', Date='{date}', Start='{start_time}', End='{end_time}', Attendees={attendees}")
    
    return {
        "action": "schedule_meet",
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "attendees": attendees
    }

def do_schedule_meet(title, date, start_time, end_time, attendees):
    """Execute Google Meet scheduling"""
    if not title or not date or not start_time or not end_time:
        return "❌ Missing required fields for scheduling a meeting. Need title, date, start_time, end_time."

    try:
        print(f"📅 Scheduling meeting: {title}")
        print(f"📅 Date: {date}, Time: {start_time} to {end_time}")
        if attendees:
            print(f"👥 Attendees: {', '.join(attendees)}")
        
        result, error = schedule_meet(
            title=title,
            date=date,
            start_time=start_time,
            end_time=end_time,
            attendees_list=attendees,
        )

        if error:
            return error

        meet_link = result["meet_link"]
        event_id = result["event_id"]

        response = f"""✅ Google Meet Scheduled Successfully!

📋 Meeting Details:
• Title: {title}
• Date: {date}
• Time: {start_time} to {end_time}
• Meet Link: {meet_link}
• Event ID: {event_id}"""

        if attendees:
            response += f"\n• Attendees: {', '.join(attendees)}"
            
        response += "\n\n📧 Calendar invites have been sent to all attendees!"
        return response

    except Exception as e:
        return f"❌ Failed to schedule Google Meet: {e}"

# ========== INTERACTIVE EMAIL FUNCTIONS ==========

def ask_yes_no_question(question):
    """Yes/No question पूछें"""
    print(f"❓ {question} (yes/no)")
    
    if wake_detector.is_awake:
        speak_response(question)
        # Voice mode
        response = wake_detector.listen_for_command()
        if response:
            response = response.lower().strip()
            if 'yes' in response or 'haan' in response or 'ok' in response:
                return True
            elif 'no' in response or 'nahi' in response:
                return False
    else:
        # Text mode
        response = input("👉 (yes/no): ").lower().strip()
        return response in ['yes', 'y', 'haan', 'ok', '1']

def ask_multiple_choice(question, options):
    """Multiple choice question पूछें"""
    print(f"❓ {question}")
    for i, option in enumerate(options, 1):
        print(f"   {i}. {option}")
    
    if wake_detector.is_awake:
        speak_response(question)
        response = wake_detector.listen_for_command()
        if response:
            # Try to match response with options
            response_lower = response.lower()
            for i, option in enumerate(options, 1):
                if option.lower() in response_lower:
                    return option
            return options[0]  # Default to first option
    else:
        response = input("👉 Enter choice: ").strip()
        if response.isdigit() and 1 <= int(response) <= len(options):
            return options[int(response)-1]
        return options[0]  # Default

def get_user_input(prompt):
    """User input लें voice या text mode में"""
    print(f"📝 {prompt}")
    
    if wake_detector.is_awake:
        speak_response(prompt)
        response = wake_detector.listen_for_command()
        return response or ""
    else:
        return input("👉 ").strip()

def get_attachment_path():
    """Attachment file path लें"""
    print("📎 Enter attachment file path (or press Enter to skip):")
    
    if wake_detector.is_awake:
        speak_response("Please provide attachment file path or say skip")
        response = wake_detector.listen_for_command()
        if not response or 'skip' in response.lower() or 'no' in response.lower():
            return None
        return response.strip()
    else:
        path = input("👉 File path: ").strip()
        return path if path else None

def get_cc_emails():
    """CC emails लें"""
    cc_emails = []
    print("👥 Add CC emails (comma separated, or press Enter to skip):")
    
    if wake_detector.is_awake:
        speak_response("Add CC emails or say skip")
        response = wake_detector.listen_for_command()
        if response and 'skip' not in response.lower() and 'no' not in response.lower():
            emails = re.findall(r'[\w\.-]+@[\w\.-]+', response)
            cc_emails.extend(emails)
    else:
        response = input("👉 CC emails: ").strip()
        if response:
            emails = re.findall(r'[\w\.-]+@[\w\.-]+', response)
            cc_emails.extend(emails)
    
    return cc_emails

def get_bcc_emails():
    """BCC emails लें"""
    bcc_emails = []
    print("👤 Add BCC emails (comma separated, or press Enter to skip):")
    
    if wake_detector.is_awake:
        speak_response("Add BCC emails or say skip")
        response = wake_detector.listen_for_command()
        if response and 'skip' not in response.lower() and 'no' not in response.lower():
            emails = re.findall(r'[\w\.-]+@[\w\.-]+', response)
            bcc_emails.extend(emails)
    else:
        response = input("👉 BCC emails: ").strip()
        if response:
            emails = re.findall(r'[\w\.-]+@[\w\.-]+', response)
            bcc_emails.extend(emails)
    
    return bcc_emails

def add_attachment_to_email(msg, file_path):
    """Email में attachment add करें"""
    try:
        if not os.path.exists(file_path):
            return f"❌ File not found: {file_path}"
        
        with open(file_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read()) 
        
        encoders.encode_base64(part)
        filename = os.path.basename(file_path)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {filename}'
        )
        msg.attach(part)
        return f"✅ Attachment added: {filename}"
    except Exception as e:
        return f"❌ Attachment error: {str(e)}"

def generate_email_content(recipient, user_message, subject_hint=""):
    """AI से automatic email content generate करें"""
    prompt = f"""
User wants to send an email to: {recipient}
User's brief message: "{user_message}"
Subject hint: "{subject_hint}"

Please generate a complete, professional email with:
- Appropriate greeting
- Clear main content
- Professional closing
- Signature

Make it sound natural and contextual based on the recipient.
Keep it concise but complete.

Output only the email body text, no explanations.
"""
    
    try:
        email_body = get_llm_response(prompt, code_only=False).strip()
        return email_body
    except Exception as e:
        # Fallback simple email
        return f"Hello,\n\n{user_message}\n\nBest regards,\nLucifer Agent User"

def generate_email_subject(user_message, recipient=""):
    """AI से automatic subject generate करें"""
    prompt = f"""
Based on this email intent: "{user_message}"
Recipient: {recipient}

Generate a short, appropriate email subject line (max 5-6 words).
Output only the subject line, no quotes.
"""
    
    try:
        subject = get_llm_response(prompt, code_only=False).strip()
        return subject
    except Exception as e:
        return "Message from Lucifer Agent"

def send_interactive_email(recipient, user_message, subject_hint=""):
    """Interactive email system - step by step options"""
    try:
        print("🎯 Starting Interactive Email Setup...")
        
        # Step 1: Generate content using AI
        print("\n📧 Generating email content...")
        subject = generate_email_subject(user_message, recipient)
        email_body = generate_email_content(recipient, user_message, subject_hint)
        
        print(f"✅ Generated Subject: {subject}")
        print(f"✅ Generated Content:\n{email_body}")
        
        # Step 2: Ask for modifications
        modify_content = ask_yes_no_question("Do you want to modify the email content?")
        if modify_content:
            new_content = get_user_input("Enter your email content:")
            if new_content:
                email_body = new_content
        
        # Step 3: Ask for CC
        add_cc = ask_yes_no_question("Do you want to add CC recipients?")
        cc_emails = []
        if add_cc:
            cc_emails = get_cc_emails()
            if cc_emails:
                print(f"✅ CC emails: {', '.join(cc_emails)}")
        
        # Step 4: Ask for BCC
        add_bcc = ask_yes_no_question("Do you want to add BCC recipients?")
        bcc_emails = []
        if add_bcc:
            bcc_emails = get_bcc_emails()
            if bcc_emails:
                print(f"✅ BCC emails: {', '.join(bcc_emails)}")
        
        # Step 5: Ask for attachment
        add_attachment = ask_yes_no_question("Do you want to add an attachment?")
        attachment_path = None
        if add_attachment:
            attachment_path = get_attachment_path()
            if attachment_path:
                print(f"✅ Attachment: {attachment_path}")
        
        # Step 6: Final confirmation
        print("\n📋 Email Summary:")
        print(f"   To: {recipient}")
        print(f"   Subject: {subject}")
        print(f"   CC: {', '.join(cc_emails) if cc_emails else 'None'}")
        print(f"   BCC: {', '.join(bcc_emails) if bcc_emails else 'None'}")
        print(f"   Attachment: {attachment_path if attachment_path else 'None'}")
        print(f"   Content: {email_body[:100]}...")
        
        send_confirmation = ask_yes_no_question("Do you want to send this email?")
        if not send_confirmation:
            return "❌ Email cancelled by user"
        
        # Step 7: Send email
        print("\n🚀 Sending email...")
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient
        msg['Subject'] = subject
        
        if cc_emails:
            msg['Cc'] = ', '.join(cc_emails)
        
        # Add message body
        msg.attach(MIMEText(email_body, 'plain'))
        
        # Add attachment
        attachment_result = ""
        if attachment_path and os.path.exists(attachment_path):
            attachment_result = add_attachment_to_email(msg, attachment_path)
            print(attachment_result)
        
        # All recipients (To + CC + BCC)
        all_recipients = [recipient]
        if cc_emails:
            all_recipients.extend(cc_emails)
        if bcc_emails:
            all_recipients.extend(bcc_emails)
        
        # Send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg, to_addrs=all_recipients)
        server.quit()
        
        # Prepare result message
        result = f"✅ Email successfully sent to {recipient}"
        if cc_emails:
            result += f", CC: {len(cc_emails)} people"
        if bcc_emails:
            result += f", BCC: {len(bcc_emails)} people"
        if attachment_path:
            result += f", with attachment: {os.path.basename(attachment_path)}"
            
        return result
        
    except Exception as e:
        return f"❌ Email sending failed: {str(e)}"

# ========== PERMANENT MEMORY FUNCTIONS ==========

def update_chat_history(user_message=None, assistant_message=None):
    """
    Appends the user message or assistant response to the permanent
    JSON chat history file.
    """
    history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []

    if user_message:
        history.append({"role": "user", "content": user_message})
    if assistant_message:
        history.append({"role": "assistant", "content": assistant_message})

    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"❌ Error updating chat history: {e}")

def load_chat_history(last_n=50):
    """
    Loads the last_n messages from the chat history and formats
    them as a string for the LLM prompt.
    """
    if not os.path.exists(CHAT_HISTORY_FILE):
        return "No chat history found."

    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        # Get the last N messages
        recent_history = history[-last_n:]
        
        formatted_history = []
        for message in recent_history:
            role = "User" if message["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {message['content']}")
        
        return "\n".join(formatted_history)

    except Exception as e:
        print(f"❌ Error loading chat history: {e}")
        return "Error loading chat history."

def do_get_chat_history():
    """Returns the stored chat history for the user to see."""
    return load_chat_history(last_n=50)

# --- NEW: External Knowledge (Facts) Functions ---

def do_remember_fact(key, value):
    """
    Saves a specific key-value fact to the permanent facts file.
    """
    facts = {}
    if os.path.exists(FACTS_FILE):
        try:
            with open(FACTS_FILE, "r", encoding="utf-8") as f:
                facts = json.load(f)
        except json.JSONDecodeError:
            facts = {}
            
    facts[key.lower()] = value
    
    try:
        with open(FACTS_FILE, "w", encoding="utf-8") as f:
            json.dump(facts, f, indent=2)
        return f"✅ Got it. I'll remember that {key} is {value}."
    except Exception as e:
        return f"❌ Error remembering fact: {e}"

def do_get_fact(key):
    """
    Retrieves a specific fact from the permanent facts file.
    """
    if not os.path.exists(FACTS_FILE):
        return f"❌ I don't have any facts stored yet."
        
    try:
        with open(FACTS_FILE, "r", encoding="utf-8") as f:
            facts = json.load(f)
        
        value = facts.get(key.lower())
        if value:
            return f"✅ Your {key} is {value}."
        else:
            return f"❌ I don't have a fact stored for '{key}'."
    except Exception as e:
        return f"❌ Error getting fact: {e}"

# --- Wrapper functions for name ---
def do_remember_name(name):
    """Wrapper to remember the user's name using the fact system."""
    return do_remember_fact("name", name)

def do_get_name():
    """Wrapper to get the user's name from the fact system."""
    return do_get_fact("name")

# --- Short-term Context (Last Action) ---

def load_context():
    if os.path.exists(SESSION_CONTEXT_FILE):
        try:
            with open(SESSION_CONTEXT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_context(ctx):
    with open(SESSION_CONTEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2)

def update_context_from_action(action_dict):
    ctx = load_context()
    for k, v in action_dict.items():
        if k != "action" and isinstance(v, str) and v.strip():
            ctx[k.lower()] = v.strip()
    save_context(ctx)

def resolve_references_in_message(user_message):
    ctx = load_context()
    msg = user_message
    for k, v in ctx.items():
        patterns = [f"us {k}", f"that {k}", f"wahi {k}", f"wo {k}", f"the {k}"]
        for p in patterns:
            if p in msg.lower():
                msg = re.sub(re.escape(p), v, msg, flags=re.IGNORECASE)
    return msg

# ========== ULTIMATE WHATSAPP FIX ==========

def start_whatsapp_driver():
    """Start WhatsApp Web"""
    try:
        options = Options()
        
        if os.path.exists(CHROME_PATH):
            options.binary_location = CHROME_PATH
            print("✅ Using Chrome browser")
        elif os.path.exists(BRAVE_PATH):
            options.binary_location = BRAVE_PATH
            print("✅ Using Brave browser")
        else:
            return None, "❌ No browser found"
        
        options.add_argument(f"--user-data-dir={WHATSAPP_PROFILE_DIR}")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1200,800")
        
        service = Service(CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=options)
        
        return driver, None
        
    except Exception as e:
        return None, f"❌ Browser start failed: {str(e)}"

def wait_for_whatsapp_ready(driver, timeout=30):
    """Wait for WhatsApp to be ready"""
    print("⏳ Waiting for WhatsApp...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            if "web.whatsapp.com" not in driver.current_url:
                driver.get("https://web.whatsapp.com")
                time.sleep(3)
                continue
            
            # Check for any WhatsApp element
            indicators = [
                '//div[@data-testid="chat-list"]',
                '//div[@role="textbox"]',
                '//div[@title="Search input textbox"]',
                '//canvas[@aria-label="Scan me!"]'
            ]
            
            for indicator in indicators:
                try:
                    element = driver.find_element(By.XPATH, indicator)
                    if element.is_displayed():
                        print("✅ WhatsApp loaded")
                        return True
                except:
                    continue
                    
            time.sleep(2)
        except Exception as e:
            time.sleep(2)
    
    return False

def find_and_select_contact(driver, contact_name):
    """Find and select contact with proper clearing"""
    try:
        print(f"🔍 Finding: {contact_name}")
        
        # Clear any existing search first
        clear_selectors = [
            '//div[@title="Search input textbox"]',
            '//div[@contenteditable="true"][@data-tab="3"]'
        ]
        
        for selector in clear_selectors:
            try:
                search_box = driver.find_element(By.XPATH, selector)
                search_box.click()
                time.sleep(1)
                search_box.send_keys(Keys.CONTROL + "a")
                search_box.send_keys(Keys.DELETE)
                time.sleep(1)
                print("✅ Search cleared")
                break
            except:
                continue
        
        # Now search for contact
        search_box = None
        for selector in clear_selectors:
            try:
                search_box = driver.find_element(By.XPATH, selector)
                break
            except:
                continue
        
        if not search_box:
            return "❌ Search box not found"
        
        # Type contact name
        search_box.send_keys(contact_name)
        time.sleep(3)
        
        # Select contact
        contact_selectors = [
            f'//span[@title="{contact_name}"]',
            f'//span[contains(@title, "{contact_name}")]',
            '//div[@role="grid"]//div[@role="row"][1]'
        ]
        
        for selector in contact_selectors:
            try:
                contact = driver.find_element(By.XPATH, selector)
                contact.click()
                time.sleep(3)
                print(f"✅ Contact selected: {contact_name}")
                return None
            except:
                continue
        
        return f"❌ Contact '{contact_name}' not found"
        
    except Exception as e:
        return f"❌ Contact search failed: {str(e)}"

def send_message_ultimate(driver, message):
    """ULTIMATE MESSAGE SENDING - Multiple strategies"""
    try:
        print("💬 Sending message...")
        
        # Strategy 1: Try multiple input box selectors
        input_selectors = [
            '//div[@data-tab="10"]',
            '//div[@title="Type a message"]',
            '//footer//div[@contenteditable="true"]',
            '//div[@role="textbox"]',
            '//div[contains(@class, "copyable-text")]',
            '//div[@contenteditable="true"][@data-tab="1"]'
        ]
        
        input_box = None
        for selector in input_selectors:
            try:
                print(f"🔍 Trying selector: {selector}")
                input_box = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, selector))
                )
                print(f"✅ Found input box with: {selector}")
                break
            except:
                continue
        
        if not input_box:
            return "❌ No message input box found"
        
        # Strategy 2: Multiple click attempts
        for attempt in range(3):
            try:
                print(f"🖱️ Click attempt {attempt + 1}")
                input_box.click()
                time.sleep(1)
                
                # Clear any existing text
                input_box.send_keys(Keys.CONTROL + "a")
                input_box.send_keys(Keys.DELETE)
                time.sleep(0.5)
                
                # Type message character by character
                print(f"📝 Typing: {message}")
                for char in message:
                    input_box.send_keys(char)
                    time.sleep(0.05)
                
                time.sleep(1)
                
                # Strategy 3: Multiple send attempts
                for send_attempt in range(3):
                    try:
                        print(f"🚀 Send attempt {send_attempt + 1}")
                        input_box.send_keys(Keys.ENTER)
                        time.sleep(2)
                        
                        # Check if message was sent
                        if input_box.text == "":
                            print("✅ Message sent successfully!")
                            return None
                        else:
                            print("🔄 Message still in input, retrying...")
                            continue
                            
                    except Exception as e:
                        print(f"⚠️ Send attempt {send_attempt + 1} failed: {e}")
                        continue
                
                # If all send attempts failed, try JavaScript
                print("🔄 Trying JavaScript send...")
                driver.execute_script("arguments[0].value = '';", input_box)
                driver.execute_script(f"arguments[0].innerText = '{message}';", input_box)
                driver.execute_script("arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", input_box)
                input_box.send_keys(Keys.ENTER)
                time.sleep(2)
                
                print("✅ Message sent via JavaScript!")
                return None
                
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                time.sleep(1)
                continue
        
        return "❌ All message sending attempts failed"
            
    except Exception as e:
        return f"❌ Message sending failed: {str(e)}"

def send_whatsapp_message_ultimate(contact_name, message):
    """ULTIMATE WHATSAPP FUNCTION - 100% Working"""
    print(f"📱 WHATSAPP MISSION STARTED")
    print(f"🎯 Target: {contact_name}")
    print(f"💬 Payload: {message}")
    
    driver, error = start_whatsapp_driver()
    if error:
        return error
    
    try:
        # Step 1: Open WhatsApp
        print("🌐 STEP 1: Navigating to WhatsApp...")
        driver.get("https://web.whatsapp.com")
        
        # Step 2: Wait for load
        print("⏳ STEP 2: Waiting for WhatsApp...")
        if not wait_for_whatsapp_ready(driver, 40):
            return "❌ WhatsApp loading timeout"
        
        # Step 3: Check QR
        try:
            qr_code = driver.find_element(By.XPATH, '//canvas[@aria-label="Scan me!"]')
            if qr_code.is_displayed():
                return "📱 QR CODE: Please scan with your phone in browser window"
        except:
            print("✅ Already logged in")
        
        # Step 4: Find contact
        print("🔍 STEP 3: Finding contact...")
        error = find_and_select_contact(driver, contact_name)
        if error:
            return error
        
        # Step 5: Send message
        print("💬 STEP 4: Sending message...")
        error = send_message_ultimate(driver, message)
        if error:
            return error
        
        print("🎉 MISSION ACCOMPLISHED!")
        return f"✅ SUCCESS! Message sent to '{contact_name}'"
        
    except Exception as e:
        return f"❌ Mission failed: {str(e)}"
    finally:
        print("🖥️ Browser kept open for verification")
        # driver.quit()  # Uncomment to auto-close

def do_send_whatsapp(contact, message):
    """Execute WhatsApp sending"""
    if not contact or not message:
        return "❌ Please provide both contact name and message."
    
    print(f"🚀 LAUNCHING WHATSAPP...")
    start_time = time.time()
    
    result = send_whatsapp_message_ultimate(contact, message)
    
    end_time = time.time()
    print(f"⏱️ Mission duration: {end_time - start_time:.2f} seconds")
    
    return result

# ========== ALL OTHER FUNCTIONALITIES ==========

def resolve_folder_path(user_path):
    user_path = (user_path or "").strip().replace("/home/user", "~")
    if not user_path:
        return os.path.expanduser("~")
    if user_path.startswith("/"):
        user_path = "~" + user_path
    if not user_path.startswith("~"):
        user_path = os.path.join("~", user_path)
    full_path = os.path.expanduser(user_path)
    return full_path

def do_create_folder(folder_path, folder_name=None):
    try:
        folder_path = os.path.expanduser(folder_path)
        if folder_name:
            folder_path = os.path.join(folder_path, folder_name)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        return f"✅ Folder created at: {folder_path}"
    except Exception as e:
        return f"❌ Could not create folder: {e}"

def fuzzy_find_path(user_input, search_dirs_only=False, search_files_only=False):
    if not user_input:
        return None
    user_input_base = os.path.basename(user_input.strip().lower())
    home_dir = str(Path.home())
    all_matches = []

    for root, dirs, files in os.walk(home_dir, topdown=True):
        # Skip hidden directories and common cache/config dirs
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__pycache__'))]
        files = [f for f in files if not f.startswith('.')]
            
        items = dirs if search_dirs_only else files if search_files_only else dirs + files
        for name in items:
            score = difflib.SequenceMatcher(None, user_input_base, name.lower()).ratio()
            if score >= 0.7:
                full_path = os.path.join(root, name)
                all_matches.append((score, full_path))

    if not all_matches:
        return None

    all_matches.sort(reverse=True, key=lambda x: x[0])
    return all_matches[0][1]

def do_control_volume(amount: int):
    try:
        if not isinstance(amount, int):
            amount = int(str(amount).replace('%',''))
        if not amount:
            return "❌ Please specify a volume change amount (e.g., -10 or 10)."

        change = f"{abs(amount)}%"
        direction = "+" if amount > 0 else "-"
        
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{direction}{change}"], check=True, capture_output=True)

        direction_str = "increased" if amount > 0 else "decreased"
        return f"🔊 Volume {direction_str} by {abs(amount)}%."

    except FileNotFoundError:
        return "❌ 'pactl' command not found. This works only on Linux with PulseAudio/PipeWire."
    except Exception as e:
        return f"❌ Failed to change volume: {e}"

def do_list_dir_contents(path, type="all"):
    try:
        folder_path = normalize_path(path)

        if not os.path.exists(folder_path):
            return f"❌ Path not found: {folder_path}"
        if not os.path.isdir(folder_path):
            return f"❌ Not a directory: {folder_path}"

        items = os.listdir(folder_path)
        files = [f for f in items if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('.')]
        folders = [f for f in items if os.path.isdir(os.path.join(folder_path, f)) and not f.startswith('.')]

        if type == "files":
            return f"📄 Files in {folder_path}: {', '.join(files) if files else 'None'}"
        elif type == "folders":
            return f"📁 Folders in {folder_path}: {', '.join(folders) if folders else 'None'}"
        else:  # type == "all"
            return (
                f"Contents of {folder_path}:\n"
                f"📄 Files ({len(files)}): {', '.join(files) if files else 'None'}\n"
                f"📁 Folders ({len(folders)}): {', '.join(folders) if folders else 'None'}"
            )

    except Exception as e:
        return f"❌ Error while listing contents: {e}"

def normalize_path(path_str: str):
    path_str = (path_str or "").strip()
    if not path_str:
        return os.path.expanduser("~")
    lower_path = path_str.lower()
    if lower_path in ("desktop", "desk"):
        return os.path.expanduser("~/Desktop")
    if lower_path in ("documents", "docs"):
        return os.path.expanduser("~/Documents")
    if lower_path in ("downloads", "download"):
        return os.path.expanduser("~/Downloads")
    return os.path.expanduser(path_str)

def do_search_web(query):
    """
    ADVANCED V12 (FINAL):
    ✔ Smarter query classification (News / Event / Research / Generic Q&A)
    ✔ Stable DDGS engine with auto-sanitization
    ✔ Robust Google News fallback with topic-aware RSS selection
    ✔ Improved context extraction (duplicates removed, empty items skipped)
    ✔ Cleaner prompts for high-accuracy LLM answers
    ✔ Zero-crash architecture — ALL errors are gracefully handled
    """

    try:
        # Imports (ALL inside try to prevent global crash)
        import re
        import json
        import urllib.parse
        import requests
        from bs4 import BeautifulSoup
        from ddgs import DDGS

        # ================================================================
        # 0. UTILITIES
        # ================================================================
        def safe_get(val, default=""):
            return val.strip() if isinstance(val, str) else default

        def classify_query(q):
            """
            Intelligent query classifier.
            """
            ql = q.lower()

            news_terms = ["news", "latest", "update", "headlines", "breaking", "today"]
            event_terms = ["what happened", "incident", "case", "accident", "attack"]
            research_terms = ["study", "research", "paper", "report", "analysis"]
            tech_terms = ["tech", "technology", "ai", "machine learning", "startup"]

            if any(t in ql for t in news_terms):
                return "news"
            if any(t in ql for t in event_terms):
                return "news"
            if any(t in ql for t in research_terms):
                return "research"
            if any(t in ql for t in tech_terms):
                return "tech_news"
            return "qa"

        # ================================================================
        # 1. CLASSIFY QUERY
        # ================================================================
        qtype = classify_query(query)

        # ================================================================
        # 2. GOOGLE NEWS FALLBACK
        # ================================================================
        def google_news_fallback(topic):
            try:
                base = "https://news.google.com/rss/"
                tech = "headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en"

                if qtype == "tech_news":
                    url = base + tech
                else:
                    url = (
                        f"https://news.google.com/rss/search?q="
                        f"{urllib.parse.quote(topic)}"
                        f"&hl=en-US&gl=US&ceid=US:en"
                    )

                resp = requests.get(url, timeout=7, headers={
                    "User-Agent": "Mozilla/5.0"
                })

                soup = BeautifulSoup(resp.text, "xml")
                items = soup.find_all("item")[:5]

                if not items:
                    return None

                ctx = ""
                for i, item in enumerate(items):
                    title = item.title.text if item.title else ""
                    src = item.source.text if item.source else "Google News"
                    desc = BeautifulSoup(
                        item.description.text if item.description else "",
                        "html.parser"
                    ).get_text()

                    ctx += (
                        f"--- Result {i+1} ---\n"
                        f"Source: {src}\n"
                        f"Title: {safe_get(title)}\n"
                        f"Snippet: {safe_get(desc)}\n"
                    )
                return ctx

            except Exception:
                return None

        # ================================================================
        # 3. DDGS PRIMARY SEARCH
        # ================================================================
        def ddgs_search(q, qtype):
            try:
                with DDGS() as ddgs:
                    if qtype in ["news", "tech_news"]:
                        return ddgs.news(q, region="wt-wt", max_results=5)
                    else:
                        return ddgs.text(q, region="wt-wt", max_results=5)
            except:
                return None

        raw_results = ddgs_search(query, qtype)

        # ================================================================
        # 4. BUILD CONTEXT
        # ================================================================
        search_context = ""

        if raw_results:
            for i, r in enumerate(raw_results):
                title = safe_get(r.get("title", ""))
                if not title:
                    continue

                snippet = safe_get(
                    r.get("body") or r.get("excerpt") or "No summary available."
                )
                src = safe_get(r.get("source") or r.get("publisher") or "")

                search_context += (
                    f"--- Result {i+1} ---\n"
                    + (f"Source: {src}\n" if src else "")
                    + f"Title: {title}\n"
                    + f"Snippet: {snippet}\n"
                )

        # ================================================================
        # 5. FALLBACK IF DDGS EMPTY
        # ================================================================
        if not search_context.strip() and qtype in ["news", "tech_news"]:
            fallback = google_news_fallback(query)
            if fallback:
                search_context = fallback
            else:
                return f"❌ No information found for '{query}'."

        if not search_context.strip():
            return f"❌ No useful search results for '{query}'."

        # ================================================================
        # 6. PROMPT SELECTION
        # ================================================================
        if qtype in ["news", "tech_news"]:
            llm_prompt = f"""
Extract the top 3–5 REAL NEWS HEADLINES from the context. 
Write them as bullet points with 1–2 line summaries.

User Query: {query}

Context:
{search_context}

Answer:
"""
        else:
            llm_prompt = f"""
Answer the user's question using ONLY the information in the provided search context. 
Do NOT mention the context or the search.

User Query: {query}

Context:
{search_context}

Direct Answer:
"""

        # ================================================================
        # 7. LLM CALL
        # ================================================================
        try:
            llm_raw = get_llm_response(llm_prompt, code_only=False).strip()
        except Exception as e:
            return f"❌ LLM failed: {e}"

        return "🌐 " + llm_raw

    except Exception as e:
        return f"❌ Fatal error in do_search_web: {e}"

def do_get_weather(city_name=None):
    try:
        if not city_name:
            return "❌ Please specify a city to get weather info."
        url = f"https://wttr.in/{city_name}?format=3"
        resp = requests.get(url).text.strip()
        if "Unknown location" in resp:
            return f"❌ Weather info not available for '{city_name}'."
        return f"🌦️ {resp}"
    except Exception as e:
        return f"❌ Error fetching weather: {e}"

def do_trash_files(path_pattern):
    try:
        expanded_path = os.path.expanduser(path_pattern)
        items_to_trash = glob.glob(expanded_path)

        if not items_to_trash:
            if os.path.exists(expanded_path):
                items_to_trash = [expanded_path]
            else:
                fuzzy_path = fuzzy_find_path(path_pattern)
                if fuzzy_path and os.path.exists(fuzzy_path):
                    items_to_trash = [fuzzy_path]
                else:
                    return f"🤷 No files or directories found matching: {path_pattern}"

        for item in items_to_trash:
            print(f"🚮 Moving to trash: {item}")
            send2trash.send2trash(item)

        count = len(items_to_trash)
        preview = ", ".join(os.path.basename(p) for p in items_to_trash[:3])
        if count > 3:
            preview += "..."
        item_type = "item" if count == 1 else "items"
        return f"✅ Moved {count} {item_type} to the Trash (e.g., {preview})."

    except Exception as e:
        return f"❌ Error while trying to move files to Trash: {e}"

def do_fix_code(file_path):
    try:
        expanded_path = os.path.expanduser(file_path)
        resolved_path = fuzzy_find_path(expanded_path, search_files_only=True)
        if not resolved_path or not os.path.isfile(resolved_path):
            return f"❌ File not found: {file_path}"

        with open(resolved_path, "r", encoding="utf-8") as f:
            original_code = f.read()

        fix_prompt = f"""
You are a highly skilled programming assistant.
Analyze the given code, detect errors, and fix them.
Respond with ONLY a valid JSON object in the following format:
{{
  "error_type": "syntax" or "logic",
  "error_location": "<brief human-readable location of the issue>",
  "fixed_code": "<full corrected code without markdown or backticks>"
}}
Here is the code to fix:
{original_code}
"""
        llm_out = get_llm_response(fix_prompt, code_only=False).strip()
        match = re.search(r'\{[\s\S]*\}', llm_out)
        if not match:
            return f"❌ Could not parse AI output.\nRaw output:\n{llm_out}"

        try:
            data = json5.loads(match.group())
        except Exception as e:
            return f"❌ JSON parse error: {e}\nRaw output:\n{llm_out}"

        fixed_code = data.get("fixed_code", "").strip()
        if not fixed_code:
            return f"❌ AI did not return fixed code.\nRaw data:\n{data}"

        fixed_code = fixed_code.strip('```').strip()

        if len(fixed_code) < len(original_code) * 0.5:
            return "⚠️ Warning: AI output may be incomplete. Code not replaced."

        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(fixed_code)

        return (
            f"✅ Code fixed successfully!\n"
            f"🔹 Error Type: {data.get('error_type','unknown')}\n"
            f"📍 Location: {data.get('error_location','unknown')}\n"
            f"📂 File: {resolved_path}"
        )

    except Exception as e:
        return f"❌ Error fixing code: {e}"

def do_save_note(filename, content):
    try:
        notes_dir = os.path.expanduser("~/LuciferNotes")
        Path(notes_dir).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(notes_dir, filename if filename.endswith(".txt") else filename + ".txt")
        with open(filepath, "w") as f:
            f.write(content)
        # Try to open with kwrite, fallback to xdg-open
        try:
            subprocess.Popen(['kwrite', filepath])
        except FileNotFoundError:
            subprocess.Popen(['xdg-open', filepath])
        return f"📝 Note saved to: {filepath} (opened in editor)"
    except Exception as e:
        return f"❌ Could not save note: {e}"

def do_remind_me(message, after_minutes):
    try:
        after_minutes = int(after_minutes)
        def notify():
            time.sleep(after_minutes * 60)
            subprocess.run(['notify-send', 'Lucifer Reminder', message], check=True)
            try:
                subprocess.run(['espeak', f'Reminder: {message}'], check=True)
            except FileNotFoundError:
                pass # espeak not installed
        threading.Thread(target=notify, daemon=True).start()
        return f"⏰ Reminder set for {after_minutes} minutes from now."
    except Exception as e:
        return f"❌ Could not set reminder: {e}"

def do_get_network_info():
    try:
        addrs = psutil.net_if_addrs()
        info = "🌐 Network Information:\n"
        has_info = False
        for interface, addresses in addrs.items():
            if interface == 'lo':
                continue
            ipv4 = next((addr.address for addr in addresses if addr.family == psutil.AF_INET), "N/A")
            ipv6 = next((addr.address for addr in addresses if addr.family == psutil.AF_INET6), "N/A")
            if ipv4 != "N/A" or ipv6 != "N/A":
                info += f"- {interface}:\n  - IPv4: {ipv4}\n  - IPv6: {ipv6}\n"
                has_info = True
        if not has_info:
            return "❌ No active network interfaces with IP addresses were found."
        return info
    except Exception as e:
        return f"❌ Error getting network info: {e}"

def do_create_project(project_name, location, language, gui):
    try:
        location = location.replace("/home/user", "~")
        base = os.path.expanduser(os.path.join(location, project_name))
        Path(base).mkdir(parents=True, exist_ok=True)
        skeleton_code = ""
        ext = ""
        language_lower = (language or "").lower()
        gui = gui if isinstance(gui, bool) else False

        if language_lower in ["cpp", "c++"]:
            ext = "cpp"
            skeleton_code = '''#include <iostream>\nint main() {\n    std::cout << "Hello, C++ Project!" << std::endl;\n    return 0;\n}'''
        elif language_lower == "c":
            ext = "c"
            skeleton_code = '''#include <stdio.h>\nint main() {\n    printf("Hello, C Project!\\n");\n    return 0;\n}'''
        elif language_lower == "python":
            ext = "py"
            skeleton_code = 'print("Hello, Python Project!")'
        elif language_lower == "java":
            ext = "java"
            skeleton_code = f'public class {project_name} {{\n    public static void main(String[] args) {{\n        System.out.println("Hello, Java Project!");\n    }}\n}}'
        else:
            return f"✅ Created folder '{project_name}' at {base} (no code skeleton for '{language}')"

        filename = f"main.{ext}" if language_lower != "java" else f"{project_name}.{ext}"
        filepath = os.path.join(base, filename)
        with open(filepath, 'w') as f:
            f.write(skeleton_code)
        return f"✅ Project '{project_name}' created with {language} skeleton at {base}"
    except Exception as e:
        return f"❌ Project creation error: {e}"

def do_create_file(folder_path, filename, content=None):
    try:
        folder_path = os.path.expanduser(folder_path or "~")
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "w") as f:
            if content is not None:
                f.write(content)
        return f"✅ File '{filename}' created/overwritten at: {file_path}"
    except Exception as e:
        return f"❌ Could not create or write file: {e}"
    
def do_file_exists(filename, type="any"):
    resolved_path = fuzzy_find_path(filename)
    if resolved_path and os.path.exists(resolved_path):
        item_type = "folder" if os.path.isdir(resolved_path) else "file"
        if type != "any" and type != item_type:
            return f"❌ '{filename}' exists but is not a {type}."
        return f"✅ Yes, {item_type} found: {resolved_path}"
    return f"❌ No, '{filename}' does not exist in your home directory."

def do_get_system_usage():
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        ram_usage = ram.percent
        return f"💻 CPU Usage: {cpu_usage}%\n🧠 RAM Usage: {ram_usage}% ({ram.used/1024**3:.2f}GB / {ram.total/1024**3:.2f}GB)"
    except Exception as e:
        return f"❌ Error getting system usage: {e}"

def do_change_wallpaper(image_path):
    try:
        expanded_path = os.path.expanduser(image_path or "")
        if not expanded_path or not os.path.exists(expanded_path):
            return f"❌ Image file not found at: {expanded_path if expanded_path else '(no path provided)'}"
        mime, _ = mimetypes.guess_type(expanded_path)
        if not (mime and mime.startswith("image/")):
            return f"❌ Not an image: {expanded_path}"

        jscript = f"""
var allDesktops = desktops();
for (var i=0; i<allDesktops.length; i++) {{
    var d = allDesktops[i];
    d.wallpaperPlugin = "org.kde.image";
    d.currentConfigGroup = Array("Wallpaper","org.kde.image","General");
    d.writeConfig("Image", "file://{expanded_path}");
}}
"""
        try:
            subprocess.run(
                ["qdbus-qt5", "org.kde.plasmashell", "/PlasmaShell", "org.kde.PlasmaShell.evaluateScript", jscript],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return f"✅ Wallpaper changed to: {expanded_path}"
        except FileNotFoundError:
            try:
                subprocess.run(
                    ["qdbus", "org.kde.plasmashell", "/PlasmaShell", "org.kde.PlasmaShell.evaluateScript", jscript],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                return f"✅ Wallpaper changed to: {expanded_path}"
            except Exception as e:
                return f"❌ Failed to set wallpaper via qdbus: {e}"
    except Exception as e:
        return f"❌ Error changing wallpaper: {e}"

def do_rename_file(filepath, newname):
    try:
        base_dir = os.path.expanduser(os.path.dirname(filepath))
        old_path = os.path.expanduser(filepath)
        
        if not os.path.exists(old_path):
             # Try fuzzy finding if exact path fails
            old_path = fuzzy_find_path(filepath)
            if not old_path:
                return f"❌ File or folder not found: '{filepath}'"
            base_dir = os.path.dirname(old_path)

        new_path = os.path.join(base_dir, newname)

        os.rename(old_path, new_path)
        item_type = "folder" if os.path.isdir(new_path) else "file"
        return f"✅ Successfully renamed {item_type} to '{newname}' in {base_dir}"
    except Exception as e:
        return f"❌ Rename failed: {e}"

def do_wifi_status():
    try:
        ssid = subprocess.check_output("iwgetid -r", shell=True).decode().strip()
        return f"📶 Connected to WiFi: {ssid}" if ssid else "❌ Not connected to any WiFi."
    except Exception:
        return "❌ Could not fetch WiFi details."

def do_open_file(filename):
    try:
        resolved_path = fuzzy_find_path(filename)
        if not resolved_path or not os.path.isfile(resolved_path):
            return f"❌ File '{filename}' not found."
        subprocess.Popen(['xdg-open', resolved_path])
        return f"✅ File opened: {resolved_path}"
    except Exception as e:
        return f"❌ Error opening file: {e}"

def do_play_music(song):
    import urllib.parse
    # This URL is clean and correct
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(song)}"
    try:
        html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
        ids = re.findall(r"watch\?v=(\w{11})", html)
        
        if not ids or not isinstance(ids[0], str):
            return f"❌ No YouTube video found for '{song}'."
            
        video_url = f"https://www.youtube.com/watch?v={ids[0]}"
        webbrowser.open(video_url)
        return f"✅ Playing '{song}' on YouTube."
    except Exception as e:
        return f"❌ Failed to play song: {e}"

def do_stop_music():
    try:
        # Try playerctl first
        subprocess.run(['playerctl', 'stop'], check=True, capture_output=True)
        return "✅ Music stopped (via playerctl)."
    except Exception:
        # Fallback to killing browser
        try:
            os.system("pkill -f 'brave|firefox|chrome|chromium'")
            return "✅ Stopped browser music (browser killed)."
        except Exception as e:
            return f"❌ Error stopping music: {e}"

def do_control_media(cmd):
    try:
        subprocess.run(['playerctl', cmd], check=True, capture_output=True)
        return f"Media {cmd} executed."
    except Exception as e:
        return f"❌ Failed: {e}"

def do_change_brightness(amount: int):
    try:
        if not isinstance(amount, int):
            amount = int(str(amount).replace('%',''))
            
        current = int(subprocess.check_output(['brightnessctl', 'g']).decode().strip())
        max_val = int(subprocess.check_output(['brightnessctl', 'm']).decode().strip())
        # Calculate change based on percentage of max
        change = int((amount / 100) * max_val)
        new = max(1, min(max_val, current + change))
        
        subprocess.run(['brightnessctl', 's', str(new)], check=True)
        return f"✅ Brightness set to {int((new/max_val)*100)}%."
    except Exception as e:
        return f"❌ Failed to change brightness: {e}"

def do_delete_file(filepath):
    try:
        # 1. Resolve path with fuzzy matching
        resolved_path = fuzzy_find_path(filepath)
        
        # 2. Check if path exists at all
        if not resolved_path or not os.path.exists(resolved_path):
            return f"❌ File or Folder not found: {filepath}"
            
        # 3. If it is a FOLDER
        if os.path.isdir(resolved_path):
            # We use send2trash for folders because recursive delete is dangerous
            try:
                send2trash.send2trash(resolved_path)
                return f"✅ Moved folder to Trash: {resolved_path}"
            except Exception as e:
                return f"❌ Could not trash folder (permission error?): {e}"

        # 4. If it is a FILE
        os.remove(resolved_path)
        return f"✅ Successfully deleted file: {resolved_path}"

    except Exception as e:
        return f"❌ Error deleting item: {e}"

def do_open_browser():
    webbrowser.open("https://google.com")
    return "✅ Browser opened."

def do_setup_python_environment(project_path, packages=None):
    """
    Creates a full Python environment with a venv, requirements.txt, 
    and a setup script for the user to run.
    """
    if packages is None:
        packages = []

    try:
        # 1. Resolve path and ensure it exists
        full_path = os.path.expanduser(project_path)
        if not os.path.isdir(full_path):
            Path(full_path).mkdir(parents=True, exist_ok=True)

        # 2. Handle 'tkinter' (it's special on Linux)
        req_packages = []
        tkinter_note = ""
        for pkg in packages:
            if pkg.lower() == "tkinter":
                tkinter_note = """
# NOTE: 'tkinter' was requested. 
# On many Linux systems, it must be installed via your system package manager.
# If you get a 'No module named _tkinter' error, please run one of these:
#
#   sudo apt-get install python3-tk  (for Debian/Ubuntu)
#   sudo pacman -S tk                (for Arch/Manjaro)
#   sudo dnf install python3-tkinter (for Fedora)
#
"""
            else:
                req_packages.append(pkg)

        # 3. Create requirements.txt
        req_path = os.path.join(full_path, "requirements.txt")
        with open(req_path, "w", encoding="utf-8") as f:
            if req_packages:
                f.write("\n".join(req_packages) + "\n")
            else:
                f.write("# No additional Python packages required.\n")

        # 4. Create the setup.sh script
        setup_script_path = os.path.join(full_path, "setup.sh")
        script_content = f"""#!/bin/bash
# This script will set up your Python virtual environment.
{tkinter_note}
echo "Creating virtual environment in 'venv'..."
python -m venv venv

echo "Activating environment..."
source venv/bin/activate

echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo "---"
echo "✅ Setup complete!"
echo "You can now run your script from this folder with:"
echo "source venv/bin/activate"
echo "python {os.path.basename(full_path)}.py"
echo "---"
"""
        with open(setup_script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        # 5. Make the script executable
        os.chmod(setup_script_path, 0o755)

        return f"✅ Created environment at {full_path}.\nRun 'cd {full_path} && ./setup.sh' to install."

    except Exception as e:
        return f"❌ Failed to create Python environment: {e}"
    
def do_navigate_to(url):
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    webbrowser.open(url)
    return f"✅ Navigating to {url}"

def do_search_website(query):
    # FIX: Cleaned URL
    url = f"https://www.google.com/search?q={query.replace(' ','+')}"
    webbrowser.open(url)
    return f"✅ Searching: {query}"

def start_driver():
    try:
        options = Options()
        options.binary_location = BRAVE_BINARY
        # Use a consistent profile path
        profile_dir = os.path.expanduser("~/.config/BraveSoftware/Brave-Browser/Default")
        options.add_argument(f"--user-data-dir={profile_dir}")
        # options.add_argument("--profile-directory=Default") # Often redundant with user-data-dir
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--start-maximized")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        
        driver_path = shutil.which("chromedriver") or CHROMEDRIVER_PATH
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=options)
        
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', { get: () => undefined })"
        })
        return driver
    except Exception as e:
        print(f"❌ Failed to start Selenium driver: {e}")
        print("Please ensure 'chromedriver' is in your PATH and 'brave' is at /usr/bin/brave")
        return None

def brute_force_find_element(driver, selectors, clickable=False, timeout=5):
    for xpath in selectors:
        try:
            if clickable:
                return WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.XPATH, xpath)))
            else:
                return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))
        except Exception:
            continue
    return None

def send_whatsapp_message(contact_name, message):
    driver = start_driver()
    if not driver:
        return "❌ Could not start browser. Check paths and permissions."
        
    wait = WebDriverWait(driver, 30)
    driver.get("https://web.whatsapp.com")

    try:
        search_box_selectors = [
            '//div[@contenteditable="true" and @data-testid="chat-list-search"]',
            '//div[@role="search"]//div[@contenteditable="true"]'
        ]
        search = brute_force_find_element(driver, search_box_selectors, clickable=True, timeout=60) # Long timeout for login
        if not search:
            raise Exception("Could not find the search box. (Timeout or login required?)")

        search.click()
        time.sleep(0.5)
        search.send_keys(contact_name)
        time.sleep(2)  # Let results load

        row_selectors = [
            f'//div[@role="gridcell"]//span[@title="{contact_name}"]',
            f'//span[contains(@title, "{contact_name}")]'
        ]
        contact_element = brute_force_find_element(driver, row_selectors, clickable=True, timeout=10)
        
        if not contact_element:
             raise Exception(f"Could not find contact '{contact_name}' in chat list.")
        
        contact_element.click()
        time.sleep(1)

        input_selectors = [
            '//footer//div[@contenteditable="true" and @data-testid="conversation-compose-box-input"]',
            '//div[@contenteditable="true" and @data-tab="10"]'
        ]
        input_box = brute_force_find_element(driver, input_selectors, clickable=True, timeout=8)
        if not input_box:
            raise Exception("Could not locate the message input box.")

        input_box.click()
        time.sleep(0.5)
        input_box.send_keys(message)
        time.sleep(0.2)
        input_box.send_keys(Keys.ENTER)
        time.sleep(1)

        return f"✅ Message sent to '{contact_name}'."

    except Exception as e:
        return f"❌ Failed to send message: {e}"
    finally:
        try:
            if driver:
                driver.quit()
        except:
            pass

def do_process_document(file_path, query=None):
    try:
        expanded_path = os.path.expanduser(file_path)
        resolved_path = fuzzy_find_path(expanded_path, search_files_only=True)
        if not resolved_path or not os.path.exists(resolved_path):
            return f"❌ File not found: {file_path}"

        ext = os.path.splitext(resolved_path)[1].lower()
        extracted_text = ""

        if ext == ".pdf":
            with fitz.open(resolved_path) as doc:
                for page in doc:
                    extracted_text += page.get_text()
        elif ext == ".docx":
            doc = docx.Document(resolved_path)
            extracted_text = "\n".join(p.text for p in doc.paragraphs)
        elif ext in [".jpg", ".jpeg", ".png"]:
            img = Image.open(resolved_path)
            extracted_text = pytesseract.image_to_string(img)
        else:
            return f"❌ Unsupported file type: {ext}"

        if not extracted_text.strip():
            return "❌ No text could be extracted from this file."
        
        extracted_text = extracted_text.strip()[:8000] # Limit context size

        if not query:
            prompt = f"Summarize the following document in bullet points:\n\n{extracted_text}"
        else:
            prompt = f"Based *only* on the following document, answer the user's query.\n\nDOCUMENT:\n{extracted_text}\n\nQUERY:\n{query}"

        response_json = get_llm_response(prompt, code_only=False)
        # Extract chat message from LLM's JSON response
        try:
            return json.loads(response_json).get("message", response_json)
        except Exception:
            return response_json # Fallback if response wasn't JSON

    except Exception as e:
        return f"❌ Error processing document: {e}"

def do_send_email(recipient, message):
    try:
        subject_prompt = f"Generate a short professional subject line for this email:\n\n{message}"
        subject_response = get_llm_response(subject_prompt, code_only=False)
        subject = json.loads(subject_response).get("message", "Message from Agent")
        
        print(f"📌 Subject generated: {subject}")

        attachments = []
        attach_input = input("📎 Any attachments? (Enter file paths separated by commas or 'no'): ").strip()
        if attach_input.lower() != "no" and attach_input:
            for path in attach_input.split(","):
                p = os.path.expanduser(path.strip())
                if os.path.isfile(p):
                    attachments.append(p)
                else:
                    print(f"❌ Not found: {p}")
        
        cc_list = []
        cc_input = input("👥 CC emails? (comma separated or 'no'): ").strip()
        if cc_input.lower() != "no" and cc_input:
            cc_list = [c.strip() for c in cc_input.split(",")]

        enhance_prompt = f"Rewrite the following email professionally:\n\n{message}"
        enhanced_response = get_llm_response(enhance_prompt, code_only=False)
        enhanced_message = json.loads(enhanced_response).get("message", message)
        print("\n✍️ Enhanced draft:\n", enhanced_message)

        confirm = input("\n✅ Send this email? (yes/no): ").strip().lower()
        if confirm != "yes":
            return "❌ Email sending cancelled."

        return send_email_with_attachments(recipient, cc_list, subject, enhanced_message, attachments)
    except Exception as e:
        return f"❌ Error preparing email: {e}"

def send_email_with_attachments(to_email, cc_emails, subject, body, attachments=None):
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    creds = None
    try:
        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Assumes credentials.json is in the same directory
                if not os.path.exists("credentials.json"):
                    return "❌ 'credentials.json' not found. Please download from Google Cloud Console."
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            with open("token.pickle", "wb") as token:
                pickle.dump(creds, token)

        service = build("gmail", "v1", credentials=creds)
        msg = EmailMessage()
        msg['To'] = to_email
        if cc_emails:
            msg['Cc'] = ", ".join(cc_emails)
        msg['Subject'] = subject
        msg.set_content(body)

        for file_path in attachments or []:
            mime_type, _ = mimetypes.guess_type(file_path)
            maintype, subtype = "application", "octet-stream"
            if mime_type:
                try:
                    maintype, subtype = mime_type.split("/", 1)
                except ValueError:
                    pass # Keep default
            with open(file_path, "rb") as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype,
                                   filename=os.path.basename(file_path))

        raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw_message}).execute()
        return f"✅ Email sent to {to_email}."
    except Exception as e:
        return f"❌ Failed to send email: {e}"

def do_tell_time():
    return "⏰ " + datetime.now().strftime("%I:%M %p") # 12-hour format

def do_move_file_folder(source_path, destination_path):
    try:
        src = os.path.expanduser(source_path)
        dst = os.path.expanduser(destination_path)
        if not os.path.exists(src):
            src = fuzzy_find_path(source_path)
            if not src:
                return f"❌ Source '{source_path}' does not exist."

        if not os.path.exists(dst):
            Path(dst).mkdir(parents=True, exist_ok=True)

        final_dst = dst
        if os.path.isdir(dst):
            final_dst = os.path.join(dst, os.path.basename(src))

        shutil.move(src, final_dst)
        return f"✅ Moved '{src}' to '{final_dst}'."
    except Exception as e:
        return f"❌ Failed to move: {e}"

def do_tell_date():
    return "📆 " + datetime.now().strftime("%A, %d %B %Y")

def do_search_pdfs(topic, max_results=5):
    """
    Search for PDFs on a specific topic and return top results with download links.
    """
    try:
        import urllib.parse
        from ddgs import DDGS
        
        # Search query optimized for PDFs
        search_query = f"{topic} filetype:pdf"
        
        print(f"[PDF Search] Searching for: {search_query}")
        
        results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(search_query, region="wt-wt", max_results=max_results * 2))
        
        # Filter for actual PDF links
        for i, result in enumerate(search_results):
            url = result.get("href") or result.get("url", "")
            title = result.get("title", "Untitled")
            snippet = result.get("body", "No description")
            
            if url.lower().endswith('.pdf') or 'pdf' in url.lower():
                results.append({
                    "number": len(results) + 1,
                    "title": title,
                    "url": url,
                    "snippet": snippet[:150]
                })
                
                if len(results) >= max_results:
                    break
        
        if not results:
            return "❌ No PDFs found for this topic."
        
        # Format results for display
        output = f"📄 Found {len(results)} PDFs about '{topic}':\n\n"
        for pdf in results:
            output += f"{pdf['number']}. **{pdf['title']}**\n"
            output += f"   URL: {pdf['url']}\n"
            output += f"   Preview: {pdf['snippet']}...\n\n"
        
        # Store results in context for later use
        ctx = load_context()
        ctx["last_pdf_search"] = results
        save_context(ctx)
        
        return output
        
    except Exception as e:
        return f"❌ PDF search failed: {e}"

def do_download_pdf(selection):
    """
    Download a PDF from the last search results.
    Selection can be a number (1-5) or "all"
    """
    try:
        import requests
        from pathlib import Path
        
        # Load last search results from context
        ctx = load_context()
        pdf_results = ctx.get("last_pdf_search", [])
        
        if not pdf_results:
            return "❌ No PDF search results found. Please search for PDFs first."
        
        # Handle selection
        if isinstance(selection, str) and selection.lower() == "all":
            selections = list(range(1, len(pdf_results) + 1))
        else:
            try:
                selections = [int(selection)]
            except (ValueError, TypeError):
                return f"❌ Invalid selection: {selection}. Use a number (1-{len(pdf_results)}) or 'all'."
        
        # Create downloads folder
        download_dir = os.path.expanduser("~/Downloads/PDFs")
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        
        downloaded_files = []
        
        for sel in selections:
            if sel < 1 or sel > len(pdf_results):
                continue
            
            pdf = pdf_results[sel - 1]
            url = pdf["url"]
            title = pdf["title"]
            
            # Clean filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title[:100]  # Limit length
            filename = f"{safe_title}.pdf"
            filepath = os.path.join(download_dir, filename)
            
            print(f"[Download] Downloading: {title}")
            
            # Download the PDF
            response = requests.get(url, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                downloaded_files.append(filepath)
                print(f"[Download] ✅ Saved: {filepath}")
            else:
                print(f"[Download] ❌ Failed: HTTP {response.status_code}")
        
        if not downloaded_files:
            return "❌ No PDFs were downloaded successfully."
        
        # Store downloaded files in context
        ctx["last_downloaded_pdfs"] = downloaded_files
        save_context(ctx)
        
        output = f"✅ Downloaded {len(downloaded_files)} PDF(s) to {download_dir}:\n"
        for fpath in downloaded_files:
            output += f"- {os.path.basename(fpath)}\n"
        
        return output
        
    except Exception as e:
        return f"❌ PDF download failed: {e}"

def do_extract_and_summarize_pdf(file_path=None):
    """
    Extract text from a PDF and generate an intelligent summary.
    If file_path is None, uses the last downloaded PDF from context.
    """
    try:
        import fitz  # PyMuPDF
        
        # If no file path provided, use last downloaded
        if not file_path:
            ctx = load_context()
            downloaded_pdfs = ctx.get("last_downloaded_pdfs", [])
            if not downloaded_pdfs:
                return "❌ No PDF found. Please download a PDF first."
            file_path = downloaded_pdfs[0]  # Use first/most recent
        
        # Expand path
        file_path = os.path.expanduser(file_path)
        
        if not os.path.exists(file_path):
            return f"❌ PDF not found: {file_path}"
        
        print(f"[Extract] Processing: {file_path}")
        
        # Extract text from PDF
        extracted_text = ""
        with fitz.open(file_path) as doc:
            total_pages = len(doc)
            print(f"[Extract] Total pages: {total_pages}")
            
            # Extract from all pages (or limit if too large)
            max_pages = min(total_pages, 50)  # Limit to first 50 pages
            for page_num in range(max_pages):
                page = doc[page_num]
                extracted_text += page.get_text()
        
        if not extracted_text.strip():
            return "❌ No text could be extracted from this PDF."
        
        # Truncate if too long (keep within LLM context limits)
        char_limit = 15000
        if len(extracted_text) > char_limit:
            extracted_text = extracted_text[:char_limit] + "\n\n[... Document truncated for processing ...]"
        
        print(f"[Extract] Extracted {len(extracted_text)} characters")
        
        # Generate intelligent summary using LLM
        summary_prompt = f"""You are an expert document analyst. 

Analyze the following PDF content and provide a comprehensive summary.

**Instructions:**
1. Create a structured summary with clear sections
2. Identify key topics, main arguments, and important findings
3. Extract critical data points, statistics, or facts
4. Note any actionable insights or recommendations
5. Keep the summary concise but informative (300-500 words)

**PDF Content:**
{extracted_text}

**Generate a professional summary in markdown format:**
"""
        
        print("[LLM] Generating summary...")
        llm_response = get_llm_response(summary_prompt, code_only=False).strip()
        
        # Parse LLM response
        try:
            if llm_response.startswith('{'):
                llm_data = json.loads(llm_response)
                summary = llm_data.get("message", llm_response)
            else:
                summary = llm_response
        except:
            summary = llm_response
        
        # Store summary in context
        ctx = load_context()
        ctx["last_pdf_summary"] = summary
        ctx["last_pdf_path"] = file_path
        save_context(ctx)
        
        return f"📄 **PDF Summary: {os.path.basename(file_path)}**\n\n{summary}"
        
    except Exception as e:
        return f"❌ PDF extraction/summary failed: {e}"

def do_save_pdf_summary_as_note():
    """
    Save the last PDF summary as a note file.
    """
    try:
        ctx = load_context()
        summary = ctx.get("last_pdf_summary")
        pdf_path = ctx.get("last_pdf_path")
        
        if not summary:
            return "❌ No PDF summary found. Please extract and summarize a PDF first."
        
        # Generate note filename based on PDF name
        if pdf_path:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            note_filename = f"Summary_{pdf_name}.txt"
        else:
            note_filename = f"PDF_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Save using existing save_note function
        return do_save_note(note_filename, summary)
        
    except Exception as e:
        return f"❌ Failed to save summary as note: {e}"

def do_announce(message):
    try:
        subprocess.run(['espeak', message], check=True, capture_output=True)
    except FileNotFoundError:
        pass # espeak not installed
    except Exception:
        pass # Other error
    return f"[Speaking]: {message}"

def do_system_info():
    try:
        uname = platform.uname()
        ram_gb = round(psutil.virtual_memory().total / (1024.**3), 2)
        return f"{uname.system} {uname.release} on {uname.node}. CPU: {uname.processor}. RAM: {ram_gb} GB"
    except Exception as e:
        return f"❌ Error getting system info: {e}"

def do_battery_status():
    try:
        b = psutil.sensors_battery()
        if not b:
            return "🔋 Battery status not available on this system."
        return f"🔋 {b.percent}% {'charging' if b.power_plugged else 'discharging'}"
    except Exception:
        return "🔋 Battery status not available."

def do_bluetooth_devices():
    try:
        output = subprocess.check_output("bluetoothctl paired-devices", shell=True).decode().strip()
        return f"🔵 Paired Bluetooth Devices:\n{output}" if output else "🔵 No Bluetooth devices paired."
    except Exception:
        return "❌ Could not get Bluetooth devices (is 'bluetoothctl' installed?)."

def do_connected_devices():
    try:
        out = subprocess.check_output("lsusb", shell=True).decode()
        return f"🖱️ Connected USB Devices:\n{out}"
    except Exception:
        return "❌ Failed to fetch USB devices (is 'lsusb' installed?)."

def do_general_knowledge(question):
    # This just uses the web search
    return do_search_web(question)

def do_chat(resp):
    return resp

def do_extract_pdf_text(query):
    try:
        url = subprocess.check_output([
            "qdbus", "org.kde.okular", "/okular", "org.kde.okular.getDocumentUrl"
        ]).decode().strip()
        if not url.startswith("file://"):
            return "❌ No PDF open in Okular."
        filepath = url.replace("file://", "")
        pdf_text = subprocess.check_output(["pdftotext", filepath, "-"]).decode()
        
        pdf_text = pdf_text.strip()[:8000] # Limit context
        
        if not query:
            return "📄 PDF content (first 8000 chars):\n" + pdf_text
        else:
            prompt = f"Based *only* on the following PDF content, answer the user's query.\n\nDOCUMENT:\n{pdf_text}\n\nQUERY:\n{query}"
            summary_response = get_llm_response(prompt, code_only=False)
            return json.loads(summary_response).get("message", summary_response)
    except Exception as e:
        return f"❌ PDF extraction error: {e}"

# ========== LLM INTENT EXTRACTION ==========

def extract_llm_intent(user_message):
    
    # Load the recent chat history to provide context
    chat_history_context = load_chat_history(last_n=50)
    
    # Load external facts to provide context
    facts_context = "{}"
    if os.path.exists(FACTS_FILE):
        try:
            with open(FACTS_FILE, "r", encoding="utf-8") as f:
                facts_context = f.read()
        except Exception:
            pass # Ignore if facts can't be read

    # First check if it's a meeting command using manual parsing
    meeting_keywords = ['schedule', 'meeting', 'google meet', 'calendar', 'meet']
    if any(keyword in user_message.lower() for keyword in meeting_keywords):
        print("🎯 Detected meeting command, using manual parser...")
        meeting_action = parse_meeting_command(user_message)
        return [meeting_action]

    system_prompt = f"""
You are an intelligent Linux desktop voice/text agent.

--- PERMANENT FACTS (EXTERNAL KNOWLEDGE) ---
{facts_context}
--- END OF FACTS ---

--- RECENT CONVERSATION HISTORY ---
{chat_history_context}
--- END OF HISTORY ---

Always use this conversation history and facts for context.
Recognize user intent and output *only* JSON in this schema (no extra explanation):
Do NOT include <think> or any explanation. Output must be only a valid JSON object or array of them.

{{"action":"...", ...}}

Or an array of multiple actions:
[
  {{"action":"...", ...}},
  {{"action":"...", ...}}
]

--- ACTIONS ---

If user says "my name is X" or "remember my name is X":
→ {{"action": "remember_fact", "key": "name", "value": "X"}}

If user says "remember my [KEY] is [VALUE]" (e.g., "remember my favorite color is blue"):
→ {{"action": "remember_fact", "key": "favorite color", "value": "blue"}}
If user says "run a hallucination test":
→ {{"action": "run_lab_hallucination", "query": "<user query>"}}

If user asks "what is my name" or "tell me my name":
→ {{"action": "get_fact", "key": "name"}}

If user asks "what is my [KEY]":
→ {{"action": "get_fact", "key": "[KEY]"}}

### GOOGLE MEET SCHEDULING ###
If user wants to schedule a meeting:
→ {{"action": "schedule_meet", "title": "Meeting Title", "date": "2024-12-31", "start_time": "14:00", "end_time": "15:00", "attendees": ["email1@example.com", "email2@example.com"]}}

### INTERACTIVE EMAIL SYSTEM ###
If user wants to send an email:
→ {{"action": "send_email", "recipient": "email@example.com", "message": "email content", "subject": "optional subject"}}

### AUTONOMOUS PDF WORKFLOW ###

When user requests a PDF workflow (e.g., "Find a PDF about cybersecurity, download it, summarize it, and save as note"):

1. search_pdfs: {{"action": "search_pdfs", "topic": "cybersecurity", "max_results": 5}}
   - Returns numbered list of PDFs (1-5)

2. User will respond with a number (e.g., "2")
   - You respond: {{"action": "download_pdf", "selection": 2}}

3. After download completes:
   {{"action": "extract_and_summarize_pdf"}}

4. After summary is generated:
   {{"action": "save_pdf_summary_as_note"}}

Example full workflow:
User: "Find me a PDF about machine learning, download it, extract text, summarize it and save as a note"
→ First response: {{"action": "search_pdfs", "topic": "machine learning"}}
→ (User sees list and says "2")
→ Second response: [
    {{"action": "download_pdf", "selection": 2}},
    {{"action": "extract_and_summarize_pdf"}},
    {{"action": "save_pdf_summary_as_note"}}
  ]

### END PDF WORKFLOW ###
### LABS MODULE (NEW) ###

If the user says:
- "run a hallucination test"
- "hallucination stress test"
- "run hallucination lab"
- "perform hallucination test"
- "start hallucination analysis"

Respond with EXACTLY:
{{ "action": "run_lab_hallucination", "query": "<user message>" }}

If the user says:
- "run code in lab"
- "execute this in lab"
- "test this code in lab"
- "lab run"
- "lab execute"

Respond with EXACTLY:
{{ "action": "run_lab_execute", "cmd": "<user message or code>" }}

### CRITICAL RULE FOR WEB SEARCH ###
If the user asks a general knowledge question, a real-time question (like sports scores, weather, or news), or any question you don't know the answer to, you MUST use the `search_web` action.
DO NOT answer from your own knowledge.

Example:
User: "who won india vs aus t20 series 2025"
→ Respond with:
{{"action": "search_web", "query": "who won india vs aus t20 series 2025"}}

User: "What is the capital of France?"
→ Respond with:
{{"action": "search_web", "query": "capital of France"}}
### END CRITICAL RULE ###

### CRITICAL RULE: PYTHON ENVIRONMENT SETUP ###
If the user asks to "setup an environment", "install packages", "run pip install", or "create a project" with specific libraries (e.g., "with GUI support", "using pandas"):
You MUST NOT run `sudo` or use `--break-system-packages`.
You MUST use the following *safe* 3-action sequence:
1.  `create_folder` (for the project)
2.  `create_file` (for the main script, e.g., `calculator.py`)
3.  `setup_python_environment` (This action creates the venv, requirements.txt, and setup.sh script)

Example:
User: "Setup an environment for a GUI calculator on my desktop."
→ Respond with:
[
  {{"action": "create_folder", "folder_path": "~/Desktop/CalculatorProject"}},
  {{"action": "create_file", "folder_path": "~/Desktop/CalculatorProject", "filename": "calculator.py", "content": "# A simple tkinter calculator\nimport tkinter as tk\n..."}},
  {{"action": "setup_python_environment", "project_path": "~/Desktop/CalculatorProject", "packages": ["tkinter"]}}
]

Example 2:
User: "Create a data analysis project in Documents using pandas and matplotlib."
→ Respond with:
[
  {{"action": "create_folder", "folder_path": "~/Documents/DataAnalysis"}},
  {{"action": "create_file", "folder_path": "~/Documents/DataAnalysis", "filename": "analysis.py", "content": "# Data analysis script\nimport pandas as pd\n..."}},
  {{"action": "setup_python_environment", "project_path": "~/Documents/DataAnalysis", "packages": ["pandas", "matplotlib"]}}
]
### END CRITICAL RULE ###

### CRITICAL RULE FOR FILE CREATION ###
If a user asks to create a file inside a specific folder (e.g., "in folder X", "on desktop in folder Y"),
you MUST *ALWAYS* return a `create_folder` action *first*, followed by the `create_file` action.
This ensures the folder exists before trying to write the file.

Example:
User: "In my Documents, make a 'tests' folder and create a file 'run.py' in it."
→ Respond with:
[
  {{"action": "create_folder", "folder_path": "~/Documents/tests"}},
  {{"action": "create_file", "folder_path": "~/Documents/tests", "filename": "run.py", "content": "# Python test file"}}
]
### END CRITICAL RULE ###

- chat: for info, answer, chitchat, or if no other action matches
- create_folder: {{"action":"create_folder", "folder_path":"/absolute/or/~/relative/path"}}
- create_file: {{"action":"create_file", "folder_path":"~/path", "filename":".extension", "content":"..."}}
  - If user asks for code, 'content' MUST be the full, working code.
  - If user asks for info ("about Bengalis"), 'content' MUST be a generated paragraph.
  - DO NOT just copy the user's words into 'content'.
- control_volume: {{"action": "control_volume","amount": -10 or 10}}
if user say kaha hai prediction.py or where is x.extension then respond with 
- file_exists: {{"action":"file_exists", "filename":"...", "type":"file" or "folder" or "any"}}
- play_music: {{"action":"play_music", "song":"Song Name"}}
- stop_music: {{"action":"stop_music"}}
- next_music: {{"action":"next_music"}}
- previous_music: {{"action":"previous_music"}}
- fix_code: {{"action":"fix_code", "file_path":"~/path/to/code.ext"}}
- get_weather: {{"action":"get_weather", "city":"Nagpur"}}
- move_file_folder: {{"action":"move_file_folder", "source_path":"~/path/to/file_or_folder", "destination_path":"~/path/to/destination_folder"}}
- wifi_status: {{"action":"wifi_status"}}
- bluetooth_devices: {{"action":"bluetooth_devices"}}
- connected_devices: {{"action":"connected_devices"}}
- general_knowledge: {{"action":"general_knowledge", "question":"Who is the CEO of Tesla?"}}
- process_document: {{"action":"process_document", "file_path":"~/Documents/file.pdf", "query":"Summarize" }}
- list_dir_contents: {{"action":"list_dir_contents", "path":"~/Downloads", "type":"files" or "folders" or "all"}}
- trash_files: {{"action":"trash_files", "path_pattern":"~/Downloads/old_file.txt"}}
- change_wallpaper: {{"action":"change_wallpaper", "image_path":"/path/to/image.jpg"}}
- system_usage: {{"action":"system_usage"}}
- network_info: {{"action":"network_info"}}
- delete_file: {{"action":"delete_file", "filepath":"/path/to/file.txt"}} # Only for files
- open_browser: {{"action":"open_browser"}}
- navigate_to: {{"action":"navigate_to", "url":"https..."}}
- search_website: {{"action":"search_website", "query":"..."}}
- save_note: {{"action": "save_note", "filename": "xyz.txt", "content": "your content"}}
- remind_me: {{"action": "remind_me", "message": "your message", "after_minutes": 10}}
- search_web: {{"action": "search_web", "query": "Who won India vs England"}}
- send_whatsapp: {{"action":"send_whatsapp", "contact":"...", "message":"..."}}
- tell_time: {{"action":"tell_time"}}
- rename_file: {{"action":"rename_file", "filepath":"~/Downloads/old.pdf", "newname":"new.pdf"}}
- tell_date: {{"action":"tell_date"}}
- announce: {{"action":"announce", "message":"..."}}
- system_info: {{"action":"system_info"}}
- battery_status: {{"action":"battery_status"}}
- change_brightness: {{"action":"change_brightness", "amount": -50}}
- extract_pdf_text: {{"action":"extract_pdf_text", "query":"..."}}
- get_chat_history: {{"action": "get_chat_history"}}

If the request does not match these, default to {{"action":"chat", "message":"..."}} with a concise answer.
Never output any explanation outside of JSON!
"""
    prompt = system_prompt + "\n\nUser message: " + user_message
    
    # Only call the LLM ONCE
    llm_out = get_llm_response(prompt, code_only=False).strip()
    
    # Clean up common LLM artifacts
    llm_out = re.sub(r'<think>.*?</think>', '', llm_out, flags=re.DOTALL).strip()
    llm_out = llm_out.strip('```json').strip('```').strip()

    # --- THIS IS THE FIX ---
    # The regex is now GREEDY (no '?'). It will find the longest match.
    match = re.search(r'(\[.*\]|\{.*\})', llm_out, re.DOTALL)
    # -----------------------

    if match:
        try:
            # Use json5 to be more tolerant of syntax
            parsed = json5.loads(match.group(1))
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception as e:
            print(f"❌ JSON parse error: {e}")
            print(f"Raw LLM output: {llm_out}")

    # fallback if no valid JSON matched
    # Return the raw output for debugging, wrapped in a chat action
    return [{"action": "chat", "message": f"LLM Error: {llm_out}"}]

# ========== MAIN ACTION MAPPING ==========

action_mapping = {
    "chat": lambda args: do_chat(args.get("message")),
    "search_pdfs": lambda args: do_search_pdfs(args.get("topic"), int(args.get("max_results", 5))),
    "run_lab_hallucination": lambda args: labs.run_hallucination_lab(args.get("query")),
    "run_lab_execute": lambda args: labs.run_code_lab(args.get("cmd")),
    "schedule_meet": lambda args: do_schedule_meet(
        args.get("title"), args.get("date"), args.get("start_time"), 
        args.get("end_time"), args.get("attendees", [])
    ),
    "download_pdf": lambda args: do_download_pdf(args.get("selection")),
    "extract_and_summarize_pdf": lambda args: do_extract_and_summarize_pdf(args.get("file_path")),
    "save_pdf_summary_as_note": lambda args: do_save_pdf_summary_as_note(),
    "remember_fact": lambda args: do_remember_fact(args.get("key"), args.get("value")),
    "get_fact": lambda args: do_get_fact(args.get("key")),
    "get_chat_history": lambda args: do_get_chat_history(),
    "remember_name": lambda args: do_remember_name(args.get("name")), # Legacy
    "get_name": lambda args: do_get_name(), # Legacy
    "create_folder": lambda args: do_create_folder(args.get("folder_path"), args.get("filename")),
    "create_project": lambda args: do_create_project(args.get("project_name"), args.get("location"), args.get("language"), args.get("gui")),
    "create_file": lambda args: do_create_file(args.get("folder_path"), args.get("filename"), args.get("content")),
    "file_exists": lambda args: do_file_exists(args.get("filename"), args.get("type", "any")),
    "open_file": lambda args: do_open_file(args.get("filename")),
    "play_music": lambda args: do_play_music(args.get("song")),
    "process_document": lambda args: do_process_document(args.get("file_path"), args.get("query")),
    "stop_music": lambda args: do_stop_music(),
    "next_music": lambda args: do_control_media("next"),
    "previous_music": lambda args: do_control_media("previous"),
    "fix_code": lambda args: do_fix_code(args.get("file_path")),
    "control_volume": lambda args: do_control_volume(args.get("amount", 0)),
    "search_web": lambda args: do_search_web(args.get("query")),
    "open_browser": lambda args: do_open_browser(),
    "navigate_to": lambda args: do_navigate_to(args.get("url")),
    "search_website": lambda args: do_search_website(args.get("query")),
    "send_whatsapp": lambda args: do_send_whatsapp(args.get("contact"), args.get("message")),
    "send_email": lambda args: send_interactive_email(args.get("recipient"), args.get("message"), args.get("subject", "")),
    "system_usage": lambda args: do_get_system_usage(),
    "get_weather": lambda args: do_get_weather(args.get("city", "Nagpur")),
    "wifi_status": lambda args: do_wifi_status(),
    "bluetooth_devices": lambda args: do_bluetooth_devices(),
    "connected_devices": lambda args: do_connected_devices(),
    "general_knowledge": lambda args: do_general_knowledge(args.get("question")),
    "save_note": lambda args: do_save_note(args.get("filename"), args.get("content")),
    "remind_me": lambda args: do_remind_me(args.get("message"), int(args.get("after_minutes", 5))),
    "trash_files": lambda args: do_trash_files(args.get("path_pattern")),
    "network_info": lambda args: do_get_network_info(),
    "move_file_folder": lambda args: do_move_file_folder(args.get("source_path"), args.get("destination_path")),
    "delete_file": lambda args: do_delete_file(args.get("filepath")),
    "change_wallpaper": lambda args: do_change_wallpaper(args.get("image_path")),
    "tell_time": lambda args: do_tell_time(),
    "tell_date": lambda args: do_tell_date(),
    "announce": lambda args: do_announce(args.get("message")),
    "system_info": lambda args: do_system_info(),
    "battery_status": lambda args: do_battery_status(),
    "change_brightness": lambda args: do_change_brightness(int(args.get("amount", 0))),
    "extract_pdf_text": lambda args: do_extract_pdf_text(args.get("query")),
    "rename_file": lambda args: do_rename_file(args.get("filepath"), args.get("newname")),
    "list_dir_contents": lambda args: do_list_dir_contents(args.get("path", ""), args.get("type", "all")),
    "setup_python_environment": lambda args: do_setup_python_environment(args.get("project_path"), args.get("packages")),
}

# ========== MAIN INTENT HANDLER ==========

def handle_intent(user_input):
    user_message = user_input.get('message', '') if isinstance(user_input, dict) else user_input
    
    # 1. Save user message to permanent history
    update_chat_history(user_message=user_message)

    # 2. Resolve references (like "that file")
    user_message = resolve_references_in_message(user_message)

    # 3. Extract actions from LLM (now with history context)
    actions = extract_llm_intent(user_message)
    results = []
    
    # Ensure actions is a list
    if not isinstance(actions, list):
        actions = [actions]

    for action in actions:
        if not isinstance(action, dict) or "action" not in action:
            results.append(f"❌ Invalid action format: {action}")
            continue

        # 4. Update short-term context (for "that")
        update_context_from_action(action)
        
        action_name = action.get("action")
        handler = action_mapping.get(action_name)
        
        if callable(handler):
            try:
                # Pass all args except 'action' as a single dict
                handler_args = {k: v for k, v in action.items() if k != "action"}
                handler_result = handler(handler_args) # Pass as dictionary
            except Exception as e:
                handler_result = f"❌ Error executing '{action_name}': {e}"
        else:
            handler_result = f"❌ Unknown action: '{action_name}'"

        # Speak short confirmation
        if action_name not in ["chat", "get_fact", "get_chat_history", "tell_time", "tell_date", "get_weather", "system_usage", "wifi_status", "network_info", "system_info", "battery_status"]:
            speak_response("Done")
        elif action_name == "chat":
            speak_response("Okay") # Or speak the chat message itself
            # speak_response(handler_result) 
            
        results.append(handler_result)

    # 5. Combine results and save assistant response to history
    if results:
        combined_result = "\n".join([str(r) for r in results])
        update_chat_history(assistant_message=combined_result) # Save response
        return combined_result
    else:
        fallback_response = "I'm sorry, I couldn't process your request."
        update_chat_history(assistant_message=fallback_response) # Save response
        return fallback_response

# ========== VOICE CONTROL LOOP ==========

def voice_control_loop():
    """Main voice control loop with wake word detection"""
    print("🚀 Starting Zuno Voice Assistant...")
    
    wake_detector.start_listening()
    
    try:
        while True:
            command = None
            try:
                # Check for command from wake word activation
                command = wake_detector.command_queue.get_nowait()
                if command:
                    print(f"🎯 Processing (from queue): {command}")
            except queue.Empty:
                pass
            
            # If awake, listen for follow-up command
            if not command and wake_detector.is_awake:
                command = wake_detector.listen_for_command()
                if command:
                    print(f"🎯 Processing (follow-up): {command}")
                else:
                    # No command heard, check for sleep
                    wake_detector.should_sleep()

            if command:
                speak_response("Okay")
                user_input = {'intent': 'general_query', 'message': command}
                result = handle_intent(user_input)
                print(f"📝 Result: {result}")
                
                # Speak the result, but keep it short
                if len(result) > 150:
                    speak_response("Task complete. The result is on your screen.")
                else:
                    speak_response(result)
                
                # Command processed, reset wake time
                if wake_detector.is_awake:
                    wake_detector.last_wake_time = time.time()
            
            time.sleep(0.1) # Main loop sleep
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping Zuno Voice Assistant...")
        wake_detector.stop_listening()

# ========== MAIN FUNCTION ==========

def main():
    print("👿 Lucifer Agent Ready. Choose mode:")
    print("1. Voice Mode (Say 'Hey Zuno')")
    print("2. Text Mode (Type commands)")
    print("3. Exit")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        voice_control_loop()
    elif choice == "2":
        print("👿 Text Mode Active. Type commands (type 'exit' or 'quit' to stop).")
        while True:
            try:
                q = input("👿 Command me: ").strip()
                if q.lower() in ("exit", "quit"):
                    print("👿 Lucifer Agent terminated.")
                    break
                if not q:
                    continue
                
                # Special local command (example)
                if q.lower().startswith("pdf"):
                    query = q[3:].strip()
                    print(do_extract_pdf_text(query))
                    continue
                
                user_input = {
                    'intent': 'general_query',
                    'message': q,
                    'entities': {}
                }
                result = handle_intent(user_input)
                print(result)
            except (KeyboardInterrupt, EOFError):
                print("\n👿 Agent interrupted.")
                break
    else:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
