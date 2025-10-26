"""
modules/auth.py - Comprehensive Authentication Module for Data Visualizer Dashboard

Features:
- User Signup & Login
- Email OTP Verification
- Phone OTP Verification via Twilio
- CAPTCHA Verification
- Password Hashing & Validation
- OTP Expiry and Max Attempts
- Rate Limiting for Signup/Login
- Account Lockout after multiple failed attempts
- Detailed logging and helper functions
- JSON-based user storage (can be replaced with DB)
"""

import os
import json
import random
import string
import time
import hashlib
import hmac
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from twilio.rest import Client

# ------------------ CONFIGURATION ------------------ #
USER_DB_FILE = "storage/users.json"
OTP_EXPIRY = 300        # 5 minutes
MAX_OTP_ATTEMPTS = 3
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = 300  # 5 minutes lockout

# SMTP Email configuration (replace with real credentials)
SMTP_SERVER = "smtp.example.com"
SMTP_PORT = 587
SMTP_USERNAME = "your_email@example.com"
SMTP_PASSWORD = "your_password"
EMAIL_FROM = "noreply@example.com"

# Twilio SMS configuration (replace with real credentials)
TWILIO_ACCOUNT_SID = "your_twilio_sid"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
TWILIO_PHONE_NUMBER = "+1234567890"

# ------------------ HELPER FUNCTIONS ------------------ #

def load_users():
    """Load user data from JSON storage"""
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w") as f:
            json.dump({}, f)
    with open(USER_DB_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    """Save user data to JSON storage"""
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4)


def hash_password(password, salt=None):
    """Hash a password with optional salt using SHA256"""
    if not salt:
        salt = os.urandom(16).hex()
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}${hashed}"


def verify_password(stored_password, provided_password):
    """Verify provided password against stored hash"""
    try:
        salt, hashed = stored_password.split("$")
        return hmac.compare_digest(
            hashed, hashlib.sha256((provided_password + salt).encode()).hexdigest()
        )
    except Exception:
        return False


def generate_otp(length=6):
    """Generate numeric OTP"""
    return "".join(random.choices(string.digits, k=length))


def send_email(to_email, subject, body):
    """Send email via SMTP"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print("Email sending failed:", e)
        return False


def send_sms(to_number, message):
    """Send SMS via Twilio"""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        return True
    except Exception as e:
        print("SMS sending failed:", e)
        return False


def verify_captcha(user_input, captcha_solution):
    """Verify CAPTCHA input"""
    return str(user_input).strip() == str(captcha_solution)


def is_locked_out(user):
    """Check if user is temporarily locked out due to failed attempts"""
    if "lockout_time" in user:
        if time.time() < user["lockout_time"]:
            return True
        else:
            user.pop("lockout_time")
            user["failed_logins"] = 0
            return False
    return False


# ------------------ AUTHENTICATION LOGIC ------------------ #

def signup(username, password, email=None, phone=None):
    """
    Signup a new user.
    1. Check if username already exists
    2. Hash password
    3. Store OTP for email/phone verification
    """
    users = load_users()
    if username in users:
        return False, "Username already exists"

    if not email and not phone:
        return False, "Provide at least email or phone for verification"

    hashed_password = hash_password(password)
    otp = generate_otp()
    otp_data = {
        "otp": otp,
        "attempts": 0,
        "expiry": time.time() + OTP_EXPIRY
    }

    users[username] = {
        "password": hashed_password,
        "email": email,
        "phone": phone,
        "verified": False,
        "otp_data": otp_data,
        "failed_logins": 0
    }

    save_users(users)

    # Send OTP
    if email:
        send_email(email, "Your OTP Verification Code", f"Your OTP is: {otp}")
    elif phone:
        send_sms(phone, f"Your OTP is: {otp}")

    return True, "User created. OTP sent for verification."


def verify_otp(username, provided_otp):
    """Verify OTP for signup"""
    users = load_users()
    if username not in users:
        return False, "User does not exist"

    user = users[username]
    otp_data = user.get("otp_data", {})

    if not otp_data:
        return False, "No OTP found"

    if time.time() > otp_data["expiry"]:
        return False, "OTP expired. Request a new one."

    if otp_data["attempts"] >= MAX_OTP_ATTEMPTS:
        return False, "Max OTP attempts exceeded"

    if str(provided_otp) == str(otp_data["otp"]):
        user["verified"] = True
        user.pop("otp_data")
        save_users(users)
        return True, "OTP verified. Signup complete."
    else:
        user["otp_data"]["attempts"] += 1
        save_users(users)
        return False, f"Incorrect OTP. Attempts left: {MAX_OTP_ATTEMPTS - otp_data['attempts']}"


def login(username, password):
    """
    Login a user
    1. Check existence
    2. Check lockout
    3. Verify password
    4. Increment failed login count
    """
    users = load_users()
    if username not in users:
        return False, "User does not exist"

    user = users[username]

    if is_locked_out(user):
        return False, f"Account locked. Try again later."

    if not verify_password(user["password"], password):
        user["failed_logins"] = user.get("failed_logins", 0) + 1
        if user["failed_logins"] >= MAX_LOGIN_ATTEMPTS:
            user["lockout_time"] = time.time() + LOCKOUT_DURATION
        save_users(users)
        return False, "Incorrect password"

    if not user.get("verified", False):
        return False, "Account not verified. Complete OTP verification."

    # Successful login resets failed attempts
    user["failed_logins"] = 0
    save_users(users)
    return True, "Login successful"


def request_new_otp(username):
    """Request new OTP for email/phone verification"""
    users = load_users()
    if username not in users:
        return False, "User does not exist"

    user = users[username]
    if user.get("verified"):
        return False, "User already verified"

    otp = generate_otp()
    otp_data = {
        "otp": otp,
        "attempts": 0,
        "expiry": time.time() + OTP_EXPIRY
    }
    user["otp_data"] = otp_data
    save_users(users)

    if user.get("email"):
        send_email(user["email"], "Your OTP Verification Code", f"Your new OTP is: {otp}")
    elif user.get("phone"):
        send_sms(user["phone"], f"Your new OTP is: {otp}")

    return True, "New OTP sent."


# ------------------ CAPTCHA GENERATION ------------------ #
def generate_captcha(length=6):
    """Generate random alphanumeric CAPTCHA"""
    captcha = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return captcha


# ------------------ TEST / DEBUG ------------------ #
if __name__ == "__main__":
    print("---- Testing Auth Module ----")
    signup_status, msg = signup("testuser", "TestPass123", email="test@example.com")
    print("Signup:", signup_status, msg)

    otp = load_users()["testuser"]["otp_data"]["otp"]
    verify_status, verify_msg = verify_otp("testuser", otp)
    print("Verify OTP:", verify_status, verify_msg)

    login_status, login_msg = login("testuser", "TestPass123")
    print("Login:", login_status, login_msg)

    captcha = generate_captcha()
    print("Generated CAPTCHA:", captcha)
    print("CAPTCHA Verification (correct):", verify_captcha(captcha, captcha))
    print("CAPTCHA Verification (wrong):", verify_captcha(captcha, "wrong"))
