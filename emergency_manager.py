import os
import threading
from datetime import datetime

from dotenv import load_dotenv
from twilio.rest import Client


load_dotenv()


class EmergencyManager:

    def __init__(self, emergency_contact="+919361010422"):

        self.emergency_contact = emergency_contact

        self.emergency_triggered = False
        self.lock = threading.Lock()

        self.current_location = None

        # =====================================================
        # TWILIO CONFIGURATION
        # =====================================================

        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")

        self.twilio_client = None

        if (
            self.twilio_account_sid
            and self.twilio_auth_token
            and self.twilio_phone_number
        ):
            try:
                self.twilio_client = Client(
                    self.twilio_account_sid,
                    self.twilio_auth_token
                )

                print("✅ Twilio SMS service initialized.")

            except Exception as e:
                print(f"❌ Twilio initialization failed: {e}")

        else:
            print("⚠️ Twilio configuration is incomplete.")

    # =========================================================
    # UPDATE GPS LOCATION
    # =========================================================

    def update_location(self, latitude, longitude):

        try:
            latitude = float(latitude)
            longitude = float(longitude)

            self.current_location = {
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": datetime.now().isoformat()
            }

            print(
                f"📍 GPS updated: "
                f"{latitude}, {longitude}"
            )

            return True

        except (TypeError, ValueError) as e:

            print(f"❌ Invalid GPS coordinates: {e}")

            return False

    # =========================================================
    # GET CURRENT LOCATION
    # =========================================================

    def get_location(self):
        return self.current_location

    # =========================================================
    # GOOGLE MAPS LINK
    # =========================================================

    def get_maps_link(self, latitude, longitude):

        return (
            "https://www.google.com/maps/"
            f"search/?api=1&query={latitude},{longitude}"
        )

    # =========================================================
    # CREATE EMERGENCY MESSAGE
    # =========================================================

    def create_emergency_message(self, location=None):

        if location is None:
            location = self.get_location()

        current_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # -----------------------------------------------------
        # GPS AVAILABLE
        # -----------------------------------------------------

        if location:

            latitude = location["latitude"]
            longitude = location["longitude"]

            maps_link = self.get_maps_link(
                latitude,
                longitude
            )

            message = (
                "🚨 DRIVER EMERGENCY ALERT 🚨\n\n"
                "The driver has not responded to repeated "
                "drowsiness warnings.\n\n"
                f"Time: {current_time}\n\n"
                f"Latitude: {latitude}\n"
                f"Longitude: {longitude}\n\n"
                "📍 Driver Location:\n"
                f"{maps_link}\n\n"
                "Please check on the driver immediately."
            )

        # -----------------------------------------------------
        # GPS NOT AVAILABLE
        # -----------------------------------------------------

        else:

            message = (
                "🚨 DRIVER EMERGENCY ALERT 🚨\n\n"
                "The driver has not responded to repeated "
                "drowsiness warnings.\n\n"
                f"Time: {current_time}\n\n"
                "⚠️ GPS location is currently unavailable.\n\n"
                "Please check on the driver immediately."
            )

        return message

    # =========================================================
    # SEND SMS
    # =========================================================

        # =========================================================
    # SEND SMS
    # =========================================================

    def send_sms(self, message):

        if self.twilio_client is None:

            print("❌ Twilio is not configured.")
            return False

        if not self.emergency_contact:

            print("❌ Emergency contact is missing.")
            return False

        try:

            # -------------------------------------------------
            # TWILIO TRIAL TEMPLATE
            # -------------------------------------------------
            #
            # Trial accounts do not allow custom SMS bodies.
            # "sms_internal_alerts" is one of Twilio's
            # predefined trial SMS templates.
            #
            sms = self.twilio_client.messages.create(
                body="sms_internal_alerts",
                from_=self.twilio_phone_number,
                to=self.emergency_contact
            )

            print("\n" + "=" * 60)
            print("📱 SMS SENT SUCCESSFULLY")
            print("=" * 60)
            print(f"Emergency Contact: {self.emergency_contact}")
            print(f"Twilio Message SID: {sms.sid}")
            print("=" * 60 + "\n")

            return True

        except Exception as e:

            print("\n" + "=" * 60)
            print("❌ SMS SENDING FAILED")
            print("=" * 60)
            print(f"Error: {e}")
            print("=" * 60 + "\n")

            return False

    # =========================================================
    # SEND EMERGENCY ALERT
    # =========================================================

    def send_emergency_alert(self):

        # -----------------------------------------------------
        # PREVENT DUPLICATE ALERTS
        # -----------------------------------------------------

        with self.lock:

            if self.emergency_triggered:

                print(
                    "⚠️ Emergency alert already triggered."
                )

                return False

            # Mark as triggered before sending
            # so multiple calls cannot send duplicate SMS.
            self.emergency_triggered = True

        print("\n" + "=" * 60)
        print("🚨 EMERGENCY ALERT TRIGGERED")
        print("=" * 60)

        # -----------------------------------------------------
        # GET LATEST GPS
        # -----------------------------------------------------

        location = self.get_location()

        # -----------------------------------------------------
        # CREATE MESSAGE
        # -----------------------------------------------------

        message = self.create_emergency_message(location)

        print("\n📱 Emergency Contact:")
        print(self.emergency_contact)

        print("\n📨 Emergency Message:")
        print(message)

        # -----------------------------------------------------
        # SEND SMS
        # -----------------------------------------------------

        success = self.send_sms(message)

        # -----------------------------------------------------
        # IF SMS FAILED
        # -----------------------------------------------------

        if not success:

            with self.lock:
                self.emergency_triggered = False

            print(
                "⚠️ Emergency alert was not sent."
            )

            return False

        print(
            "✅ Emergency SMS process completed."
        )

        return True

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        with self.lock:
            self.emergency_triggered = False

        print("🔄 Emergency manager reset.")

    # =========================================================
    # EMERGENCY STATUS
    # =========================================================

    def is_emergency_triggered(self):
        return self.emergency_triggered