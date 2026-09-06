import threading
import time
import winsound
import subprocess
import queue


class AlarmManager:
    def __init__(self):
        # Main alarm control
        self.alarm_active = threading.Event()

        # Prevent multiple alarm types from running simultaneously
        self.current_alarm = None
        self.lock = threading.Lock()

        # Speech queue
        self.speech_queue = queue.Queue()
        self.speech_worker_running = True

        # Start one permanent speech worker
        self.speech_thread = threading.Thread(
            target=self._speech_worker,
            daemon=True
        )
        self.speech_thread.start()

        # Prevent repeated phone/distraction alerts
        self.last_spoken = {}
        self.speak_cooldown = 3.0

    # =========================================================
    # SPEECH
    # =========================================================

    def _speech_worker(self):
        """
        Dedicated speech thread.

        This keeps text-to-speech completely separate from:
        - video processing
        - alarm beep
        - Streamlit/WebRTC
        """

        while self.speech_worker_running:

            try:
                message = self.speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if message is None:
                break

            try:
                self._speak_windows(message)
            except Exception as e:
                print(f"Speech error: {e}")

            self.speech_queue.task_done()

    def _speak_windows(self, message):
        """
        Windows built-in SpeechSynthesizer.

        Runs independently from the beep thread.
        """

        # Escape single quotes for PowerShell
        safe_message = message.replace("'", "''")

        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speak.Rate = 0; "
            "$speak.Volume = 100; "
            f"$speak.Speak('{safe_message}'); "
            "$speak.Dispose();"
        )

        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                command
            ],
            creationflags=subprocess.CREATE_NO_WINDOW,
            timeout=10
        )

    def speak(self, message):
        """
        Add speech to the queue.

        IMPORTANT:
        This function does NOT wait for speech to finish.
        """

        if not message:
            return

        self.speech_queue.put(message)

    def speak_alert(self, alert_type):
        """
        One-shot voice alerts.

        These are used for phone/distraction events.
        """

        messages = {
            "phone": "Warning. Please put the phone down.",
            "distraction": "Warning. Please keep your eyes on the road.",
            "drowsy": "Warning. You appear to be drowsy. Please wake up.",
            "critical": "Critical warning. Driver is not responding.",
            "emergency": "Emergency detected. Emergency contact will be notified."
        }

        message = messages.get(alert_type)

        if message:
            self.speak(message)

    # =========================================================
    # BEEP
    # =========================================================

    def _beep_loop(self, scenario):
        """
        Continuous alarm beep.

        Runs on its OWN thread, so speech can happen simultaneously.
        """

        while self.alarm_active.is_set():

            try:

                if scenario == "drowsy":
                    winsound.Beep(1000, 300)
                    time.sleep(0.2)

                elif scenario == "critical":
                    winsound.Beep(1400, 400)
                    time.sleep(0.15)

                elif scenario == "emergency":
                    winsound.Beep(1800, 500)
                    time.sleep(0.1)

                else:
                    winsound.Beep(1000, 300)
                    time.sleep(0.2)

            except Exception as e:
                print(f"Beep error: {e}")
                break

    # =========================================================
    # REPEATED VOICE
    # =========================================================

    def _voice_loop(self, scenario):
        """
        Repeats the appropriate sentence while the alarm is active.

        Beep and voice are completely independent.
        """

        messages = {
            "drowsy":
                "Warning. You appear to be drowsy. Please wake up.",

            "critical":
                "Critical warning. Driver is not responding.",

            "emergency":
                "Emergency detected. Please respond immediately."
        }

        message = messages.get(scenario)

        if not message:
            return

        # Wait before repeating
        interval = {
            "drowsy": 4.0,
            "critical": 3.0,
            "emergency": 3.0
        }.get(scenario, 4.0)

        while self.alarm_active.is_set():

            self.speak(message)

            # Wait while still allowing the alarm to stop
            if self.alarm_active.wait(interval):
                break

    # =========================================================
    # START ALARMS
    # =========================================================

    def _start_alarm(self, scenario):
        """
        Internal common alarm starter.
        """

        with self.lock:

            # Already running same alarm
            if self.alarm_active.is_set() and self.current_alarm == scenario:
                return

            # Stop previous alarm
            self.alarm_active.clear()

            self.current_alarm = scenario
            self.alarm_active.set()

            print(f"🔊 Starting {scenario} alarm")

            # -------------------------------------------------
            # BEEP THREAD
            # -------------------------------------------------

            beep_thread = threading.Thread(
                target=self._beep_loop,
                args=(scenario,),
                daemon=True
            )

            beep_thread.start()

            # -------------------------------------------------
            # VOICE THREAD
            # -------------------------------------------------

            voice_thread = threading.Thread(
                target=self._voice_loop,
                args=(scenario,),
                daemon=True
            )

            voice_thread.start()

    def start_alarm(self, scenario="drowsy"):
        self._start_alarm(scenario)

    def start_critical_alarm(self):
        self._start_alarm("critical")

    def start_emergency_alarm(self):
        self._start_alarm("emergency")

    # =========================================================
    # STOP
    # =========================================================

    def stop_alarm(self):
        """
        Stop beep + repeated voice alarm.
        """

        with self.lock:

            if not self.alarm_active.is_set():
                return

            print("🛑 Alarm stopped")

            self.alarm_active.clear()
            self.current_alarm = None

            # Clear pending speech so old alarm messages
            # don't play after recovery.
            try:
                while True:
                    self.speech_queue.get_nowait()
                    self.speech_queue.task_done()
            except queue.Empty:
                pass

    # =========================================================
    # CLEANUP
    # =========================================================

    def shutdown(self):
        """
        Stop speech worker when application closes.
        """

        self.stop_alarm()

        self.speech_worker_running = False
        self.speech_queue.put(None)