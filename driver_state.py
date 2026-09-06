import time
from enum import Enum


class DriverState(Enum):
    NORMAL = "NORMAL"
    DROWSY = "DROWSY"
    ALARM = "ALARM"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class DriverStateMachine:

    def __init__(
        self,
        drowsy_duration=2.0,
        critical_duration=10.0,
        emergency_duration=10.0
    ):
        self.state = DriverState.NORMAL

        # Time thresholds
        self.drowsy_duration = drowsy_duration
        self.critical_duration = critical_duration
        self.emergency_duration = emergency_duration

        # Timers
        self.drowsy_start_time = None
        self.critical_start_time = None
        self.emergency_start_time = None

        self.last_recovery_time = time.time()

    def update(self, drowsy_detected, driver_recovered=False):
        """
        Update driver state based on detection signals.

        drowsy_detected:
            True when the detection system believes the driver is drowsy.

        driver_recovered:
            True when the driver has returned to a normal state.
        """

        now = time.time()

        # ==================================================
        # DRIVER RECOVERED
        # ==================================================
        if driver_recovered:

            self.state = DriverState.NORMAL

            self.drowsy_start_time = None
            self.critical_start_time = None
            self.emergency_start_time = None

            self.last_recovery_time = now

            return self.state

        # ==================================================
        # NORMAL
        # ==================================================
        if self.state == DriverState.NORMAL:

            if drowsy_detected:

                # Start drowsiness timer
                if self.drowsy_start_time is None:
                    self.drowsy_start_time = now

                elapsed = now - self.drowsy_start_time

                # Drowsiness persisted long enough
                if elapsed >= self.drowsy_duration:

                    self.state = DriverState.ALARM

                    # Reset timers for next stage
                    self.drowsy_start_time = None
                    self.critical_start_time = None
                    self.emergency_start_time = None

            else:

                # Driver is normal again
                self.drowsy_start_time = None

        # ==================================================
        # ALARM
        # ==================================================
        elif self.state == DriverState.ALARM:

            if drowsy_detected:

                # Start timer when ALARM begins
                if self.critical_start_time is None:
                    self.critical_start_time = now

                elapsed = now - self.critical_start_time

                # Drowsiness continues for 10 seconds
                if elapsed >= self.critical_duration:

                    self.state = DriverState.CRITICAL

                    # Start CRITICAL timer
                    self.emergency_start_time = now

            else:

                # Driver responded/recovered
                self.state = DriverState.NORMAL

                self.drowsy_start_time = None
                self.critical_start_time = None
                self.emergency_start_time = None

        # ==================================================
        # CRITICAL
        # ==================================================
        elif self.state == DriverState.CRITICAL:

            if drowsy_detected:

                # emergency_start_time was started
                # when CRITICAL was entered
                if self.emergency_start_time is None:
                    self.emergency_start_time = now

                elapsed = now - self.emergency_start_time

                # No recovery for another 10 seconds
                if elapsed >= self.emergency_duration:

                    self.state = DriverState.EMERGENCY

            else:

                # Driver recovered
                self.state = DriverState.NORMAL

                self.drowsy_start_time = None
                self.critical_start_time = None
                self.emergency_start_time = None

        # ==================================================
        # EMERGENCY
        # ==================================================
        elif self.state == DriverState.EMERGENCY:

            # Emergency remains active until
            # driver recovery is confirmed.
            #
            # driver_recovered is handled at the
            # beginning of this function.

            pass

        return self.state

    # ======================================================
    # GET CURRENT STATE
    # ======================================================

    def get_state(self):
        return self.state

    # ======================================================
    # GET STATE TEXT
    # ======================================================

    def get_state_text(self):
        return self.state.value