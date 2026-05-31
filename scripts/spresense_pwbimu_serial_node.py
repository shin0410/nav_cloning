#!/usr/bin/env python3
import re
import struct
import time

import rospy
import serial
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32


HEX_CSV_RE = re.compile(r"^[0-9a-fA-F]{8}(,[0-9a-fA-F]{8}){7}$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def hex_to_float(value):
    return struct.unpack(">f", bytes.fromhex(value))[0]


def covariance_param(name, default_diag):
    value = rospy.get_param(name, None)
    if value is None:
        return [
            default_diag,
            0.0,
            0.0,
            0.0,
            default_diag,
            0.0,
            0.0,
            0.0,
            default_diag,
        ]
    if isinstance(value, (int, float)):
        diag = float(value)
        return [diag, 0.0, 0.0, 0.0, diag, 0.0, 0.0, 0.0, diag]
    if isinstance(value, list) and len(value) == 9:
        return [float(x) for x in value]
    raise ValueError("{} must be a scalar or 9-element list".format(name))


def clean_line(raw):
    line = raw.decode("ascii", errors="ignore").strip()
    line = ANSI_RE.sub("", line)
    if "nsh>" in line:
        line = line.split("nsh>", 1)[1].strip()
    return "".join(ch for ch in line if ch.isprintable())


class SpresensePwbImuSerialNode:
    def __init__(self):
        self.port = rospy.get_param("~port", "/dev/ttyUSB0")
        self.baud = int(rospy.get_param("~baud", 115200))
        self.frame_id = rospy.get_param("~frame_id", "spresense_imu_link")
        self.start_logger = bool(rospy.get_param("~start_logger", True))
        self.start_command = rospy.get_param(
            "~start_command", "pwbimu_logger -s 120 -a 16 -g 500 -f 1 -o uart"
        )
        self.command_delay = float(rospy.get_param("~command_delay", 0.5))
        self.startup_wait = float(rospy.get_param("~startup_wait", 2.0))
        self.start_attempts = int(rospy.get_param("~start_attempts", 3))
        self.resend_start_after = float(rospy.get_param("~resend_start_after", 3.0))
        self.log_ignored_lines = bool(rospy.get_param("~log_ignored_lines", True))
        self.ignored_log_limit = int(rospy.get_param("~ignored_log_limit", 20))
        self.ignored_log_count = 0
        self.valid_count = 0
        self.start_attempt_count = 0
        self.last_start_wall = 0.0

        imu_topic = rospy.get_param("~imu_topic", "/spresense/imu/data_raw")
        temp_topic = rospy.get_param("~temperature_topic", "/spresense/imu/temperature")

        self.angular_velocity_covariance = covariance_param(
            "~angular_velocity_covariance", 0.01
        )
        self.linear_acceleration_covariance = covariance_param(
            "~linear_acceleration_covariance", 0.1
        )

        self.imu_pub = rospy.Publisher(imu_topic, Imu, queue_size=50)
        self.temp_pub = rospy.Publisher(temp_topic, Float32, queue_size=10)

        self.serial = serial.Serial(self.port, self.baud, timeout=1.0)
        rospy.on_shutdown(self.close)

        if self.startup_wait > 0.0:
            rospy.loginfo("Waiting %.1f seconds for Spresense serial prompt", self.startup_wait)
            time.sleep(self.startup_wait)

        if self.start_logger:
            self.send_start_command()

    def send_start_command(self):
        # Wake NSH prompt, discard banner/prompt text, then start UART logging.
        self.serial.write(b"\r\n")
        self.serial.flush()
        time.sleep(self.command_delay)
        self.serial.reset_input_buffer()
        self.serial.write((self.start_command + "\r\n").encode("ascii"))
        self.serial.flush()
        self.start_attempt_count += 1
        self.last_start_wall = time.monotonic()
        rospy.loginfo(
            "Started Spresense IMU logger attempt %d/%d: %s",
            self.start_attempt_count,
            self.start_attempts,
            self.start_command,
        )

    def close(self):
        if hasattr(self, "serial") and self.serial and self.serial.is_open:
            self.serial.close()

    def parse_line(self, line):
        if not HEX_CSV_RE.match(line):
            return None

        fields = line.split(",")
        timestamp = int(fields[0], 16)
        temp = hex_to_float(fields[1])
        gx = hex_to_float(fields[2])
        gy = hex_to_float(fields[3])
        gz = hex_to_float(fields[4])
        ax = hex_to_float(fields[5])
        ay = hex_to_float(fields[6])
        az = hex_to_float(fields[7])
        return timestamp, temp, gx, gy, gz, ax, ay, az

    def maybe_retry_start_command(self):
        if not self.start_logger or self.valid_count > 0:
            return
        if self.start_attempt_count >= self.start_attempts:
            return
        if time.monotonic() - self.last_start_wall < self.resend_start_after:
            return
        rospy.logwarn("No valid Spresense IMU CSV yet; retrying start command")
        self.send_start_command()

    def log_ignored_line(self, line):
        if not self.log_ignored_lines or not line:
            return
        if self.ignored_log_count >= self.ignored_log_limit:
            return
        self.ignored_log_count += 1
        rospy.loginfo("Ignoring serial line %d: %r", self.ignored_log_count, line)

    def spin(self):
        rospy.loginfo("Reading Spresense IMU from %s at %d baud", self.port, self.baud)
        while not rospy.is_shutdown():
            try:
                line = clean_line(self.serial.readline())
                parsed = self.parse_line(line)
                if parsed is None:
                    self.log_ignored_line(line)
                    self.maybe_retry_start_command()
                    continue

                _, temp, gx, gy, gz, ax, ay, az = parsed
                self.valid_count += 1
                if self.valid_count == 1:
                    rospy.loginfo("Received first valid Spresense IMU CSV")
                stamp = rospy.Time.now()

                msg = Imu()
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                msg.orientation.w = 1.0
                msg.orientation_covariance[0] = -1.0
                msg.angular_velocity.x = gx
                msg.angular_velocity.y = gy
                msg.angular_velocity.z = gz
                msg.angular_velocity_covariance = self.angular_velocity_covariance
                msg.linear_acceleration.x = ax
                msg.linear_acceleration.y = ay
                msg.linear_acceleration.z = az
                msg.linear_acceleration_covariance = self.linear_acceleration_covariance
                self.imu_pub.publish(msg)
                self.temp_pub.publish(Float32(data=temp))
            except serial.SerialException as exc:
                if rospy.is_shutdown():
                    break
                rospy.logerr("Serial error on %s: %s", self.port, exc)
                rospy.sleep(1.0)
            except (ValueError, struct.error) as exc:
                rospy.logwarn_throttle(5.0, "Failed to parse IMU line: %s", exc)


if __name__ == "__main__":
    rospy.init_node("spresense_pwbimu_serial_node")
    try:
        SpresensePwbImuSerialNode().spin()
    except serial.SerialException as exc:
        rospy.logfatal("Could not open Spresense serial port: %s", exc)
    except ValueError as exc:
        rospy.logfatal("Bad parameter: %s", exc)
