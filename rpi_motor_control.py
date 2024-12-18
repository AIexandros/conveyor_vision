from flask import Flask, render_template_string, request 
from time import sleep
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib

def control_motor(direction_case):
    #define GPIO pins
    GPIO_pins = (14, 15, 18) # Microstep Resolution MS1-MS3 -> GPIO Pin
    direction = 20       # Direction -> GPIO Pin
    step = 21            # Step -> GPIO Pin
    Enable_Pin = 16

    # Set up GPIO mode
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(Enable_Pin, GPIO.OUT)

    # Declare an instance of the motor class
    mymotortest = RpiMotorLib.A4988Nema(direction, step, GPIO_pins, "DRV8825")

    # Enable the motor
    GPIO.output(Enable_Pin, GPIO.LOW)

    if direction_case is None:
        # Stop the motor by disabling the driver
        GPIO.output(Enable_Pin, GPIO.HIGH)
    else:
        # Move the motor without blocking indefinitely
        mymotortest.motor_go(direction_case, "Full", 1, 0.00045, False, 0)
    
    # Clean up GPIO settings
    GPIO.cleanup()
