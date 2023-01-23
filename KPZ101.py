# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:39:33 2023

@author: Owen
"""

"""
kpz101_pythonnet
==================
An example of using the .NET API with the pythonnet package for controlling a KPZ101
"""
import os
import time
import sys
import clr

clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.KCube.PiezoCLI.dll")
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.PiezoCLI import *
from System import Decimal  # necessary for real world units

def Initialise(snum): # type(snum)=str

        DeviceManagerCLI.BuildDeviceList()
    
        # create new device  
        # Connect, begin polling, and enable
        device = KCubePiezo.CreateKCubePiezo(snum)
        if not device == None:
            device.Connect(snum)
            #print(device.IsSettingsInitialized())
            if not device.IsSettingsInitialized():
                device.WaitForSettingsInitialized(1000) #initilise
        
        # Get Device Information and display description
        #device_info = device.GetDeviceInfo()
        #print(device_info.Description)
    
        # Start polling and enable
        device.StartPolling(100)  #100ms polling rate
        time.sleep(1)
        device.EnableDevice()
        time.sleep(0.25)  # Wait for device to enable
    
        # Load the device configuration
        device_config = device.GetPiezoConfiguration(snum)
    
        # This shows how to obtain the device settings
        device_settings = device.PiezoDeviceSettings
    
        # Set the Zero point of the device
        #print("Setting Zero Point")
        device.SetZero()
        print('device',snum,'initialised')
        
        return device
    
def Set_V(device,V):
    # Get the maximum voltage output of the KPZ
    #max_voltage = device.GetMaxOutputVoltage()  # This is stored as a .NET decimal
    
    #device.SetZero()
    # Go to a voltage
    dev_voltage = Decimal(V)
    print(f'Going to voltage {dev_voltage}')

    device.SetOutputVoltage(dev_voltage)
    time.sleep(0.2)
    print(f'Moved to Voltage {device.GetOutputVoltage()}')
    


def Kill(device):
    device.StopPolling()
    device.Disconnect()
    print('device killed')
    

    
    