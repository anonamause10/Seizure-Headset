import bluetooth
from bluetooth.btcommon import BluetoothError
import json
import time

def connect_bluetooth_addr(addr):
    for i in range(5):
        if i > 0:
            time.sleep(1)
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        try:
            sock.connect((addr, 7))
            sock.setblocking(False)
            return sock
        except BluetoothError:
            print("e")
    return None

def connect_magic():
    """ Tries to connect to the first MindWave Mobile it can find.
        If this computer hasn't connected to the headset before, you may need
        to make it visible to this computer's bluetooth adapter. This is done
        by pushing the switch on the left side of the headset to the "on/pair"
        position until the blinking rythm switches.

        The address is then put in a file for later reference.

    """
    nearby_devices = bluetooth.discover_devices(lookup_names = True, duration=5)

    for addr, name in nearby_devices:
        if name == "Force Trainer II":
            print ("found")
            return (connect_bluetooth_addr(addr), addr)
    return (None, "")
