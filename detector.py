import serial
ser = serial.Serial('/dev/tty.usbmodem14101',baudrate=57600,timeout=5)
#ser = serial.Serial('COM6',baudrate=57600,timeout=5)
ser.flushInput()

while True:
    try:
        ser_bytes = ser.readline()
        #print(ser_bytes)
        decoded_bytes = ser_bytes[0:len(ser_bytes)-2].decode("utf-8")
        if not "ERROR" in decoded_bytes:
            print(decoded_bytes)
    except:
        print("Keyboard Interrupt")
        print(ser.readline())
        break
