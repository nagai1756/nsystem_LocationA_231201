from time import sleep
import Jetson.GPIO as GPIO
import smbus

class sht31_temp_humi():
    def __init__(self):
        ##SHT31設定部
        # i2cのアドレス
        i2c_addr = 0x45
        # SMBusモジュールの設定
        bus = smbus.SMBus(1)

        # i2c通信の設定
        bus.write_byte_data(i2c_addr, 0x21, 0x30)    
        sleep(1)
        self.i2c_addr = i2c_addr
        self.bus = bus
        print("SHT31_check_OK")

    # SHT31(温湿度センサ)の測定
    def SHT31(self):
        self.bus.write_byte_data(self.i2c_addr, 0xE0, 0x00)
        data = self.bus.read_i2c_block_data(self.i2c_addr, 0x00, 6)

        # 温度計算
        temp_mlsb = ((data[0] << 8) | data[1])
        temp = -45 + 175 * int(str(temp_mlsb), 10) / (pow(2, 16) - 1)

        # 湿度計算
        humi_mlsb = ((data[3] << 8) | data[4])
        humi = 100 * int(str(humi_mlsb), 10) / (pow(2, 16) - 1)
        return [temp, humi]

