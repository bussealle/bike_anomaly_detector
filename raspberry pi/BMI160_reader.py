# -*- coding: utf-8 -*-


import smbus
import sys, getopt
import datetime
import os
import csv
from time import time,sleep


bus = smbus.SMBus(1)

BMI160_DEVICE_ADDRESS = 0x69

BMI160_REGA_USR_CHIP_ID      = 0x00
BMI160_REGA_USR_ACC_CONF_ADDR     = 0x40
BMI160_REGA_USR_ACC_RANGE_ADDR    = 0x41
BMI160_REGA_USR_GYR_CONF_ADDR     = 0x42
BMI160_REGA_USR_GYR_RANGE_ADDR    = 0x43
BMI160_REGA_USR_FOC_CONF_ADDR = 0x69

BMI160_REGA_CMD_CMD_ADDR          =   0x7e
BMI160_REGA_CMD_EXT_MODE_ADDR     =   0x7f

CMD_SOFT_RESET_REG      = 0xb6

CMD_START_FOC = 0x03

CMD_PMU_ACC_SUSPEND     = 0x10
CMD_PMU_ACC_NORMAL      = 0x11
CMD_PMU_ACC_LP1         = 0x12
CMD_PMU_ACC_LP2         = 0x13
CMD_PMU_GYRO_SUSPEND    = 0x14
CMD_PMU_GYRO_NORMAL     = 0x15
CMD_PMU_GYRO_FASTSTART  = 0x17

BMI160_USER_DATA_14_ADDR = 0X12 # accel x
BMI160_USER_DATA_15_ADDR = 0X13 # accel x
BMI160_USER_DATA_16_ADDR = 0X14 # accel y
BMI160_USER_DATA_17_ADDR = 0X15 # accel y
BMI160_USER_DATA_18_ADDR = 0X16 # accel z
BMI160_USER_DATA_19_ADDR = 0X17 # accel z

BMI160_USER_DATA_8_ADDR  = 0X0C 
BMI160_USER_DATA_9_ADDR  = 0X0D
BMI160_USER_DATA_10_ADDR = 0X0E
BMI160_USER_DATA_11_ADDR = 0X0F
BMI160_USER_DATA_12_ADDR = 0X10
BMI160_USER_DATA_13_ADDR = 0X11

BMI160_REGA_STATUS = 0x1B

ACC_SENS = 8192.0 # 4g Typ
GYR_SENS = 32.8 # 1000rad/s Typ

ACC_READY = 128
GYR_READY = 64
FOC_COMPLETED = 8

#initialize
# ACC sample rate 25/2 times in Hz
#bus.write_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_USR_ACC_CONF_ADDR, 0x25)

# ACC sample rate 100 times in Hz
bus.write_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_USR_ACC_CONF_ADDR, 0x28)
# ACC g-range +-4g
bus.write_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_USR_ACC_RANGE_ADDR, 0x5)
# GYR sample rate 25 times in Hz
#bus.write_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_USR_GYR_CONF_ADDR, 0x26)

# GYR sample rate 100 times in Hz
bus.write_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_USR_GYR_CONF_ADDR, 0x28)
# GYR scale resolution +-1000 rad/s
bus.write_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_USR_GYR_RANGE_ADDR, 0x1)
# FOC gyro enabled accel set to [0g,0g,1g]
bus.write_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_USR_FOC_CONF_ADDR, 0x7E)


#command register
#bus.write_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_CMD_CMD_ADDR, CMD_SOFT_RESET_REG)


acc_offset_val = [0, 0, 0]
gyr_offset_val = [0 ,0, 0]


def fast_offset_compensation() :
    sleep(0.1)
    #start FOC
    print "--- Start FOC ---"
    bus.write_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_CMD_CMD_ADDR, CMD_START_FOC)
    sleep(0.1)
    #wait completion
    start_time = time()
    while True:
        sleep(0.05)
        status = bus.read_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_STATUS)
        if FOC_COMPLETED == (status & pow(2,3)):
            print "--- DONE ---"
            break
        passed_time = time() - start_time
        if passed_time > 0.5:
            print "<!-- FOC FAILED -->"
            break



def enable_accel( ) :
    acc_value = [ 0, 0, 0, 0, 0, 0]
    #op_mode set to 0 and go to normal mode
    sleep(0.1)
    bus.write_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_CMD_CMD_ADDR, CMD_PMU_ACC_NORMAL)
    sleep(0.1)
    
    #read acc xyz
    acc_value = bus.read_i2c_block_data(BMI160_DEVICE_ADDRESS, BMI160_USER_DATA_14_ADDR, 6)

    acc_x =  sign_conv((acc_value[1] << 8) | acc_value[0])
    acc_y =  sign_conv((acc_value[3] << 8) | acc_value[2])
    acc_z =  sign_conv((acc_value[5] << 8) | acc_value[4])


    #Need to be remap according to 1 pin postion
    acc_offset_val[0] = acc_x
    acc_offset_val[1] = acc_y
    acc_offset_val[2] = acc_z
    print "accel x = %d, y = %d z = %d" % (acc_x, acc_y, acc_z)
    return;


def enable_gyro( ) :
    gyro_value = [ 0, 0, 0, 0, 0, 0]
    #op_mode set to 0 and go to normal mode
    sleep(0.1)
    bus.write_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_CMD_CMD_ADDR, CMD_PMU_GYRO_NORMAL)
    sleep(0.1)
    
    #read gyro xyz
    gyro_value = bus.read_i2c_block_data(BMI160_DEVICE_ADDRESS, BMI160_USER_DATA_8_ADDR, 6)

    gyro_x =  sign_conv((gyro_value[1] << 8) | gyro_value[0])
    gyro_y =  sign_conv((gyro_value[3] << 8) | gyro_value[2])
    gyro_z =  sign_conv((gyro_value[5] << 8) | gyro_value[4])
        

    #Need to be remap according to 1 pin postion
    print "gyro x = %d, y = %d z = %d" % (gyro_x, gyro_y, gyro_z)
    return;

def sign_conv(binary):
    return -(binary & 0b1000000000000000) | (binary & 0b0111111111111111)

def get_accel():
    
    status = bus.read_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_STATUS)
    res = None
    if ACC_READY == (status & pow(2,7)):
        acc_value = bus.read_i2c_block_data(BMI160_DEVICE_ADDRESS, BMI160_USER_DATA_14_ADDR, 6)
        acc_x =  sign_conv((acc_value[1] << 8) | acc_value[0])
        acc_y =  sign_conv((acc_value[3] << 8) | acc_value[2])
        acc_z =  sign_conv((acc_value[5] << 8) | acc_value[4])
        res = [acc_x, acc_y, acc_z]
    
    return res;

def get_gyro():

    status = bus.read_byte_data(BMI160_DEVICE_ADDRESS, BMI160_REGA_STATUS)
    res = None
    if GYR_READY == (status & pow(2,6)):
        gyro_value = bus.read_i2c_block_data(BMI160_DEVICE_ADDRESS, BMI160_USER_DATA_8_ADDR, 6)
        gyro_x =  sign_conv((gyro_value[1] << 8) | gyro_value[0])
        gyro_y =  sign_conv((gyro_value[3] << 8) | gyro_value[2])
        gyro_z =  sign_conv((gyro_value[5] << 8) | gyro_value[4])
        res = [gyro_x,gyro_y,gyro_z]

    return res;

def get_timestamp():
    return datetime.datetime.now().strftime("%M:%S.%f")


def start_logging():
    #input tag
    print "input log name"
    tag_name = raw_input('>> ')
    #label
    print "input default label"
    def_label = raw_input('>> ')


    # prepare csv
    now = datetime.datetime.now()
    f_acc_name = "accel_{0:%Y%m%d-%H%M%S}_".format(now)
    f_gyr_name = "gyro_{0:%Y%m%d-%H%M%S}_".format(now)
    f_acc_name = f_acc_name + tag_name + ".csv"
    f_gyr_name = f_gyr_name + tag_name + ".csv"
    print "start-logging: "+f_acc_name
    print "start-logging: "+f_gyr_name
    dir_name = "./log_seq"
    if not(os.path.exists(dir_name)):
        os.mkdir(dir_name)

    acc_file = csv.writer(open("./log_seq/{}".format(f_acc_name), 'ab'))
    gyr_file = csv.writer(open("./log_seq/{}".format(f_gyr_name), 'ab'))
    
    acc_file.writerow(["Time","Acc_x","Acc_y","Acc_z","Label"])
    gyr_file.writerow(["Time","Gyr_x","Gyr_y","Gyr_z","Label"])
    
    #start logging
    try:
        while True:
            #pass
            acc_data = get_accel()
            gyr_data = get_gyro()
            if acc_data is not None:
                acc_data.insert(0,get_timestamp())
                acc_data.extend(def_label)
                acc_file.writerow(map(str, acc_data))
            if gyr_data is not None:
                gyr_data.insert(0,get_timestamp())
                gyr_data.extend(def_label)
                gyr_file.writerow(map(str, gyr_data))
            #break
    except KeyboardInterrupt:
        print  "KeyboardInterrupt\n"



enable_accel()
enable_gyro()
fast_offset_compensation()
start_logging()
sys.exit()
