# -*- coding: utf-8 -*-

import sys,os
import getopt
import copy
import smbus
import datetime
import subprocess
import requests

import httplib

from time import time,sleep

import numpy as np

from keras.models import model_from_json


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

proc = None
starttime = None

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
    norm = int(np.linalg.norm([acc_x,acc_y,acc_z]))
    print "accel x = %d, y = %d z = %d norm = %d" % (acc_x, acc_y, acc_z, norm)
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


def get_model():
    argvs = sys.argv
    if len(argvs)!=2:
        print "need one arg"
        sys.exit()

    filepath = argvs[1]

    model = model_from_json(open(filepath).read())
    model.load_weights(filepath.replace('.json','.hdf5'))

    model.summary()

    return model

def max_in_batch(result):
    result = np.sum(result, axis=0)
    return np.argmax(result)

def norm_to_strength(norm):
    global proc

    norm = int(norm) - 8800

    if norm < 0:
        norm = 0

    if norm in range(0,1000):
        return 1
    elif norm in range(1000,4000):
        return 2
    elif norm in range(4000,11200):
        return 3
    else:
        return 4

def play_sound(label,strength):
    global proc
    global starttime

    timefrag = False
    nowtime = time()
    if starttime is None or nowtime - starttime > 5.0:
        timefrag = True
        starttime = nowtime

    if strength == 3:
        cmd = "aplay -D hw:1,0 /home/pi/program/auch03_alerm_mo.wav"
    elif strength == 4:
        cmd = "aplay -D hw:1,0 /home/pi/program/auch04_alerm_mo.wav"
    elif timefrag and label == 1:
        cmd = "aplay -D hw:1,0 /home/pi/program/lying_alerm_mo.wav"
    elif timefrag and label == 2:
        cmd = "aplay -D hw:1,0 /home/pi/program/lifting_alerm_mo.wav"
    elif proc is not None and proc.poll() is None and label==0:
        proc.terminate()
    else:
        return

    if proc is None or proc.poll() is not None:
        proc = subprocess.Popen(cmd.strip().split(" "))
    return

def req_http_get(label,strength):
    try:
        url = "http://localhost:8888/{}_{}".format(label,strength)
        #print url
        r = requests.get(url)
        #print r.status_code
    except:
        print "server is down"


def start_predict(model):  
    
    buf = []
    buf_norm = []
    acc_batch = []
    norm_batch = []
    BATCH_SIZE = 50
    prev_label = None
    prev_strength = None

    timestep = model.get_layer(index=0).input.get_shape()[1]
    #start predict
    print 'start prediction ----->'
    print timestep

    try:
        while True:
            acc_data = get_accel()
            if acc_data is not None:
                buf.append(acc_data)
                buf_norm.append(np.linalg.norm(acc_data))  
                if len(buf) == int(timestep):
                    tmp = copy.deepcopy(buf)
                    acc_batch.append(tmp)
                    norm_batch.append(max(buf_norm))
                    del(buf[0])
                    del(buf_norm[0])
                
                if len(acc_batch) == BATCH_SIZE:
                    tmp =np.array(acc_batch)
                    tmp = tmp.reshape(len(acc_batch), timestep, len(acc_data))
                    result = model.predict(tmp)
                    label = max_in_batch(result)
                    if label==3:
                        strength = norm_to_strength(max(norm_batch))
                    else:
                        strength = 0
                    play_sound(label, strength) 
                    if prev_label!=label or prev_strength!=strength:
                        if label == 3:
                            label = prev_label
                        if label!=prev_label:
                            if label==0:
                                print 'nothing'
                            elif label==1:
                                print 'lying'
                            elif label==2:
                                print 'lifting'
                            elif label==3:
                                print 'attack'
                        prev_label = label
                        req_http_get(label,strength)
                        #play_sound(label,strength)
                        prev_strength = strength
                    acc_batch = []
                    norm_batch = []
            #gyr_data = get_gyro()
            #if gyr_data is not None:
            #   
            #break
            

    except KeyboardInterrupt:
        print  "KeyboardInterrupt\n"


if __name__ == "__main__":
    enable_accel()
    enable_gyro()
    fast_offset_compensation()
    start_predict(get_model())



