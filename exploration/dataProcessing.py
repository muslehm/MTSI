from __future__ import print_function

import json
import os
import base64
import io

# Loading DataFrame libraries
import numpy as np
import pandas as pd
import time

# Loading DTW library
from dtaidistance import dtw


# Cycle Object
class Cycle:
    def __init__(self, cycle):
        self.cycleNumber = cycle['SerialCycleNumber']
        self.timeCreated = cycle['TimeStampCreated']
        self.date = cycle['Date']
        self.timeStampSaved = cycle['TimeStampSaved']

        # read Environment Data
        environment = cycle['Environment']
        self.newMaterialAirTemperature = environment['ActNewMaterialAirTemperature']
        self.newMaterialAirHumidity = environment['ActNewMaterialAirHumidity']
        self.regrindMaterialAirTemperature = environment['ActRegrindMaterialAirTemperature']
        self.regrindMaterialAirHumidity = environment['ActRegrindMaterialAirHumidity']


        # read material data
        material = cycle['Material']
        self.materialName = material['MaterialName']
        self.batchId = material['BatchId']
        self.materialConsumptionFeedQuantity = material['ActMaterialConsumptionFeedQuantity']
        self.newMaterialFeedQuantity = material['ActNewMaterialFeedQuantity']
        self.newMaterialFeedPercentage = material['ActNewMaterialFeedPercentage']
        self.masterbatchFeedQuantity = material['ActMasterbatchFeedQuantity']
        self.masterbatchFeedPercentage = material['ActMasterbatchFeedPercentage']
        self.regrindMaterialFeedQuantity = material['ActRegrindMaterialFeedQuantity']
        self.regrindMaterialFeedPercentage = material['ActRegrindMaterialFeedPercentage']

        # read mold data
        mold = cycle['Mold']
        self.moldNumber = mold['MoldNumber']
        self.movableMoldTemperature = mold['MovableMoldTemperature']
        self.fixedMoldTemperature = mold['FixedMoldTemperature']

        # read machine data
        machine = cycle['Machine']
        self.machineNumber = machine['MachineNumber']
        self.mashineState = machine['MashineState']
        self.autoStep = machine['AutoStep']
        self.hydraulicTemperature = machine['ActHydraulicTemperature']

        self.temperatureFeedZone = machine['ActTemperatureFeedZone']
        self.setTemperatureFeedZone = machine['SetTemperatureFeedZone']

        self.temperatureZoneA1 = machine['ActTemperatureZoneA1']
        self.setTemperatureZoneA1 = machine['SetTemperatureZoneA1']

        self.temperatureZoneA2 = machine['ActTemperatureZoneA2']
        self.setTemperatureZoneA2 = machine['SetTemperatureZoneA2']

        self.temperatureZoneA3 = machine['ActTemperatureZoneA3']
        self.setTemperatureZoneA3 = machine['SetTemperatureZoneA3']

        self.temperatureZoneA10 = machine['ActTemperatureZoneA10']
        self.setTemperatureZoneA10 = machine['SetTemperatureZoneA10']

        self.temperatureZoneA11 = machine['ActTemperatureZoneA11']
        self.setTemperatureZoneA11 = machine['SetTemperatureZoneA11']

        self.headTemperatureZone1 = machine['ActHeadTemperatureZone1']
        self.setHeadTemperatureZone1 = machine['SetHeadTemperatureZone1']

        self.headTemperatureZone2 = machine['ActHeadTemperatureZone2']
        self.setHeadTemperatureZone2 = machine['SetHeadTemperatureZone2']

        self.headTemperatureZone3 = machine['ActHeadTemperatureZone3']
        self.setHeadTemperatureZone3 = machine['SetHeadTemperatureZone3']

        self.dieTemperatureZone1 = machine['ActDieTemperatureZone1']
        self.setDieTemperatureZone1 = machine['SetDieTemperatureZone1']

        self.dieTemperatureZone2 = machine['ActDieTemperatureZone2']
        self.setDieTemperatureZone2 = machine['SetDieTemperatureZone2']

        self.dieTemperatureZone3 = machine['ActDieTemperatureZone3']
        self.setDieTemperatureZone3 = machine['SetDieTemperatureZone3']

        self.ActExtruderTorqueList = machine['ActExtruderTorqueList']
        self.ActExtruderSpeedList = machine['ActExtruderSpeedList']
        self.ActExtruderPressureList = machine['ActExtruderPressureList']
        self.ActMeltTemperatureList = machine['ActMeltTemperatureList']

        self.ventingTime = machine['ActVentingTime']
        self.setVentingTime = machine['SetVentingTime']

        self.pushOutProfile = machine['ActPushOutProfileList']
        self.setPushOutProfile = machine['SetPushOutProfile']

        self.WTCProfile = machine['ActWTCProfileList']
        self.setWTCProfile = machine['SetWTCProfile']

        self.RWTC1Profile = machine['ActRWTC1ProfileList']
        self.setRWTC1Profile = machine['SetRWTC1Profile']

        self.RWTC2Profile = machine['ActRWTC2ProfileList']
        self.setRWTC2Profile = machine['SetRWTC2Profile']

        self.suctionProfile = machine['ActSuctionProfileList']
        self.setSuctionProfile = machine['SetSuctionProfile']

        self.supportAirProfile = machine['ActSupportAirProfile']
        self.setSupportAirProfile = machine['SetSupportAirProfile']

        self.setSuctionAirBasicSpeed = machine['SetSuctionAirBasicSpeed']
        self.setSuctionAirWorkSpeed = machine['SetSuctionAirWorkSpeed']
        self.setDelayBlowingTime = machine['SetDelayBlowingTime']
        self.setBlowingTime = machine['SetBlowingTime']
        self.setSupportAirOverProfile = machine['SetSupportAirOverProfile']



        # read BottomFlashLength
        self.BottomFlashLenght = cycle['BottomFlashLenght']


    def fixList(self, theList):
        ams = theList.values()
        for i in ams:
            i = i[1:-1]
            trimList = i
        splitList = trimList.split(', ')
        splitList = [float(i) for i in splitList]
        cleanList = list(filter(lambda a: a != -1001.0, splitList))
        return cleanList


#Calculate Distances to Simplify Time Series Data
def calc_distance(actual, cycle_settings):
    x1 = np.array(actual)
    y1 = x1.astype(np.float)
    x2 = np.array(cycle_settings)
    y2 = x2.astype(np.float)
    distance = dtw.distance_fast(y1, y2)
    return distance


# Calculate Average
def calc_average(column):
    test_sum = []
    ss = 0
    for col in column:
        test_sum.append(len(col))
        ss += 1
    ff = int(sum(test_sum) / ss)

    ex_default = []
    for x in [*range(0, ff)]:
        val_av = []
        for col in column:
            try:
                val_av.append(col[x])
            except:
                pass
        ex_default.append(sum(val_av) / len(val_av))
    return ex_default


# Calculate Average for previous three
def calc_average_three(m, column):
    if m > 2:
        ff = int((len(column[m - 1]) + len(column[m - 2]) + len(column[m - 3]))/3)
        ex_default = []
        column3 = column[(m - 3):m]
        for x in [*range(0, ff)]:
            val_av = []
            for col in column3:
                try:
                    val_av.append(col[x])
                except:
                    pass
            ex_default.append(sum(val_av) / len(val_av))
    elif m == 2:
        ff = int((len(column[m - 1]) + len(column[m - 2]))/ 2)
        ex_default = []
        column2 = column[(m - 2):m]
        for x in [*range(0, ff)]:
            val_av = []
            for col in column2:
                try:
                    val_av.append(col[x])
                except:
                    pass
            ex_default.append(sum(val_av) / len(val_av))
    elif m == 1:
        ex_default = column[m-1]
    else:
        ex_default = column[m]
    return ex_default


# Change to Product data
def transform_product(dfProduct):
    new_df = pd.DataFrame(columns = dfProduct.columns)
    for x in [*range(0, len(dfProduct)-2)]:
        df_temp = dfProduct[x:x+3].copy()
        m=0
        temp_dict = {}
        temp_dict['startTime'] = df_temp['startTime'][0]
        temp_dict['endTime'] = df_temp['endTime'][2]
        temp_dict['totalTime'] = df_temp['endTime'][2] - df_temp['startTime'][0]
        temp_dict['ejectTime'] = df_temp['ejectTime'][1]
        for col in df_temp.columns:
            if m > 3:
                y = df_temp[col][0].copy()
                y.extend(df_temp[col][1])
                y.extend(df_temp[col][2])
                temp_dict[col] = y
            else:
                pass
            m+=1
        if x<10:
            the_index = 'C0000' + str(x)
        elif x > 9 and x < 100:
            the_index = 'C000' + str(x)
        elif x > 99 and x < 1000:
            the_index = 'C00' + str(x)
        elif x > 999 and x < 10000:
            the_index = 'C0' + str(x)
        new_df.loc[the_index] = temp_dict
    return new_df


def transform_product_selective(dfProduct):
    new_df = pd.DataFrame(columns = dfProduct.columns)
    for x in [*range(0, len(dfProduct)-2)]:
        df_temp = dfProduct[x:x+3].copy()
        m=0
        temp_dict = {}
        temp_dict['startTime'] = df_temp['startTime'][0]
        temp_dict['endTime'] = df_temp['endTime'][2]
        temp_dict['totalTime'] = df_temp['endTime'][2] - df_temp['startTime'][0]
        temp_dict['ejectTime'] = df_temp['ejectTime'][1]
        for col in df_temp.columns:
            if m > 13:
                y = df_temp[col][0].copy()
                temp_dict[col] = y
            elif m < 14 and m > 3:
                y = df_temp[col][1].copy()
                temp_dict[col] = y
            else:
                pass
            m+=1
        if x < 10:
            the_index = 'C0000' + str(x)
        elif x > 9 and x < 100:
            the_index = 'C000' + str(x)
        elif x > 99 and x < 1000:
            the_index = 'C00' + str(x)
        elif x > 999 and x < 10000:
            the_index = 'C0' + str(x)
        new_df.loc[the_index] = temp_dict
    return  new_df


# Create DataFrame
def create_dataframe_previous_three(df_cycle_main):
    start = time.time()

    m = 0
    ex_av3 = []
    for i in df_cycle_main['exPressure']:
        ex_default = calc_average_three(m, df_cycle_main['exPressure'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['exPressure'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['exSpeed']:
        ex_default = calc_average_three(m, df_cycle_main['exSpeed'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['exSpeed'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['exTorque']:
        ex_default = calc_average_three(m, df_cycle_main['exTorque'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['exTorque'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['RWTC1']:
        ex_default = calc_average_three(m, df_cycle_main['RWTC1'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['RWTC1'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['RWTC2']:
        ex_default = calc_average_three(m, df_cycle_main['RWTC2'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['RWTC2'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['suction']:
        ex_default = calc_average_three(m, df_cycle_main['suction'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['suction'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['WTC']:
        ex_default = calc_average_three(m, df_cycle_main['WTC'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['WTC'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['dieTempZ1']:
        ex_default = calc_average_three(m, df_cycle_main['dieTempZ1'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['dieTempZ1'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['dieTempZ2']:
        ex_default = calc_average_three(m, df_cycle_main['dieTempZ2'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['dieTempZ2'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['dieTempZ3']:
        ex_default = calc_average_three(m, df_cycle_main['dieTempZ3'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['dieTempZ3'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['headTempZ1']:
        ex_default = calc_average_three(m, df_cycle_main['headTempZ1'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['headTempZ1'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['headTempZ2']:
        ex_default = calc_average_three(m, df_cycle_main['headTempZ2'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['headTempZ2'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['headTempZ3']:
        ex_default = calc_average_three(m, df_cycle_main['headTempZ3'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['headTempZ3'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['tempFZ']:
        ex_default = calc_average_three(m, df_cycle_main['tempFZ'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['tempFZ'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['tempZA1']:
        ex_default = calc_average_three(m, df_cycle_main['tempZA1'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['tempZA1'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['tempZA10']:
        ex_default = calc_average_three(m, df_cycle_main['tempZA10'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['tempZA10'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['tempZA11']:
        ex_default = calc_average_three(m, df_cycle_main['tempZA11'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['tempZA11'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['tempZA2']:
        ex_default = calc_average_three(m, df_cycle_main['tempZA2'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['tempZA2'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['tempZA3']:
        ex_default = calc_average_three(m, df_cycle_main['tempZA3'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['tempZA3'][m] = i
        m += 1

    m = 0
    ex_av3 = []
    for i in df_cycle_main['meltTemp']:
        ex_default = calc_average_three(m, df_cycle_main['meltTemp'])
        ex_av3.append(calc_distance(i, ex_default))
        m += 1

    m = 0
    for i in ex_av3:
        df_cycle_main['meltTemp'][m] = i
        m += 1

    ex_default = ''
    end = time.time()
    print("Calculate Distances")
    print(end - start)
    return df_cycle_main


# Create DataFrame average
def create_dataframe_av(df_cycle_main):
    start = time.time()
    ex_default = calc_average(df_cycle_main['exPressure'])
    m = 0
    for i in df_cycle_main['exPressure']:
        df_cycle_main['exPressure'][m] = calc_distance(i, ex_default)
        m += 1

    ex_default = calc_average(df_cycle_main['exSpeed'])
    m = 0
    for i in df_cycle_main['exSpeed']:
        df_cycle_main['exSpeed'][m] = calc_distance(i, ex_default)
        m += 1

    ex_default = calc_average(df_cycle_main['exTorque'])
    m = 0
    for i in df_cycle_main['exTorque']:
        df_cycle_main['exTorque'][m] = calc_distance(i, ex_default)
        m += 1

    ex_default = calc_average(df_cycle_main['RWTC1'])
    m = 0
    for i in df_cycle_main['RWTC1']:
        df_cycle_main['RWTC1'][m] = calc_distance(i, ex_default)
        m += 1

    ex_default = calc_average(df_cycle_main['RWTC2'])
    m = 0
    for i in df_cycle_main['RWTC2']:
        df_cycle_main['RWTC2'][m] = calc_distance(i, ex_default)
        m += 1

    ex_default = calc_average(df_cycle_main['suction'])
    m = 0
    for i in df_cycle_main['suction']:
        df_cycle_main['suction'][m] = calc_distance(i, ex_default)
        m += 1

    ex_default = calc_average(df_cycle_main['WTC'])
    m = 0
    for i in df_cycle_main['WTC']:
        df_cycle_main['WTC'][m] = calc_distance(i, ex_default)
        m += 1

    ex_default = calc_average(df_cycle_main['dieTempZ1'])
    m = 0
    for i in df_cycle_main['dieTempZ1']:
        df_cycle_main['dieTempZ1'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    ex_default = calc_average(df_cycle_main['dieTempZ2'])
    for i in df_cycle_main['dieTempZ2']:
        df_cycle_main['dieTempZ2'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    ex_default = calc_average(df_cycle_main['dieTempZ3'])
    for i in df_cycle_main['dieTempZ3']:
        df_cycle_main['dieTempZ3'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    ex_default = calc_average(df_cycle_main['headTempZ1'])
    for i in df_cycle_main['headTempZ1']:
        df_cycle_main['headTempZ1'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    ex_default = calc_average(df_cycle_main['headTempZ2'])
    for i in df_cycle_main['headTempZ2']:
        df_cycle_main['headTempZ2'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    ex_default = calc_average(df_cycle_main['headTempZ3'])
    for i in df_cycle_main['headTempZ3']:
        df_cycle_main['headTempZ3'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    ex_default = calc_average(df_cycle_main['tempFZ'])
    for i in df_cycle_main['tempFZ']:
        df_cycle_main['tempFZ'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    ex_default = calc_average(df_cycle_main['tempZA1'])
    for i in df_cycle_main['tempZA1']:
        df_cycle_main['tempZA1'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    ex_default = calc_average(df_cycle_main['tempZA10'])
    for i in df_cycle_main['tempZA10']:
        df_cycle_main['tempZA10'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    ex_default = calc_average(df_cycle_main['tempZA11'])
    for i in df_cycle_main['tempZA11']:
        df_cycle_main['tempZA11'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    ex_default = calc_average(df_cycle_main['tempZA2'])
    for i in df_cycle_main['tempZA2']:
        df_cycle_main['tempZA2'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    ex_default = calc_average(df_cycle_main['tempZA3'])
    for i in df_cycle_main['tempZA3']:
        df_cycle_main['tempZA3'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    ex_default = calc_average(df_cycle_main['meltTemp'])
    for i in df_cycle_main['meltTemp']:
        df_cycle_main['meltTemp'][m] = calc_distance(i, ex_default)
        m += 1
    ex_default = ''
    end = time.time()
    print("Calculate Distances")
    print(end - start)
    return df_cycle_main

# Create DataFrame previous datapoint
def create_dataframe(df_cycle_main):
    start = time.time()

    m = 0
    extr = []
    for i in df_cycle_main['exPressure']:
        if len(i) > 0:
            if m != 0:
                settings_ = extr
                extr = i
                df_cycle_main['exPressure'][m] = calc_distance(i, settings_)
            else:
                settings_ = i
                extr = i
                df_cycle_main['exPressure'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    extr = []
    for i in df_cycle_main['exSpeed']:
        if len(i) > 0:
            if m != 0:
                settings_ = extr
                extr = i
                df_cycle_main['exSpeed'][m] = calc_distance(i, settings_)
            else:
                settings_ = i
                extr = i
                df_cycle_main['exSpeed'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    extr = []
    for i in df_cycle_main['exTorque']:
        if len(i) > 0:
            if m != 0:
                settings_ = extr
                extr = i
                df_cycle_main['exTorque'][m] = calc_distance(i, settings_)
            else:
                settings_ = i
                extr = i
                df_cycle_main['exTorque'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    wtc_wtc = []
    for i in df_cycle_main['RWTC1']:
        if m != 0:
            settings_ = wtc_wtc
            wtc_wtc = i
            df_cycle_main['RWTC1'][m] = calc_distance(i, settings_)
        else:
            #settings_ = dfSetting['setRWTC1'][m]
            settings_ = i
            wtc_wtc = i
            df_cycle_main['RWTC1'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    wtc_wtc = []
    for i in df_cycle_main['RWTC2']:
        if m != 0:
            settings_ = wtc_wtc
            wtc_wtc = i
            df_cycle_main['RWTC2'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            wtc_wtc = i
            df_cycle_main['RWTC2'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    wtc_wtc = []
    for i in df_cycle_main['suction']:
        if m != 0:
            settings_ = wtc_wtc
            wtc_wtc = i
            df_cycle_main['suction'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            wtc_wtc = i
            df_cycle_main['suction'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    wtc_wtc = []
    for i in df_cycle_main['WTC']:
        if m != 0:
            settings_ = wtc_wtc
            wtc_wtc = i
            df_cycle_main['WTC'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            wtc_wtc = i
            df_cycle_main['WTC'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['dieTempZ1']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['dieTempZ1'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            temp_temp = i
            df_cycle_main['dieTempZ1'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['dieTempZ2']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['dieTempZ2'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            temp_temp = i
            df_cycle_main['dieTempZ2'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['dieTempZ3']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['dieTempZ3'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            temp_temp = i
            df_cycle_main['dieTempZ3'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['headTempZ1']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['headTempZ1'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            temp_temp = i
            df_cycle_main['headTempZ1'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['headTempZ2']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['headTempZ2'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            temp_temp = i
            df_cycle_main['headTempZ2'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['headTempZ3']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['headTempZ3'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            temp_temp = i
            df_cycle_main['headTempZ3'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['tempFZ']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['tempFZ'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            temp_temp = i
            df_cycle_main['tempFZ'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['tempZA1']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['tempZA1'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            temp_temp = i
            df_cycle_main['tempZA1'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['tempZA10']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['tempZA10'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            temp_temp = i
            df_cycle_main['tempZA10'][m] = calc_distance(i, settings_)

        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['tempZA11']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['tempZA11'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            temp_temp = i
            df_cycle_main['tempZA11'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['tempZA2']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['tempZA2'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            temp_temp = i
            df_cycle_main['tempZA2'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['tempZA3']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['tempZA3'][m] = calc_distance(i, settings_)
        else:
            settings_ = i
            temp_temp = i
            df_cycle_main['tempZA3'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    temp_temp = []
    for i in df_cycle_main['meltTemp']:
        if m != 0:
            settings_ = temp_temp
            temp_temp = i
            df_cycle_main['meltTemp'][m] = calc_distance(i, settings_)
        else:
            #settings_ = [dfSetting['setMeltTemp'][m]]*len(i)
            settings_ = i
            temp_temp = i
            df_cycle_main['meltTemp'][m] = calc_distance(i, settings_)
        m += 1
    settings_ = ''
    temp_temp = ''
    end = time.time()
    print("Calculate Distances")
    print(end - start)
    return df_cycle_main

# Create DataFrame to settings
def create_dataframe_settings(df_cycle_main, dfSetting):
    start = time.time()
    ex_default = calc_average(df_cycle_main['exPressure'])
    m = 0
    for i in df_cycle_main['exPressure']:
        df_cycle_main['exPressure'][m] = calc_distance(i, ex_default)
        m += 1

    ex_default = calc_average(df_cycle_main['exSpeed'])
    m = 0
    for i in df_cycle_main['exSpeed']:
        df_cycle_main['exSpeed'][m] = calc_distance(i, ex_default)
        m += 1

    ex_default = calc_average(df_cycle_main['exTorque'])
    m = 0
    for i in df_cycle_main['exTorque']:
        df_cycle_main['exTorque'][m] = calc_distance(i, ex_default)
        m += 1

    m = 0
    for i in df_cycle_main['RWTC1']:
        df_cycle_main['RWTC1'][m] = calc_distance(i, dfSetting['setRWTC1'][m])
        m += 1

    m = 0
    for i in df_cycle_main['RWTC2']:
        df_cycle_main['RWTC2'][m] = calc_distance(i, dfSetting['setRWTC2'][m])
        m += 1

    m = 0
    for i in df_cycle_main['suction']:
        df_cycle_main['suction'][m] = calc_distance(i, dfSetting['setSuction'][m])
        m += 1

    m = 0
    for i in df_cycle_main['WTC']:
        df_cycle_main['WTC'][m] = calc_distance(i, dfSetting['setWTC'][m])
        m += 1

    m = 0
    for i in df_cycle_main['dieTempZ1']:
        settings_ = [dfSetting['setDieTempZ1'][m]] * len(i)
        df_cycle_main['dieTempZ1'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['dieTempZ2']:
        settings_ = [dfSetting['setDieTempZ2'][m]] * len(i)
        df_cycle_main['dieTempZ2'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['dieTempZ3']:
        settings_ = [dfSetting['setDieTempZ3'][m]] * len(i)
        df_cycle_main['dieTempZ3'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['headTempZ1']:
        settings_ = [dfSetting['setHeadTempZ1'][m]] * len(i)
        df_cycle_main['headTempZ1'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['headTempZ2']:
        settings_ = [dfSetting['setHeadTempZ2'][m]] * len(i)
        df_cycle_main['headTempZ2'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['headTempZ3']:
        settings_ = [dfSetting['setHeadTempZ3'][m]] * len(i)
        df_cycle_main['headTempZ3'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['tempFZ']:
        settings_ = [dfSetting['setTempFZ'][m]] * len(i)
        df_cycle_main['tempFZ'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['tempZA1']:
        settings_ = [dfSetting['setTempZA1'][m]] * len(i)
        df_cycle_main['tempZA1'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['tempZA10']:
        settings_ = [dfSetting['setTempZA10'][m]] * len(i)
        df_cycle_main['tempZA10'][m] = calc_distance(i, settings_)

        m += 1

    m = 0
    for i in df_cycle_main['tempZA11']:
        settings_ = [dfSetting['setTempZA11'][m]] * len(i)
        df_cycle_main['tempZA11'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['tempZA2']:
        settings_ = [dfSetting['setTempZA2'][m]] * len(i)
        df_cycle_main['tempZA2'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['tempZA3']:
        settings_ = [dfSetting['setTempZA3'][m]] * len(i)
        df_cycle_main['tempZA3'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['meltTemp']:
        settings_ = [dfSetting['setMeltTemp'][m]] * len(i)
        df_cycle_main['meltTemp'][m] = calc_distance(i, settings_)
        m += 1
    settings_ = ''
    ex_default = ''
    end = time.time()
    print("Calculate Distances")
    print(end - start)
    return df_cycle_main


# Reading Json File Function
def process_cycles(cycles_, method):
    # Build Vectors
    settings = {}
    cycles = {}
    meta = {}
    y = []
    x = []
    av = 0
    m = 0
    for i in cycles_:
        z = vars(i)
        m = 0
        setting = {}
        cycle = {}
        for j in z:
            m += 1
            if j[0:3] == 'set':
                setting[j] = z[j]
            elif m > 1:
                cycle[j] = z[j]
            else:
                meta[j] = z[j]
        settings['C{}'.format(i.cycleNumber)] = setting
        cycles['C{}'.format(i.cycleNumber)] = cycle
    df_cycle_main = pd.DataFrame(cycles)
    df_cycle_main = df_cycle_main.transpose()
    dfSetting = pd.DataFrame(settings)
    dfSetting = dfSetting.transpose()
    cycles = ''
    meta = ''
    settings = ''
    cycles_ = ''
    dfFull = df_cycle_main.merge(dfSetting, left_index=True, right_index=True)

    if method == 1:
        df_cycle_main = create_dataframe(df_cycle_main)
    elif method == 2:
        df_cycle_main = create_dataframe_av(df_cycle_main)
    elif method == 3:
        df_cycle_main = create_dataframe_previous_three(df_cycle_main)
    elif method == 4:
        df_cycle_main = create_dataframe_settings(df_cycle_main, dfSetting)
    elif method == 5:
        dfProduct = transform_product(df_cycle_main)
        df_cycle_main = create_dataframe_av(dfProduct)
    elif method == 6:
        dfProduct = transform_product_selective(df_cycle_main)
        df_cycle_main = create_dataframe_av(dfProduct)
    dfSetting = ''
    dfProduct = ''
    return df_cycle_main, dfFull


def read_cycles(folder, method):
    start = time.time()
    cycles_ = []
    json_folder_path = os.path.join(folder)
    json_files = [files for files in os.listdir(json_folder_path) if files.endswith("json")]
    json_files.sort()
    for json_file in json_files:
        json_file_path = os.path.join(json_folder_path, json_file)
        with open(json_file_path, "r") as file:
            cycle_file = json.load(file)
            if cycle_file['SerialCycleNumber'] != '0':
                cycle_data = Cycle(cycle_file)
                cycles_.append(cycle_data)
    end = time.time()
    print("Create Cycle objects:")
    print(end - start)
    df_cycle_main, dfFull = process_cycles(cycles_, method)
    cycles_ = ''
    return df_cycle_main, dfFull


def read_cycles_(list_of_contents, list_of_names, list_of_dates, method):
    start = time.time()
    if list_of_contents is not None:
        cycles_ =[]
        for contents, filename, date in zip(list_of_contents, list_of_names, list_of_dates):
            if 'json' in filename:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                data = json.load(io.StringIO(decoded.decode('utf-8')))
                if 151.0 > float(data['CycleTotalTime']) > 49.0:
                    cycle_data = Cycle(data)
                    cycles_.append(cycle_data)
    end = time.time()
    print("Create Cycle objects:")
    print(end - start)
    df_cycle_main, dfFull = process_cycles(cycles_, method)

    list_of_contents = ''
    list_of_names =''
    list_of_dates = ''
    cycles_ = ''
    return df_cycle_main, dfFull

def update_method(df_full, method):
    df_cycle_main = df_full.iloc[:,0:24]
    dfSetting = df_full.iloc[:,24:]
    if method == 1:
        df_cycle_main = create_dataframe(df_cycle_main)
    elif method == 2:
        df_cycle_main = create_dataframe_av(df_cycle_main)
    elif method == 3:
        df_cycle_main = create_dataframe_previous_three(df_cycle_main)
    elif method == 4:
        df_cycle_main = create_dataframe_settings(df_cycle_main, dfSetting)
    elif method == 5:
        dfProduct = transform_product(df_cycle_main)
        df_cycle_main = create_dataframe_av(dfProduct)
    elif method == 6:
        dfProduct = transform_product_selective(df_cycle_main)
        df_cycle_main = create_dataframe_av(dfProduct)
    df_full = ''
    dfSetting = ''
    return df_cycle_main

