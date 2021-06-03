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
        self.cycleNumber = cycle['CycleNumber']
        self.startTime = int(cycle['CycleStartTime'])
        self.endTime = int(cycle['CycleEndTime'])
        self.totalTime = float(cycle['CycleTotalTime'])

        data = cycle['CycleData']

        # Eject Cylinder

        eject_cylinder = data['EjectCylinder']

        if 'SetRWTC1Profile' in eject_cylinder:
            self.setRWTC1 = self.fix_list(eject_cylinder['SetRWTC1Profile'])
        else:
            self.setRWTC1 = 0

        if 'SetRWTC2Profile' in eject_cylinder:
            self.setRWTC2 = self.fix_list(eject_cylinder['SetRWTC2Profile'])
        else:
            self.setRWTC2 = 0

        if 'SetSuctionProfile' in eject_cylinder:
            self.setSuction = self.fix_list(eject_cylinder['SetSuctionProfile'])
        else:
            self.setSuction = 0

        if 'SetWTCProfile' in eject_cylinder:
            self.setWTC = self.fix_list(eject_cylinder['SetWTCProfile'])
        else:
            self.setWTC = 0

        self.ejectTime = float(eject_cylinder['ActEjectTime'])

        if 'ActRWTC1Profile' in eject_cylinder:
            self.RWTC1 = self.fix_list(eject_cylinder['ActRWTC1Profile'])
        else:
            self.RWTC1 = 0

        if 'ActRWTC2Profile' in eject_cylinder:
            self.RWTC2 = self.fix_list(eject_cylinder['ActRWTC2Profile'])
        else:
            self.RWTC2 = 0

        if 'ActSuctionProfile' in eject_cylinder:
            self.suction = self.fix_list(eject_cylinder['ActSuctionProfile'])
        else:
            self.suction = 0.0

        if 'ActWTCProfile' in eject_cylinder:
            self.WTC = self.fix_list(eject_cylinder['ActWTCProfile'])
        else:
            self.WTC = 0.0

        self.dieTempZ1 = []
        if 'ActDieTemperatureProfileZone1' in eject_cylinder:
            die_temperature_profile_zone1 = eject_cylinder['ActDieTemperatureProfileZone1']

            for t in die_temperature_profile_zone1.values():
                self.dieTempZ1.append(float(t))
        else:
            self.dieTempZ1.append(0.0)

        self.dieTempZ2 = []
        if 'ActDieTemperatureProfileZone2' in eject_cylinder:
            die_temperature_profile_zone2 = eject_cylinder['ActDieTemperatureProfileZone2']

            for t in die_temperature_profile_zone2.values():
                self.dieTempZ2.append(float(t))
        else:
            self.dieTempZ2.append(0.0)

        self.dieTempZ3 = []
        if 'ActDieTemperatureProfileZone3' in eject_cylinder:
            die_temperature_profile_zone3 = eject_cylinder['ActDieTemperatureProfileZone3']

            for t in die_temperature_profile_zone3.values():
                self.dieTempZ3.append(float(t))
        else:
            self.dieTempZ3.append(0.0)

        self.headTempZ1 = []
        if 'ActHeadTemperatureProfileZone1' in eject_cylinder:
            head_temperature_profile_zone1 = eject_cylinder['ActHeadTemperatureProfileZone1']

            for t in head_temperature_profile_zone1.values():
                self.headTempZ1.append(float(t))
        else:
            self.headTempZ1.append(0.0)

        self.headTempZ2 = []
        if 'ActHeadTemperatureProfileZone2' in eject_cylinder:
            head_temperature_profile_zone2 = eject_cylinder['ActHeadTemperatureProfileZone2']

            for t in head_temperature_profile_zone2.values():
                self.headTempZ2.append(float(t))
        else:
            self.headTempZ2.append(0.0)

        self.headTempZ3 = []
        if 'ActHeadTemperatureProfileZone3' in eject_cylinder:
            head_temperature_profile_zone3 = eject_cylinder['ActHeadTemperatureProfileZone3']

            for t in head_temperature_profile_zone3.values():
                self.headTempZ3.append(float(t))
        else:
            self.headTempZ3.append(0.0)

        if 'SetDieTemperatureProfileZone1' in eject_cylinder:
            self.setDieTempZ1 = 210
        else:
            self.setDieTempZ1 = 0.0

        if 'SetDieTemperatureProfileZone2' in eject_cylinder:
            self.setDieTempZ2 = 210
        else:
            self.setDieTempZ2 = 0.0

        if 'SetDieTemperatureProfileZone3' in eject_cylinder:
            self.setDieTempZ3 = 215
        else:
            self.setDieTempZ3 = 0.0

        if 'SetHeadTemperatureProfileZone1' in eject_cylinder:
            self.setHeadTempZ1 = 210
        else:
            self.setHeadTempZ1 = 0.0

        if 'SetHeadTemperatureProfileZone2' in eject_cylinder:
            self.setHeadTempZ2 = 210
        else:
            self.setHeadTempZ2 = 0.0

        if 'SetHeadTemperatureProfileZone3' in eject_cylinder:
            self.setHeadTempZ3 = 210
        else:
            self.setHeadTempZ3 = 0.0

        ###############
        del eject_cylinder

        # Extruder Data
        extruder = data['Extruder']

        self.meltTemp = []
        if 'ActMeltTemperatureProfile' in extruder:
            act_melt_temperature_profile = extruder['ActMeltTemperatureProfile']

            for t in act_melt_temperature_profile.values():
                self.meltTemp.append(float(t))
        else:
            self.meltTemp.append(0.0)

        self.tempFZ = []
        if 'ActTemperatureProfileFeedZone' in extruder:
            act_temperature_profile_feed_zone = extruder['ActTemperatureProfileFeedZone']

            for t in act_temperature_profile_feed_zone.values():
                self.tempFZ.append(float(t))
        else:
            self.tempFZ.append(0.0)

        self.tempZA1 = []
        if 'ActTemperatureProfileZoneA1' in extruder:
            act_temperature_profile_zone_a1 = extruder['ActTemperatureProfileZoneA1']

            for t in act_temperature_profile_zone_a1.values():
                self.tempZA1.append(float(t))
        else:
            self.tempZA1.append(0.0)

        self.tempZA10 = []
        if 'ActTemperatureProfileZoneA10' in extruder:
            act_temperature_profile_zone_a10 = extruder['ActTemperatureProfileZoneA10']

            for t in act_temperature_profile_zone_a10.values():
                self.tempZA10.append(float(t))
        else:
            self.tempZA10.append(0.0)

        self.tempZA11 = []
        if 'ActTemperatureProfileZoneA11' in extruder:
            act_temperature_profile_zone_a11 = extruder['ActTemperatureProfileZoneA11']

            for t in act_temperature_profile_zone_a11.values():
                self.tempZA11.append(float(t))
        else:
            self.tempZA11.append(0.0)

        self.tempZA2 = []
        if 'ActTemperatureProfileZoneA2' in extruder:
            act_temperature_profile_zone_a2 = extruder['ActTemperatureProfileZoneA2']

            for t in act_temperature_profile_zone_a2.values():
                self.tempZA2.append(float(t))
        else:
            self.tempZA2.append(0.0)

        self.tempZA3 = []
        if 'ActTemperatureProfileZoneA3' in extruder:
            act_temperature_profile_zone_a3 = extruder['ActTemperatureProfileZoneA3']

            for t in act_temperature_profile_zone_a3.values():
                self.tempZA3.append(float(t))
        else:
            self.tempZA3.append(0.0)

        self.exPressure = []
        if 'ActExtruderPressureProfile' in extruder:
            act_extruder_pressure_profile = extruder['ActExtruderPressureProfile']

            for t in act_extruder_pressure_profile.values():
                self.exPressure.append(float(t))
        else:
            self.exPressure.append(0.0)

        self.exSpeed = []
        if 'ActExtruderSpeedProfile' in extruder:
            act_extruder_speed_profile = extruder['ActExtruderSpeedProfile']

            for t in act_extruder_speed_profile.values():
                self.exSpeed.append(float(t))
        else:
            self.exSpeed.append(0.0)

        self.exTorque = []
        if 'ActExtruderTorqueProfile' in extruder:
            act_extruder_torque_profile = extruder['ActExtruderTorqueProfile']

            for t in act_extruder_torque_profile.values():
                self.exTorque.append(float(t))
        else:
            self.exTorque.append(0.0)

        if 'SetMeltTemperatureProfile' in extruder:
            self.setMeltTemp = 210.0
        else:
            self.setMeltTemp = 0.0

        if 'SetTemperatureProfileFeedZone' in extruder:
            self.setTempFZ = 250.0
        else:
            self.setTempFZ = 0.0

        if 'SetTemperatureProfileZoneA1' in extruder:
            self.setTempZA1 = 190.0
        else:
            self.setTempZA1 = 0.0

        if 'SetTemperatureProfileZoneA10' in extruder:
            self.setTempZA10 = 220.0
        else:
            self.setTempZA10 = 0.0

        if 'SetTemperatureProfileZoneA11' in extruder:
            self.setTempZA11 = 220.0
        else:
            self.setTempZA11 = 0.0

        if 'SetTemperatureProfileZoneA2' in extruder:
            self.setTempZA2 = 205.0
        else:
            self.setTempZA2 = 0.0

        if 'SetTemperatureProfileZoneA3' in extruder:
            self.setTempZA3 = 215.0
        else:
            self.setTempZA3 = 0.0

        del extruder

        del data

    @staticmethod
    def fix_list(the_list):
        ams = the_list.values()
        trim_list = ''
        for i in ams:
            i = i[1:-1]
            trim_list = i
        split_list = trim_list.split(', ')
        split_list = [float(i) for i in split_list]
        clean_list = list(filter(lambda a: a != -1001.0, split_list))
        return clean_list


# Calculate Distances to Simplify Time Series Data
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
            except Exception as e:
                print(e)
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
                except Exception as e:
                    print(e)
                    pass
            ex_default.append(sum(val_av) / len(val_av))
    elif m == 2:
        ff = int((len(column[m - 1]) + len(column[m - 2])) / 2)
        ex_default = []
        column2 = column[(m - 2):m]
        for x in [*range(0, ff)]:
            val_av = []
            for col in column2:
                try:
                    val_av.append(col[x])
                except Exception as e:
                    print(e)
                    pass
            ex_default.append(sum(val_av) / len(val_av))
    elif m == 1:
        ex_default = column[m-1]
    else:
        ex_default = column[m]
    return ex_default


# Change to Product data
def transform_product(df_product):
    new_df = pd.DataFrame(columns=df_product.columns)
    for x in [*range(0, len(df_product)-2)]:
        df_temp = df_product[x:x+3].copy()
        m = 0
        temp_dict = {}
        pass
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
            m += 1
        the_index = ''
        if x < 10:
            the_index = 'C0000' + str(x)
        elif 9 < x < 100:
            the_index = 'C000' + str(x)
        elif 99 < x < 1000:
            the_index = 'C00' + str(x)
        elif 999 < x < 10000:
            the_index = 'C0' + str(x)
        new_df.loc[the_index] = temp_dict
    return new_df


def transform_product_selective(df_product):
    new_df = pd.DataFrame(columns=df_product.columns)
    for x in [*range(0, len(df_product)-2)]:
        df_temp = df_product[x:x+3].copy()
        m = 0
        temp_dict = {}
        pass
        temp_dict['startTime'] = df_temp['startTime'][0]
        temp_dict['endTime'] = df_temp['endTime'][2]
        temp_dict['totalTime'] = df_temp['endTime'][2] - df_temp['startTime'][0]
        temp_dict['ejectTime'] = df_temp['ejectTime'][1]
        the_index = ''
        for col in df_temp.columns:
            if m > 13:
                y = df_temp[col][0].copy()
                temp_dict[col] = y
            elif 3 < m < 14:
                y = df_temp[col][1].copy()
                temp_dict[col] = y
            else:
                pass
            m += 1
        if x < 10:
            the_index = 'C0000' + str(x)
        elif 9 < x < 100:
            the_index = 'C000' + str(x)
        elif 99 < x < 1000:
            the_index = 'C00' + str(x)
        elif 999 < x < 10000:
            the_index = 'C0' + str(x)
        new_df.loc[the_index] = temp_dict
    return new_df


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

    del ex_default
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
    del ex_default
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
            settings_ = i
            temp_temp = i
            df_cycle_main['meltTemp'][m] = calc_distance(i, settings_)
        m += 1
    del settings_
    del temp_temp
    end = time.time()
    print("Calculate Distances")
    print(end - start)
    return df_cycle_main


# Create DataFrame to settings
def create_dataframe_settings(df_cycle_main, df_setting):
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
        df_cycle_main['RWTC1'][m] = calc_distance(i, df_setting['setRWTC1'][m])
        m += 1

    m = 0
    for i in df_cycle_main['RWTC2']:
        df_cycle_main['RWTC2'][m] = calc_distance(i, df_setting['setRWTC2'][m])
        m += 1

    m = 0
    for i in df_cycle_main['suction']:
        df_cycle_main['suction'][m] = calc_distance(i, df_setting['setSuction'][m])
        m += 1

    m = 0
    for i in df_cycle_main['WTC']:
        df_cycle_main['WTC'][m] = calc_distance(i, df_setting['setWTC'][m])
        m += 1

    m = 0
    for i in df_cycle_main['dieTempZ1']:
        settings_ = [df_setting['setDieTempZ1'][m]] * len(i)
        df_cycle_main['dieTempZ1'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['dieTempZ2']:
        settings_ = [df_setting['setDieTempZ2'][m]] * len(i)
        df_cycle_main['dieTempZ2'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['dieTempZ3']:
        settings_ = [df_setting['setDieTempZ3'][m]] * len(i)
        df_cycle_main['dieTempZ3'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['headTempZ1']:
        settings_ = [df_setting['setHeadTempZ1'][m]] * len(i)
        df_cycle_main['headTempZ1'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['headTempZ2']:
        settings_ = [df_setting['setHeadTempZ2'][m]] * len(i)
        df_cycle_main['headTempZ2'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['headTempZ3']:
        settings_ = [df_setting['setHeadTempZ3'][m]] * len(i)
        df_cycle_main['headTempZ3'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['tempFZ']:
        settings_ = [df_setting['setTempFZ'][m]] * len(i)
        df_cycle_main['tempFZ'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['tempZA1']:
        settings_ = [df_setting['setTempZA1'][m]] * len(i)
        df_cycle_main['tempZA1'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['tempZA10']:
        settings_ = [df_setting['setTempZA10'][m]] * len(i)
        df_cycle_main['tempZA10'][m] = calc_distance(i, settings_)

        m += 1

    m = 0
    for i in df_cycle_main['tempZA11']:
        settings_ = [df_setting['setTempZA11'][m]] * len(i)
        df_cycle_main['tempZA11'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['tempZA2']:
        settings_ = [df_setting['setTempZA2'][m]] * len(i)
        df_cycle_main['tempZA2'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['tempZA3']:
        settings_ = [df_setting['setTempZA3'][m]] * len(i)
        df_cycle_main['tempZA3'][m] = calc_distance(i, settings_)
        m += 1

    m = 0
    for i in df_cycle_main['meltTemp']:
        settings_ = [df_setting['setMeltTemp'][m]] * len(i)
        df_cycle_main['meltTemp'][m] = calc_distance(i, settings_)
        m += 1
    del settings_
    del ex_default
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
    df_setting = pd.DataFrame(settings)
    df_setting = df_setting.transpose()
    del cycles
    del meta
    del settings
    del cycles_
    df_full = df_cycle_main.merge(df_setting, left_index=True, right_index=True)

    if method == 1:
        df_cycle_main = create_dataframe(df_cycle_main)
    elif method == 2:
        df_cycle_main = create_dataframe_av(df_cycle_main)
    elif method == 3:
        df_cycle_main = create_dataframe_previous_three(df_cycle_main)
    elif method == 4:
        df_cycle_main = create_dataframe_settings(df_cycle_main, df_setting)
    elif method == 5:
        df_product = transform_product(df_cycle_main)
        df_cycle_main = create_dataframe_av(df_product)
    elif method == 6:
        df_product = transform_product_selective(df_cycle_main)
        df_cycle_main = create_dataframe_av(df_product)
    df_setting = ''
    df_product = ''
    print(df_product, df_setting)
    return df_cycle_main, df_full


# noinspection PyTypeChecker
def read_cycles(folder, method):
    start = time.time()
    cycles_ = []
    json_folder_path = os.path.join(folder)
    json_files = [files for files in os.listdir(json_folder_path) if files.endswith('.json')]
    json_files.sort()
    for json_file in json_files:
        json_file_path = os.path.join(json_folder_path, json_file)
        with open(json_file_path, "r") as file:
            cycle_file = json.load(file)
            if 151.0 > float(cycle_file['CycleTotalTime']) > 49.0:
                cycle_data = Cycle(cycle_file)
                cycles_.append(cycle_data)
    end = time.time()
    print("Create Cycle objects:")
    print(end - start)
    df_cycle_main, df_full = process_cycles(cycles_, method)
    del cycles_
    return df_cycle_main, df_full


def read_cycles_(list_of_contents, list_of_names, list_of_dates, method):
    start = time.time()
    cycles_ = []
    if list_of_contents is not None:
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
    df_cycle_main, df_full = process_cycles(cycles_, method)

    del list_of_contents
    del list_of_names
    del list_of_dates
    del cycles_
    return df_cycle_main, df_full


def update_method(df_full, method):
    df_cycle_main = df_full.iloc[:, 0:24]
    df_setting = df_full.iloc[:, 24:]
    if method == 1:
        df_cycle_main = create_dataframe(df_cycle_main)
    elif method == 2:
        df_cycle_main = create_dataframe_av(df_cycle_main)
    elif method == 3:
        df_cycle_main = create_dataframe_previous_three(df_cycle_main)
    elif method == 4:
        df_cycle_main = create_dataframe_settings(df_cycle_main, df_setting)
    elif method == 5:
        df_product = transform_product(df_cycle_main)
        df_cycle_main = create_dataframe_av(df_product)
    elif method == 6:
        df_product = transform_product_selective(df_cycle_main)
        df_cycle_main = create_dataframe_av(df_product)
    del df_full
    del df_setting
    return df_cycle_main
