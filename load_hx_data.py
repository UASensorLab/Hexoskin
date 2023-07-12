"""
(c) Copyright 2015 Hexoskin
Permission to use, copy, modify, and distribute this software for any purpose with or without fee is hereby granted,
provided that the above copyright notice and this permission notice appear in all copies. The software is provided
"as is" and the author disclaims all warranties with regard to this software including all implied warranties of
merchantability and fitness. In no event shall the author be liable for any special, direct, indirect, or
consequential damages or any damages whatsoever resulting from loss of use, data or profits, whether in an action of
contract, negligence or other tortious action, arising out of or in connection with the use or performance of this
software.
"""

import csv
import json
import os
import struct
import wave

__version__ = '1.2.0'

global DEVICE_MODEL
DEVICE_MODEL = None


def _get_device_model(filepath):
    if DEVICE_MODEL:
        return DEVICE_MODEL
    else:
        if load_info(os.path.dirname(filepath))['freq'] == 256:
            return 'hx'
        else:
            return 'lsrs'



def set_device_model(device_model):
    """
    set the device model used
    device_model = 'hx', 'hms', 'lsrs'
    """
    global DEVICE_MODEL
    DEVICE_MODEL = device_model


def load_np_data(filepath, offset=0, skip_start=0, skip_end=None, no_timestamps=False):
    """
    Loads a  file according to the specified format. Time t=0 corresponds to
    the beginning of the record.
    :param str filepath:
     :param float offset: an offset in second to add to the time of the data
    :param float skip_start: remove all data before skip_start (relative to record start)
    :param float skip_end: remove all data after skip_end (relative to record start)
    :param boolean no_timestamps: don't return timestamps

    :return: array
    """
    # note the import is placed here so the rest of the code can run without the numpy module
    from numpy import array
    return array(load_data(filepath, offset, skip_start, skip_end,no_timestamps))


def load_data(filepath, offset=0, skip_start=0, skip_end=None, no_timestamps=False):
    """
    Loads a  file according to the specified format. Time t=0 corresponds to
    the beginning of the record.
    :param str filepath:
    :param float offset: an offset in second to add to the time of the data
    :param float skip_start: remove all data before skip_start (relative to record start)
    :param float skip_end: remove all data after skip_end (relative to record start)
    :param boolean no_timestamps: don't return timestamps

    :return: list
    """
    data = []
    if os.path.exists(filepath + '.wav'):
        data = load_wave(filepath + '.wav', offset, skip_start, skip_end, no_timestamps)
    elif os.path.exists(filepath + '.csv'):
        data = load_csv(filepath + '.csv', offset, skip_start, skip_end, no_timestamps)
    elif os.path.exists(filepath + '.txt'):
        data = load_csv(filepath + '.txt', offset, skip_start, skip_end, no_timestamps)
    elif os.path.exists(filepath) and filepath.endswith('.wav'):
        data = load_wave(filepath, offset, skip_start, skip_end, no_timestamps)
    elif os.path.exists(filepath) and filepath.endswith('.csv'):
        data = load_csv(filepath, offset, skip_start, skip_end, no_timestamps)
    elif os.path.exists(filepath) and filepath.endswith('.txt'):
        data = load_csv(filepath, offset, skip_start, skip_end, no_timestamps)
    elif filepath.endswith('.wav') and os.path.exists(filepath[:-4]):
        data = load_wave(filepath[:-4], offset, skip_start, skip_end, no_timestamps)
    elif  os.path.exists(filepath[:-4]) and filepath.endswith('.wav') :
        data = load_wave(filepath[:-4], offset, skip_start, skip_end, no_timestamps)
    elif os.path.exists(filepath[:-4]) and filepath.endswith('.csv'):
        data = load_csv(filepath[:-4], offset, skip_start, skip_end, no_timestamps)
    elif os.path.exists(filepath[:-4]) and filepath.endswith('.txt'):
        data = load_csv(filepath[:-4], offset, skip_start, skip_end, no_timestamps)
    return data


def load_csv(filepath, offset=0, skip_start=0, skip_end=None, no_timestamps=False):
    """
    Loads a csv file according to the specified format. Time t=0 corresponds to
    the beginning of the record.
    :param str filepath:
    :param float offset: an offset in second to add to the time of the data
    :param float skip_start: remove all data before skip_start (relative to record start)
    :param float skip_end: remove all data after skip_end (relative to record start)
    :param boolean no_timestamps: don't return timestamps

    """
    data = []
    file_specs = getspec(filepath)
    with open(filepath, 'r') as csvfile:
        reader_csv = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader_csv:
            try:
                try:
                    int(row[1])  # ensure it is an int otherwise it was already converted, don't change second column

                    t, val = [float(row[0]) + offset, float(row[1]) * file_specs['gain'] + file_specs['offset']]
                except ValueError:
                    t, val = [float(row[0]) + offset, float(row[1])]
                if not (skip_start and t < skip_start):
                    if skip_end and t > skip_end:
                        break
                    if no_timestamps:
                        data.append([val])
                    else:
                        data.append([t, val])
            except ValueError:
                pass
    return data


def load_wave(filepath, offset=0, skip_start=0, skip_end=None, no_timestamps=False):
    """
    Loads a WAV file according to the specified format. Time t=0 corresponds to
    the beginning of the record.
    :param str filepath:
    :param float offset: an offset in second to add to the time of the data
    :param float skip_start: remove all data before skip_start (relative to record start)
    :param float skip_end: remove all data after skip_end (relative to record start)
    :param boolean no_timestamps: don't return timestamps
    :return:
    """
    # note data can be loaded with scipy.io.wavfile or numpy, but we keep here a decoding possible
    # with the basic python package
    file_specs = getspec(filepath)
    f = wave.open(filepath)
    freq_signal = f.getframerate()
    skip_start_n = int(skip_start * freq_signal) if skip_start else 0
    skip_start = skip_start_n / freq_signal
    f.setpos(skip_start_n)
    skip_end_n = int(skip_end) * freq_signal if skip_end else f.getnframes()
    skip_end_n = min(skip_end_n, f.getnframes())
    nframes = skip_end_n - skip_start_n
    if f.getsampwidth() == 4:
        yy = struct.unpack('<i' * nframes, f.readframes(nframes))
    elif file_specs['signed'] and f.getsampwidth() == 2:
        yy = struct.unpack("h" * nframes, f.readframes(nframes))
    elif f.getsampwidth() == 2:
        yy = struct.unpack("H" * nframes, f.readframes(nframes))
    else:
        raise NotImplementedError
    f.close()

    if 1 / freq_signal == file_specs['dt']:
        dt = file_specs['dt']
    elif freq_signal == file_specs['dt']:
        dt = file_specs['dt']
    else:
        raise ('invalid frequency {}'.format(filepath))
    if no_timestamps:
        return [float(y) * file_specs['gain'] + file_specs['offset'] for t, y in enumerate(yy)]
    else:
        return [(float(t) * dt + file_specs['dt_offset'] + offset + skip_start,
                 float(y) * file_specs['gain'] + file_specs['offset']) for t, y in enumerate(yy)]


def get_files_specs(device_model, as_dict=True):
    """
    Get all specs for a given frequency
    :param device_model: device frequency 256 or 1000
    :param as_dict: return result as dict instead of list

    :return: table or dict of all available datatype
    """
    GAIN_ECG = 0.0064 if device_model == "hx" else 0.001
    OFFSET_ECG = -1360 * 0.0064 if device_model == "hx" else 0
    dt_e = 1 / 256 if device_model == "hx" else 1 / 250
    dt_r = 1 / 128 if device_model == "hx" else 1 / 125
    dt_a = 1 / 64 if device_model == "hx" else 1 / 50
    dt_o = 1 / 64 if device_model == "hx" else 1 / 75
    br_gain = 1 if device_model == "hx" else .1
    gain_resp = 13.28 if device_model == "hx" else 1
    tmp_gain = 1 if device_model == "hx" else 1 / 256
    keys_list = ['filename', 'filename_out', 'quality_signal_name', 'unit', 'dt', 'dt_offset',
                 'gain', 'offset', 'signed', 'round', 'datatype', 'is_wav']

    device_freq = 256 if device_model == 'hx' else 1000

    specs = [
        # filename, csv_name, quality_signal_name, unit, dt, t_offset, gain, offset, signed, round, datatype, is_wav
        ["ECG_I", "ECG_I", "", "mV", dt_e, 0, GAIN_ECG, OFFSET_ECG, 1, 5, 4113, True],
        ["ECG_II", "ECG_II", "", "mV", dt_e, 0, GAIN_ECG, OFFSET_ECG, 1, 5, 4114, True],
        ["ECG_III", "ECG_III", "", "mV", dt_e, 0, GAIN_ECG, OFFSET_ECG, 1, 5, 4115, True],
        ["ECG_quality", "ECG_quality", "NONE", "", 1, 1, 1, 0, 0, 0, 1006, True],
        ["respiration_thoracic", "respiration_thoracic", "", "na", dt_r, 0, 1, 0, 0, 5, 4129, True],
        ["respiration_abdominal", "respiration_abdominal", "", "na", dt_r, 0, 1, 0, 0, 5, 4130, True],
        ["respiration_total", "respiration_total", "", "na", dt_r, 0, 1, 0, 0, 5, 99999, True],

        ["acceleration_X", "acceleration_X", "", "G", dt_a, 0, 1 / 256, 0, 1, 8, 4145, True],
        ["acceleration_Y", "acceleration_Y", "", "G", dt_a, 0, 1 / 256, 0, 1, 8, 4146, True],
        ["acceleration_Z", "acceleration_Z", "", "G", dt_a, 0, 1 / 256, 0, 1, 8, 4147, True],
        ["coretemp", "coretemp", "", "C", 1, 0, 1 / 256, 0, 0, 5, None, True],

        ["ppgdo", "ppgdo", "", "na", dt_a, 0, 1, 0, 0, 5, 64, True],
        ["ppgDisc", "ppgDisc", "", "na", dt_a, 0, 1, 0, 0, 0, 64, True],
        ["ppgOOT", "ppgOOT", "", "na", dt_a, 0, 1, 0, 0, 5, 64, True],
        ["heart_rate", "heart_rate", "heart_rate_quality", "bpm", 1, 1, 1, 0, 0, 0, 19, True],
        ["energy_mifflin_keytel", "energy_mifflin_keytel", "", "watt", 1, 1, 1, 0, 0, 0, 42, True],
        ["heart_rate_quality", "heart_rate_quality", "NONE", "na", 1, 1, 1, 0, 0, 0, 1000, True],
        ["breathing_rate", "breathing_rate", "breathing_rate_quality", "rpm", 1, 1, br_gain, 0, 0, 0, 33, True],
        ["breathing_rate_quality", "breathing_rate_quality", "NONE", "", 1, 1, 1, 0, 0, 0, 1001, True],
        ["minute_ventilation", "minute_ventilation_raw", "", "ml/min", 1, 1, gain_resp, 0, 0, 5, 36, True],
        ["tidal_volume", "tidal_volume_raw", "", "ml", 1, 1, gain_resp, 0, 0, 5, 37, True],
        ["minute_ventilation_adjusted", "minute_ventilation_adjusted", "", "ml/min", 1, 1, 10, 0, 0, 5, 38, True],
        ["tidal_volume_adjusted", "tidal_volume_adjusted", "", "ml", 1, 1, 1, 0, 0, 5, 39, True],
        ["minute_ventilation_cl", "minute_ventilation_adjusted", "", "ml/min", 1, 1, 10, 0, 0, 5, 38, True],
        ["tidal_volume_ml", "tidal_volume_adjusted", "", "ml", 1, 1, 1, 0, 0, 5, 39, True],
        ["activity", "activity", "", "G", 1, 1, 1 / 256, 0, 1, 8, 49, True],
        ["actigraphy", "actigraphy", "", "G", 1, 1, 1 / 256, 0, 1, 8, 54, True],
        ["cadence", "cadence", "", "spm", 1, 1, 1, 0, 0, 0, 53, True],
        ["temperature_celcius", "temperature_celcius", "", "Celcius", 1, 1, tmp_gain, 0, 1, 8, 81, True],
        ["temperature", "temperature", "", "Celcius", 1, 1, tmp_gain, 0, 1, 8, 81, True],
        ["RESP_TMPR", "RESP_TMPR", "", "Celcius", 1, 1, 0.01, 0, 1, 8, 338, True],
        ["resp_temp", "resp_temp", "", "Celcius", 1, 1, 0.01, 0, 1, 8, None, True],

        ["systolic_pressure", "systolic_pressure", "", "mmHg", 1, 1, 1, 0, 0, 5, 98, True],
        ["systolic_pressure_adjusted", "systolic_pressure_adjusted", "", "mmHg", 1, 1, 1, 0, 0, 5, 99, True],
        ["systolic_pressure_quality", "systolic_pressure_quality", "NONE", "", 1, 1, 1, 0, 0, 0, 1005, True],

        ["SPO2", "SPO2", "SPO2_quality", "Percent", 1, 1, 1, 0, 0, 0, 66, True],
        ["SPO2_quality", "SPO2_quality", "NONE", "", 1, 1, 1, 0, 0, 0, 1002, True],
        ["BATT_ST_CHANNEL_CHAR", "BATT_ST_CHANNEL_CHAR", "", "na", 1, 1, 1, 0, 0, 0, 247, True],
        ["TEMP_ST_CHANNEL_CHAR", "TEMP_ST_CHANNEL_CHAR", "", "na", 1, 1, 1, 0, 0, 0, 246, True],
        ["HRV_HF", "HRV_HF", "", "ms^2", 300, 300, 1 / 10, 0, 0, 8, 277, True],
        ["HRV_LF_normalized", "HRV_LF_normalized", "", "", 300, 300, 1 / 100, 0, 0, 8, 273, True],
        ["HRV_LF", "HRV_LF", "", "ms^2", 300, 300, 1 / 10, 0, 0, 8, 276, True],
        ["NN_over_RR", "NN_over_RR", "", "%", 300, 300, 1 / 100, 0, 0, 8, 274, True],
        ["ANN", "ANN", "", "s", 300, 300, 1 / device_freq / 16, 0, 0, 8, 271, True],
        ["SDNN", "SDNN", "", "s", 300, 300, 1 / device_freq / 16, 0, 0, 8, 272, True],
        ["HRV_triangular", "HRV_triangular", "", "na", 300, 300, 1 / 100, 0, 0, 5, 275, True],
        ["RMSSD", "RMSSD", "", "s", 300, 300, 1 / device_freq / 16, 0, 0, 8, 278, True],
        ["PPG", "PPG", "", "na", dt_o, 0, 1, 0, 0, 5, 64, True],
        ["QRS", "QRS_epoch", "", "na", 1, 0, 1, 0, 0, 0, 17, False],
        ["RR_interval", "RR_interval_epoch", "RR_interval_quality", "s", 1, 0, 1 / device_freq, 0, 0, 8, 18, False],
        ["inspiration", "inspiration_epoch", "", "na", 1, 0, 1, 0, 0, 0, 34, False],
        ["expiration", "expiration_epoch", "", "", 1, 0, 1, 0, 0, 0, 35, False],
        ["step", "step_epoch", "", "step", 1, 0, 1, 0, 0, 0, 52, False],
        ["battery_percent", "battery_percent", "", "%", 1, 0, 1, 0, 0, 0, 99999, False],
        ["battery_mVolt", "battery_mVolt", "", "mV", 1, 0, 1, 0, 0, 0, 99999, False],
        ["battery_current_uA", "battery_current_uA", "", "uA", 1, 0, 1, 0, 0, 0, 99999, False],

        ["PTT", "PTT_epoch", "", "s", 1, 0, 1 / device_freq, 0, 0, 0, 97, False],
        ["device_position", "device_position_epoch", "", "na", 1, 0, 1, 0, 0, 0, 269, False],
        ["sleep_position", "sleep_position_epoch", "", "na", 1, 0, 1, 0, 0, 0, 270, False],
        ["sleep_phase", "sleep_phase_epoch", "", "na", 1, 0, 1, 0, 0, 0, 280, False],
        ["NN_interval", "NN_interval_epoch", "", "s", 1, 0, 1 / device_freq, 0, 0, 8, 318, False],
        ["RR_interval_realigned", "RR_interval_realigned_epoch", "", "s", 1, 0, 1 / device_freq, 0, 0, 8, 319, False],
        ["RR_interval_quality", "RR_interval_quality_epoch", "", "na", 1, 0, 1, 0, 0, 0, 1004, False],
    ]
    if as_dict:
        return {vals[0]: {key: val for key, val in zip(keys_list, vals)} for vals in specs}
    else:
        return specs


def getspec(filepath):
    """
    return the specs of the given file
    :param str filepath:
    :return: dict
    """
    filename = os.path.splitext(os.path.split(filepath)[1])[0]
    device_model = _get_device_model(filepath)
    try:
        tmp = get_files_specs(device_model)
        if filename in tmp:
            return tmp[filename]
        return next(a for a in tmp.values() if a['datatype'] == int(filename))

    except (ValueError, StopIteration):
        raise AssertionError('Not a valid filename {}'.format(filepath))


def load_info(directory):
    """
    load the info.json file
    :param str directory:
    :return: dict
     """
    info = os.path.join(directory, "info.json")
    if os.path.exists(info):
        with open(os.path.join(directory, "info.json"), "r") as f:
            info = json.load(f)
    else:
        info = {}
    info['freq'] = _check_freq(info, directory)
    return info


def _check_freq(info, directory):
    """Check the frequency of the data.
    This allow to load data from hexoskin (256 Hz) and Astroskin (1000 Hz).
    :param dict info:
    :return: frequency
    """

    root_start = info.get('start', info.get('start_timestamp', 0))

    if 1230768000 * 256 < root_start < 2524608000 * 256:  # 2009, 2050
        return 256
    elif 1230768000 * 1000 < root_start < 2524608000 * 1000:  # 2009, 2050
        return 1000
    elif os.path.exists(os.path.join(directory, 'ECG_I.wav')):
        with wave.open(os.path.join(directory, 'ECG_I.wav')) as f:
            return 256 if f.getframerate() == 256 else 1000
    elif os.path.exists(os.path.join(directory, '4113.wav')):
        with wave.open(os.path.join(directory, '4113.wav')) as f:
            return 256 if f.getframerate() == 256 else 1000
    else:

        raise AssertionError("start_timestamp ={}. out of range. Aborted.".format(root_start))


def get_status_info(filepath):
    """
    Return status code related to the given model
    filepath:
    aa=32


    """
    filename = os.path.splitext(os.path.split(filepath)[1])[0]
    model = _get_device_model(os.path.dirname(filepath))

    status = _get_status_info(model)
    for key, val in status.items():
        if key in filename:
            return status[key]
    raise ValueError('Status not available for the current file')


def filter_data(data, data_quality, mask):
    """
    Put nan on data where the quality is not good
    """
    from numpy import nan, array
    masked_vals = (data_quality[:, -1].astype(int) & mask) != 0
    data_out = array(data)
    data_out[masked_vals, -1] = nan
    return data_out


def _get_status_info(model='hx'):
    """
    Return status code related to the given model
    model: hx, lsrs, hms
    """
    if model not in ['hx', 'lsrs', 'hms']:
        raise ValueError('invalid choice, choices are: hx, lsrs, hms')

    if model == 'hms':
        status = {
            "heart_rate": [
                ["HR_STATUS_NOISY", 1, "0x01"],
                ["HR_STATUS_DISCONNECTED", 2, "0x02"],
                ["HR_STATUS_50_60HZ", 4, "0x04"],
                ["HR_STATUS_SATURATED", 8, "0x08"],
                ["HR_STATUS_ARTIFACTS", 16, "0x10"],
                ["HR_STATUS_UNRELIABLE_RR", 32, "0x20"],
            ]
        }
    else:
        status = {
            "heart_rate": [
                ["HR_STATUS_DISCONNECTED", 2, "0x02"],
                ["HR_STATUS_50_60HZ", 4, "0x04"],
                ["HR_STATUS_SATURATED", 8, "0x08"],
                ["HR_STATUS_ARTIFACTS", 16, "0x10"],
                ["HR_STATUS_UNRELIABLE_RR", 32, "0x20"],
            ]
        }

    if model == 'hms':
        status["breathing_rate"] = [
            ["RESP_STATUS_NO_A", 1, "0x01"],
            ["RESP_STATUS_NO_B", 2, "0x02"],
            ["RESP_STATUS_BASELINE_A", 4, "0x04"],
            ["RESP_STATUS_BASELINE_B", 8, "0x08"],
            ["RESP_STATUS_NOISY_A", 16, "0x10"],
            ["RESP_STATUS_NOISY_B", 32, "0x20"],
            ["RESP_STATUS_BR_UNSTABLE", 64, "0x40"]
        ]
    else:
        status["breathing_rate"] = [
            ["RESP_STATUS_NO_A", 1, "0x01"],
            ["RESP_STATUS_NO_B", 2, "0x02"],
            ["RESP_STATUS_BASELINE_A", 4, "0x04"],
            ["RESP_STATUS_BASELINE_B", 8, "0x08"],
            ["RESP_STATUS_NOISY_A", 16, "0x10"],
            ["RESP_STATUS_NOISY_B", 32, "0x20"],
        ]
    if model != 'hms':
        status['SPO2'] = [
            ["PPG_STATUS_SENSOR_ALARM", 4, "0x04"],
            ["PPG_STATUS_OUT_OF_TRACK", 8, "0x08"],
            ["PPG_STATUS_ARTIFACTS", 16, "0x10"],
            ["PPG_STATUS_DISCONNECTED", 32, "0x20"],
            ["PPG_P_HR_UNRELIABLE", 64, "0x40"],
            ["PPG_SPO2_UNRELIABLE", 128, "0x80"],
        ]
    if model == 'hms':
        status["ECG"] = [
            ["LEADOFF_LA", 1, "0x01"],
            ["LEADOFF_RA", 2, "0x02"],
            ["LEADOFF_LL", 4, "0x04"],
            ["LEADOFF_RL", 8, "0x08"],
            ["LEAD_I_NOISY", 16, "0x10"],
            ["LEAD_II_NOISY", 32, "0x20"],
            ["LEAD_III_NOISY", 64, "0x40"],
            ["LEAD_I_UNRELIABLE_HR", 128, "0x80"],
            ["LEAD_II_UNRELIABLE_HR", 256, "0x100"],
            ["LEAD_III_UNRELIABLE_HR", 512, "0x200"],
            ["LEAD_I_ARTIFACT", 1024, "0x400"],
            ["LEAD_II_ARTIFACT", 2048, "0x800"],
            ["LEAD_III_ARTIFACT", 4096, "0x1000"],
            ["LEAD_I_DISCONNECTED", 8192, "0x2000"],
            ["LEAD_II_DISCONNECTED", 16384, "0x4000"],
            ["LEAD_III_DISCONNECTED", 32768, "0x8000"]]
    elif model == 'lsrs':
        status["ECG"] = [["LEADOFF_LA", 1, "0x01"],
                         ["LEADOFF_RA", 2, "0x02"],
                         ["LEADOFF_LL", 4, "0x04"],
                         ["LEADOFF_RL", 8, "0x08"],
                         ["NA", 16, "0x10"],
                         ["NA", 32, "0x20"],
                         ["NA", 64, "0x40"],
                         ["LEAD_I_UNRELIABLE_RR", 128, "0x80"],
                         ["LEAD_II_UNRELIABLE_RR", 256, "0x100"],
                         ["LEAD_III_UNRELIABLE_RR", 512, "0x200"],
                         ["LEAD_I_ARTIFACT", 1024, "0x400"],
                         ["LEAD_II_ARTIFACT", 2048, "0x800"],
                         ["LEAD_III_ARTIFACT", 4096, "0x1000"],
                         ["LEAD_I_DISCONNECTED", 8192, "0x2000"],
                         ["LEAD_II_DISCONNECTED", 16384, "0x4000"],
                         ["LEAD_III_DISCONNECTED", 32768, "0x8000"]]
    if model == 'hms':
        status["systolic_pressure"] = [
            ["ECG_OFF_NOMINAL", 1, "0x01"],
            ["PPG_OFF_NOMINAL", 2, "0x02"],
            ["DELAY_OFF_NOMINAL", 4, "0x04"]]
    return status
