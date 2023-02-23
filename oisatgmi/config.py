import numpy as np
from dataclasses import dataclass
import datetime


@dataclass
class satellite:
    vcd: np.ndarray
    scd: np.ndarray
    time: datetime.datetime
    profile: np.ndarray
    tropopause: np.ndarray
    latitude_center: np.ndarray
    longitude_center: np.ndarray
    latitude_corner: np.ndarray
    longitude_corner: np.ndarray
    uncertainty: np.ndarray
    quality_flag: np.ndarray
    pressure_mid: np.ndarray
    averaging_kernels: np.ndarray
    scattering_weights: np.ndarray
    ctm_upscaled_needed: bool

@dataclass
class ctm_model:
    latitude: np.ndarray
    longitude: np.ndarray
    time: list
    gas_profile: dict
    pressure_mid: np.ndarray
    tempeature_mid: np.ndarray
    delta_p: np.ndarray
    ctmtype: str
    vcd: np.ndarray
    time_at_sat: datetime.datetime
