"""Radio channel utilities for synthetic 6G RAN simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RadioParams:
    """Parameter container for radio link budget calculations."""

    carrier_frequency_ghz: float = 28.0
    tx_power_dbm: float = 43.0
    noise_floor_dbm: float = -100.0
    shadowing_std_db: float = 3.0


class RadioChannel:
    """Simulates radio signal quality metrics for UE-cell links."""

    def __init__(self, params: RadioParams | None = None) -> None:
        self.params = params or RadioParams()

    def path_loss_db(self, distance_m: float) -> float:
        """Compute free-space-like path loss with minimum distance handling."""
        distance_km = max(distance_m / 1000.0, 0.001)
        frequency_mhz = self.params.carrier_frequency_ghz * 1000.0
        return 32.44 + 20 * math.log10(distance_km) + 20 * math.log10(frequency_mhz)

    def rsrp_dbm(self, distance_m: float, beam_gain_db: float, shadowing_db: float) -> float:
        """Estimate RSRP in dBm from distance and beam gain."""
        loss = self.path_loss_db(distance_m)
        return self.params.tx_power_dbm + beam_gain_db - loss + shadowing_db

    def sinr_db(self, signal_dbm: float, interference_dbm: float) -> float:
        """Compute SINR in dB from signal/interference/noise powers."""

        def dbm_to_mw(value_dbm: float) -> float:
            return 10 ** (value_dbm / 10.0)

        signal_mw = dbm_to_mw(signal_dbm)
        interference_mw = dbm_to_mw(interference_dbm)
        noise_mw = dbm_to_mw(self.params.noise_floor_dbm)
        sinr_linear = signal_mw / (interference_mw + noise_mw)
        return 10 * math.log10(max(sinr_linear, 1e-9))

    @staticmethod
    def sinr_to_cqi(sinr_db: float) -> int:
        """Map SINR to CQI using a coarse LTE/NR-inspired threshold table."""
        thresholds = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7]
        for i, threshold in enumerate(thresholds, start=1):
            if sinr_db < threshold:
                return i
        return 15

    @staticmethod
    def qos_class_from_sinr(sinr_db: float) -> str:
        """Categorize QoS class from SINR."""
        if sinr_db >= 15:
            return "good"
        if sinr_db >= 5:
            return "medium"
        return "poor"

    @staticmethod
    def throughput_mbps_from_sinr(sinr_db: float, bandwidth_mhz: float = 100.0) -> float:
        """Approximate throughput using Shannon-like capacity formula."""
        sinr_linear = 10 ** (sinr_db / 10.0)
        spectral_eff = math.log2(1 + max(sinr_linear, 1e-9))
        return bandwidth_mhz * spectral_eff

    @staticmethod
    def latency_ms_from_sinr(sinr_db: float) -> float:
        """Approximate latency inverse to link quality."""
        return max(1.0, 50.0 - 1.5 * sinr_db)
