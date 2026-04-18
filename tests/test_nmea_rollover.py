"""Unit tests for NMEA day-rollover handling and logger time offset."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.preprocess.lidar_slam import LiDARSLAMProcessor


def _compose_gga(seconds_utc: float, lat: float, lon: float) -> str:
    hours = int(seconds_utc // 3600)
    minutes = int((seconds_utc % 3600) // 60)
    secs = seconds_utc - hours * 3600 - minutes * 60
    hhmmss = f"{hours:02d}{minutes:02d}{secs:05.2f}"
    # convert decimal degrees to NMEA ddmm.mmmm
    lat_deg = int(abs(lat))
    lat_min = (abs(lat) - lat_deg) * 60.0
    lat_field = f"{lat_deg:02d}{lat_min:07.4f}"
    lat_hemi = "N" if lat >= 0 else "S"
    lon_deg = int(abs(lon))
    lon_min = (abs(lon) - lon_deg) * 60.0
    lon_field = f"{lon_deg:03d}{lon_min:07.4f}"
    lon_hemi = "E" if lon >= 0 else "W"
    return f"$GPGGA,{hhmmss},{lat_field},{lat_hemi},{lon_field},{lon_hemi},1,08,0.9,100.0,M,0.0,M,,"


def _compose_rmc(seconds_utc: float, lat: float, lon: float, date_ddmmyy: str, course_deg: float) -> str:
    hours = int(seconds_utc // 3600)
    minutes = int((seconds_utc % 3600) // 60)
    secs = seconds_utc - hours * 3600 - minutes * 60
    hhmmss = f"{hours:02d}{minutes:02d}{secs:05.2f}"
    lat_deg = int(abs(lat))
    lat_min = (abs(lat) - lat_deg) * 60.0
    lat_field = f"{lat_deg:02d}{lat_min:07.4f}"
    lat_hemi = "N" if lat >= 0 else "S"
    lon_deg = int(abs(lon))
    lon_min = (abs(lon) - lon_deg) * 60.0
    lon_field = f"{lon_deg:03d}{lon_min:07.4f}"
    lon_hemi = "E" if lon >= 0 else "W"
    return f"$GPRMC,{hhmmss},A,{lat_field},{lat_hemi},{lon_field},{lon_hemi},5.0,{course_deg:.1f},{date_ddmmyy},,"


def test_day_rollover_preserves_monotonic_order(tmp_path: Path) -> None:
    p = tmp_path / "r.nmea"
    lines = [
        _compose_rmc(86397.0, 35.0, 139.0, "310322", 90.0),  # 23:59:57 UTC 2022-03-31
        _compose_rmc(86398.5, 35.0001, 139.0001, "310322", 90.0),  # 23:59:58.5
        _compose_rmc(2.0, 35.0002, 139.0002, "010422", 90.0),  # 00:00:02 UTC 2022-04-01
        _compose_rmc(5.0, 35.0003, 139.0003, "010422", 90.0),  # 00:00:05 UTC 2022-04-01
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    proc = LiDARSLAMProcessor()
    timestamps, poses = proc._load_nmea_trajectory(p)
    assert len(timestamps) == 4
    # Monotonically increasing (no backward jump at midnight)
    for a, b in zip(timestamps, timestamps[1:]):
        assert b > a, f"timestamps regressed: {a} -> {b}"
    # Consecutive gap bridging midnight is ~3.5 s, within day (not -86395 s)
    assert 2.0 < timestamps[2] - timestamps[1] < 10.0


def test_time_offset_is_added_to_every_row(tmp_path: Path) -> None:
    p = tmp_path / "r.nmea"
    p.write_text(
        "\n".join(
            [
                _compose_rmc(100.0, 35.0, 139.0, "010122", 0.0),
                _compose_rmc(101.0, 35.0, 139.0, "010122", 0.0),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = LiDARSLAMProcessor()
    t0, _ = proc._load_nmea_trajectory(p)
    t1, _ = proc._load_nmea_trajectory(p, time_offset_sec=7.5)
    assert len(t0) == len(t1) == 2
    for a, b in zip(t0, t1):
        assert b - a == 7.5


def test_gga_and_rmc_merged_at_same_second(tmp_path: Path) -> None:
    """GGA (position) + RMC (course, date) sharing the same seconds should yield one entry."""
    p = tmp_path / "r.nmea"
    p.write_text(
        "\n".join(
            [
                _compose_gga(50.0, 35.0, 139.0),
                _compose_rmc(50.0, 35.0, 139.0, "010122", 45.0),
                _compose_gga(51.0, 35.0001, 139.0001),
                _compose_rmc(51.0, 35.0001, 139.0001, "010122", 45.0),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    proc = LiDARSLAMProcessor()
    timestamps, poses = proc._load_nmea_trajectory(p)
    assert len(timestamps) == 2
