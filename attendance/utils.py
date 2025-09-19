# attendance/utils.py
import hmac, hashlib
from math import radians, sin, cos, sqrt, atan2
from config import Config
from flask import current_app
from datetime import datetime, timedelta, timezone

def verify_signature(code: str, ts: str, signature_hex: str) -> bool:
    """
    HMAC-SHA256 signature verification for QR payload.
    message = "{code}|{ts}"
    """
    secret = current_app.config.get("QR_HMAC_SECRET", Config.QR_HMAC_SECRET).encode()
    msg = f"{code}|{ts}".encode()
    expected = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_hex)

def within_radius(lat1, lon1, lat2, lon2, radius_meters: int) -> bool:
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    dist = R * c
    return dist <= radius_meters

def parse_iso_ts(ts_str: str):
    from datetime import datetime
    # simple ISO parser; teammates may use dateutil.parser in future
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None

def code_within_window(code_ts_iso: str, valid_from, valid_to) -> bool:
    ts = parse_iso_ts(code_ts_iso)
    if ts is None:
        return False
    
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # Convert valid_from and valid_to to UTC naive for comparison OR convert ts to naive
    # Option 1: Convert ts and valid_from/to to naive before compare:
    ts_naive = ts.replace(tzinfo=None)
    valid_from_naive = valid_from.replace(tzinfo=None)
    valid_to_naive = valid_to.replace(tzinfo=None)

    # Accept small clock drift: +/- 30 sec (configurable)
    drift = timedelta(seconds=30)
    return (valid_from_naive - drift) <= ts_naive <= (valid_to_naive + drift)
