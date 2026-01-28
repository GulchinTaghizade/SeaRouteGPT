from dataclasses import dataclass
from datetime import date
from typing import List

@dataclass
class Cruise:
    cruise_id: str
    cruise_name: str
    cruise_line: str
    ship_code: str

    departure_port: str
    departure_date: date
    duration_days: int

    ports_of_call: List[str]
    destinations: List[str]

    price_total: float
    price_per_night: float
    room_category: str
    max_guests: int

    cruise_type: str
    is_sold_out: bool