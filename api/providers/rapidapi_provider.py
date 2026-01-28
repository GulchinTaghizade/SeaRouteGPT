import requests
from datetime import date
from models.schemas.cruise import Cruise

class RapidAPICruiseProvider:
    BASE_URL = "https://cruise-api1.p.rapidapi.com/cruises"

    def __init__(self, api_key: str):
        self.headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "cruise-api1.p.rapidapi.com"
        }

    def fetch_cruises(self) -> list[Cruise]:
        response = requests.get(self.BASE_URL, headers=self.headers)
        response.raise_for_status()
        raw_cruises = response.json()

        cruises = []
        for item in raw_cruises:
            cruises.append(self._to_cruise(item))

        return cruises

    def _to_cruise(self, item: dict) -> Cruise:
        return Cruise(
            cruise_id=item["cruiseId"],
            cruise_name=item["cruiseName"],
            cruise_line=item["cruiseLineCode"],
            ship_code=item["shipCode"],

            departure_port=item["itineraryPorts"][0],
            departure_date=date.fromisoformat(item["departureDate"]),
            duration_days=item["duration"],

            ports_of_call=item["itineraryPorts"],
            destinations=item["itineraryDestinations"],

            price_total=item["roomPriceWithTaxesFees"],
            price_per_night=item["roomPriceWithTaxesFeesPerNight"],
            room_category=item["roomTypeCategory"],
            max_guests=item["numberOfGuests"],

            cruise_type=item["cruiseType"],
            is_sold_out=item["soldOut"]
        )