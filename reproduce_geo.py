from organizer.models import GeoLocation

def test_geolocation():
    try:
        data = {'latitude': 40.7128, 'longitude': -74.0060}
        print(f"Attempting to create GeoLocation with: {data}")
        loc = GeoLocation(
            latitude=data['latitude'],
            longitude=data['longitude'],
            raw_location_string=str(data)
        )
        print(f"Success: {loc}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_geolocation()
