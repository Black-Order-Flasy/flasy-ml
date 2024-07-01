import requests

def get_forest_ratio(latitude, longitude):
    url = f"https://api.globalforestwatch.org/v1.4/forest/{latitude},{longitude}"
    response = requests.get(url)
    data = response.json()
    forest_ratio = data['forestCover']  # Assuming the API returns this field
    return forest_ratio

latitude = -6.1751
longitude = 106.8650
forest_ratio = get_forest_ratio(latitude, longitude)
print(f"Forest Ratio at ({latitude}, {longitude}): {forest_ratio}")