import psycopg2
from geopy.geocoders import Nominatim

con = psycopg2.connect(database="dvdrental", user="postgres",
                       password="Di~S8Qgs~%", host="localhost", port="5432")

cur = con.cursor()
cur.callproc('get_addresses_with_11_and_city_id')

rows = cur.fetchall()
geolocator = Nominatim(user_agent="your_app_name")
for row in rows:
    address_str = str(row[1]) + ", " + ", " + str(row[4])
    try:
        location = geolocator.geocode(address_str)
        print(location.latitude, location.longitude)
        if location is not None:
            update_query = f"UPDATE address SET latitude = {location.latitude}, longitude = {location.longitude} WHERE address_id = {row[0]}"
            cur.execute(update_query)
        else:
            update_query = f"UPDATE address SET latitude = 0, longitude = 0 WHERE address_id = {row[0]}"
            cur.execute(update_query)
    except:
        update_query = f"UPDATE address SET latitude = 0, longitude = 0 WHERE address_id = {row[0]}"
        cur.execute(update_query)

con.commit()
con.close()