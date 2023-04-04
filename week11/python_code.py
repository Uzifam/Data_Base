from pymongo import MongoClient

#client = MongoClient("mongodb://hostname:port")
client = MongoClient("mongodb://localhost") # will connect to localhost and default port 27017


db = client['test']

mycol = db["restaurants"]

def Ex_1_only_Irish():
    for i in mycol.find({'cuisine' : 'Irish'}):
        print(i)


def Ex_1_Irish_Russia():
    for i in mycol.find({'cuisine' : { "$in": ["Irish", "Russian"] } }):
        print(i)


def Ex_1_find_restaurant_by_address():
    query = { "address": {'street' : 'Prospect Park West ', 'zipcode' : '11215'} }
    result = mycol.find_one(query)
    return result


def Ex_2_add_resourant():

    restaurant = {
        'address': 'Sportivnaya 126, 420500',
        'borough': 'Innopolis',
        'cuisine': 'Serbian',
        'name': 'The Best Restaurant',
        'restaurant_id': 41712354,
        'grades': [
            {
                'grade': 'A',
                'score': 11,
                'date': '04 Apr, 2023'
            }
        ]
    }
    mycol.insert_one(restaurant)



def Ex_3_delete_brooklyn_restaurant():
    db = client['mydatabase']
    collection = db['restaurants']
    result = collection.delete_one({"borough": "Brooklyn"})
    print(result.deleted_count, "restaurant deleted successfully!")


def Ex_3_delete_all_thai_cuisines():
    db = client['mydatabase']
    collection = db['restaurants']
    result = collection.delete_many({"cuisine": "Thai"})
    print(result.deleted_count, "restaurants deleted successfully!")


def Ex_4_query_ppw_restaurants():
    restaurants = []
    cursor = mycol.find({"address.street": "Prospect Park West"})
    for document in cursor:
        restaurants.append(document)
    return restaurants

def Ex_4_update_restaurant_grades(restaurants):
    for restaurant in restaurants:
        a_grades = [grade for grade in restaurant['grades'] if grade['grade'] == 'A']
        if len(a_grades) > 1:
            mycol.delete_one({"_id": restaurant["_id"]})
        else:
            new_grade = {"grade": "A", "score": 11, "date": "04 Apr, 2023"}
            mycol.update_one({"_id": restaurant["_id"]}, {"$push": {"grades": new_grade}})


            
Ex_1_find_restaurant_by_address()