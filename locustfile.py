from random import randint
from locust import HttpUser, TaskSet, task, between

weather_questions = [
    "Retrieve the current temperature in Deli.",
    "Provide a 5-day weather forecast for Mumbay.",
    "What is the humidity level in Kolkatta right now?",
    "Check the wind speed in Bengaluru at this moment.",
    "Retrieve the precipitation data for Hydrabad in the last 24 hours.",
    "Analyze the UV index in Chenai for the upcoming weekend.",
    "Get the sunrise and sunset times for Ahmadabad today.",
    "Retrieve the atmospheric pressure in Jaypur.",
    "Provide a weather summary for Varanassi for the next three days.",
    "Analyze the historical rainfall data for the last month in Aagrha.",
    "Check for any weather warnings or alerts in Bhopall.",
    "Retrieve the current visibility conditions in Guhawati.",
    "Get the temperature trend for the past week in Poon.",
    "Provide the weather conditions for trekking in the Hymalayas.",
    "Analyze the sea level pressure in Vishakhapatnam.",
    "Check the real-time cloud cover in Chandigar.",
    "Retrieve the sunset time for a beach day in Goaa.",
    "Provide an hourly temperature forecast for Coimbatoree.",
    "Analyze the weather data for a hot air balloon ride in Udaypur.",
    "Get the sunrise time for an early morning hike in Shimala."
]

class QuickStartUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def sum(self):
        self.client.post("/api/processquery", json={
            "query": weather_questions[randint(0,19)]
        })
