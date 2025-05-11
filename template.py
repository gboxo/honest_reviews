




category_and_type = {
    "consumer_electronics": {
        "smartphone": "Smartphone",
        "tablet": "Tablet",
        "laptop": "Laptop",
        "smartwatch": "Smartwatch",
        "TV": "TV",
    },
    "home_appliances": {
        "refrigerator": "Refrigerator",
        "washing_machine": "Washing Machine",
        "dishwasher": "Dishwasher",
        "vacuum_cleaner": "Vacuum Cleaner",
        "air_conditioner": "Air Conditioner",
    },
    "sports_and_fitness": {
        "yoga_mat": "Yoga Mat",
        "running_shoes": "Running Shoes",
        "adjustable_dumbbells": "Adjustable Dumbbells",
        "resistance_bands": "Resistance Bands",
        "fitness_tracker": "Fitness Tracker",
        "exercise_ball": "Exercise Ball",
    }
}















#introduction  =f"""
"""
I am a professional content creator who reviews {product_category}. Over the past month, I have thoroughly tested a {product_type} and rated its key features on a scale from 0 to 10, where 0 means very poor and 10 means excellent.

Below is a JSON object containing the {product_type}'s details and my feature ratings. Using this data, please write a concise, two-paragraph review of the {product_type}. The review should accurately reflect the ratings but **do not mention the numeric scores explicitly**. Instead, use qualitative descriptions to convey the strengths and weaknesses of the device.
"""



"""
```json
{
  "name": "Zephyr X1",
  "model": "ZX1-2025",
  "brand": "NovaTech",
  "year": 2025,
  "operating_system": "NovaOS 3.2",
  "battery_life": 8,
  "camera_quality": 5,
  "processor_performance": 9,
  "screen_clarity": 8,
  "build_quality": 8,
  "audio_quality": 7,
  "charging_speed": 9,
  "software_experience": 8,
  "multitasking_performance": 8,
  "front_camera_quality": 6,
  "video_recording_quality": 7,
  "fingerprint_sensor_accuracy": 9
}
```
"""


final_instructions = """
**Instructions:**

- Write a brief review of about two paragraphs.
- Use qualitative language to describe each feature based on the ratings.
- Highlight both strengths and weaknesses.
- Avoid mentioning any numeric scores or exact ratings.
- The tone should be professional and informative.
"""



