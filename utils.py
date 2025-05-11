
feature_json_object = """{
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
}"""

import json

def parse_json_to_template(json_str):
    """
    Parse a JSON string and return a template string with variables as key names.
    
    Args:
        json_str (str): A valid JSON string
        
    Returns:
        str: A template string with variables in the format {{key_name}}
    """
    try:
        # Parse the JSON string
        data = json.loads(json_str)
        
        # Create template string with variables as key names
        template = ""
        for key in data:
            template += f"{key}: {{{{{key}}}}}\n"
        
        return template.strip()
    except json.JSONDecodeError:
        return "Error: Invalid JSON string"

        
def fill_template(template, dict_to_fill):
    for key, value in dict_to_fill.items():
        template = template.replace(f"{{{{{key}}}}}", str(value))
    return template
    
    
if __name__ == "__main__":
    print(parse_json_to_template(feature_json_object))
    # Fill in the template with the feature_json_object
    dict_to_fill = {
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
    template = parse_json_to_template(feature_json_object)
    # Battery life is trash
    dict_to_fill["battery_life"] = 3
    
    
    
    filled_template = fill_template(template, dict_to_fill)
    print(filled_template)
    
    
